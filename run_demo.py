"""
run_demo.py — Research-Grade Habitat Navigation Visualization
=============================================================

This script demonstrates a complete embodied AI navigation pipeline using
habitat-sim v0.3.3 and habitat-lab. It produces:

  1. A 300-step navigation loop with the agent moving forward.
  2. An egocentric RGB video (agent's first-person view) saved as an MP4.
  3. Periodic snapshot images from the agent's camera.
  4. A high-quality "God's Eye" top-down trajectory map rendered with
     matplotlib, showing the agent's path overlaid on the navigable floor plan.
  5. Optional semantic overlay on the top-down map (if semantic data exists).

Coordinate System Notes:
  - Habitat uses a right-handed coordinate system:
      x → right,  y → up,  z → backward (into the screen).
  - The top-down map is generated from the pathfinder's navmesh at the
    agent's current height (y-coordinate).
  - maps.to_grid(realworld_x, realworld_y) internally maps:
      realworld_x → agent.z  (uses lower_bound[2], the z-axis bound)
      realworld_y → agent.x  (uses lower_bound[0], the x-axis bound)
    So you must call:  to_grid(agent.z, agent.x)
  - maps.draw_path() draws on an RGB image using color indices from
    TOP_DOWN_MAP_COLORS (index 10 = blue, index 7 = green, etc.).

Requirements:
  - habitat-sim (installed or at libs/habitat-sim-main)
  - habitat-lab (at libs/habitat-lab-main)
  - Scene datasets in data/scene_datasets/
"""

# ============================================================================
# 1. Path Setup — must come before any habitat imports
# ============================================================================
import setup_path

# ============================================================================
# 2. Imports
# ============================================================================
import os
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt

import habitat_sim
from habitat.utils.visualizations import maps
from habitat_sim.utils.common import quat_from_two_vectors

# Import our dataset selector
from dataset_selector import (
    discover_datasets,
    select_dataset,
    discover_scenes,
    select_scene,
    resolve_assets,
    print_scene_config,
    SceneConfig,
    parse_semantic_txt,
    SemanticClass,
)


# ============================================================================
# 3. Constants
# ============================================================================
NUM_STEPS = 360
VIDEO_FPS = 48
SNAPSHOT_INTERVAL = 50       # Save a camera snapshot every N steps
MAP_RESOLUTION = 2048         # Resolution for the top-down map (lower for HM3D compatibility)
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1440
TOPDOWN_RENDER_RESOLUTION = 3072
STUCK_DISTANCE_EPS = 1e-3
STUCK_TURN_INTERVAL = 5
MIN_GOAL_DISTANCE = 2.5
MAX_GOAL_SAMPLE_ATTEMPTS = 48
SPAWN_CLEARANCE = 0.75
GOAL_CLEARANCE = 0.45
RECENT_GOAL_MEMORY = 4
MIN_RECENT_GOAL_SEPARATION = 2.0
TOPDOWN_CROP_PADDING_PX = 96

# --- Output directory: timestamp-based ---
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = os.path.join("Output", RUN_TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "egocentric_navigation.mp4")
OUTPUT_TRAJECTORY = os.path.join(OUTPUT_DIR, "final_trajectory_plot.png")
SNAPSHOT_PATTERN = os.path.join(OUTPUT_DIR, "snapshot_step_{step:03d}.png")


# ============================================================================
# 4. Simulator Factory
# ============================================================================
def create_simulator(scene_config: SceneConfig) -> habitat_sim.Simulator:
    """
    Create and return a habitat_sim.Simulator configured for CPU rendering.

    Uses the dynamically selected scene path from SceneConfig.
    The agent is equipped with an RGB camera sensor so we can capture
    egocentric video frames and snapshots.
    """
    scene_path_str = str(scene_config.render_asset_path.resolve())

    # --- Backend configuration ---
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path_str
    sim_cfg.gpu_device_id = -1  # CPU-only rendering
    if scene_config.dataset_config_path:
        sim_cfg.scene_dataset_config_file = str(
            scene_config.dataset_config_path.resolve()
        )

    # --- RGB camera sensor specification ---
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [VIDEO_HEIGHT, VIDEO_WIDTH]
    rgb_sensor_spec.position = [0.0, 0.88, 0.0]  # Approximate eye height
    rgb_sensor_spec.orientation = [0.0, 0.0, 0.0]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb_sensor_spec.hfov = 75.0

    topdown_sensor_spec = habitat_sim.CameraSensorSpec()
    topdown_sensor_spec.uuid = "topdown_color_sensor"
    topdown_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    topdown_sensor_spec.resolution = [
        TOPDOWN_RENDER_RESOLUTION,
        TOPDOWN_RENDER_RESOLUTION,
    ]
    topdown_sensor_spec.position = [0.0, 0.0, 0.0]
    topdown_sensor_spec.orientation = [0.0, 0.0, 0.0]
    topdown_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    topdown_sensor_spec.ortho_scale = 0.1
    topdown_sensor_spec.near = 0.1
    topdown_sensor_spec.far = 256.0

    # --- Agent configuration ---
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_spec, topdown_sensor_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.ActionSpec(
            "move_forward", habitat_sim.ActuationSpec(amount=0.08)
        ),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left", habitat_sim.ActuationSpec(amount=8.0)
        ),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right", habitat_sim.ActuationSpec(amount=8.0)
        ),
    }

    # --- Combine and instantiate ---
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    try:
        sim = habitat_sim.Simulator(cfg)
    except Exception as e:
        print(f"[ERROR] Failed to create simulator with scene: {scene_path_str}")
        print(f"  {e}")
        sys.exit(1)

    if scene_config.navmesh_path:
        try:
            navmesh_ok = sim.pathfinder.load_nav_mesh(
                str(scene_config.navmesh_path.resolve())
            )
            if not navmesh_ok:
                print(
                    f"[WARNING] Failed to load navmesh explicitly: {scene_config.navmesh_path}"
                )
        except Exception as e:
            print(f"[WARNING] Could not load navmesh {scene_config.navmesh_path}: {e}")

    if sim.pathfinder.is_loaded:
        print("Navmesh loaded successfully.")
    else:
        print("[WARNING] Pathfinder has no loaded navmesh.")

    return sim


# ============================================================================
# 5. NavigationVisualizer Class
# ============================================================================
class NavigationVisualizer:
    """
    Orchestrates the simulation loop, video recording, snapshot capture,
    and top-down trajectory visualization.
    """

    def __init__(self, sim: habitat_sim.Simulator, scene_config: SceneConfig):
        self.sim = sim
        self.scene_config = scene_config
        self.agent = sim.initialize_agent(0)
        self.follower: Optional[habitat_sim.GreedyGeodesicFollower] = None
        self.nav_island_index: int = -1
        self.topdown_render_metadata: Optional[Dict[str, float]] = None
        self.recent_goal_positions: List[np.ndarray] = []

        # Storage for the agent's world positions (x, y, z) at each step
        self.path_history: List[Tuple[float, float, float]] = []
        self._place_agent_on_navmesh()
        self._configure_follower()

        # Video writer for egocentric RGB footage
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            OUTPUT_VIDEO, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT)
        )

    # ------------------------------------------------------------------
    def _place_agent_on_navmesh(self) -> None:
        """Spawn the agent from a valid navigable point when a navmesh exists."""
        if not self.sim.pathfinder.is_loaded:
            state = self.agent.get_state()
            pos = state.position
            self.path_history.append((float(pos[0]), float(pos[1]), float(pos[2])))
            print("[WARNING] Using simulator default start pose because navmesh is unavailable.")
            return

        self.nav_island_index = self._select_navigation_island()
        start_pos = self._sample_nav_point(
            island_index=self.nav_island_index,
            min_clearance=SPAWN_CLEARANCE,
            attempts=MAX_GOAL_SAMPLE_ATTEMPTS,
        )
        state = self.agent.get_state()
        state.position = start_pos
        self.agent.set_state(state)
        self.path_history.append(
            (float(start_pos[0]), float(start_pos[1]), float(start_pos[2]))
        )
        print(f"Agent start position: {np.round(start_pos, 3).tolist()}")
        print(f"Agent island index: {self.nav_island_index}")

    # ------------------------------------------------------------------
    def _select_navigation_island(self) -> int:
        """Pick the largest navigable island to avoid tiny disconnected pockets."""
        if not self.sim.pathfinder.is_loaded or self.sim.pathfinder.num_islands <= 0:
            return -1

        best_island = 0
        best_area = float("-inf")
        for island_index in range(self.sim.pathfinder.num_islands):
            island_area = float(self.sim.pathfinder.island_area(island_index))
            if island_area > best_area:
                best_area = island_area
                best_island = island_index
        return best_island

    # ------------------------------------------------------------------
    def _sample_nav_point(
        self,
        island_index: int = -1,
        min_clearance: float = 0.0,
        attempts: int = 32,
    ) -> np.ndarray:
        """Sample a navigable point, preferring points with some obstacle clearance."""
        fallback_point: Optional[np.ndarray] = None

        for _ in range(attempts):
            if island_index >= 0:
                point = self.sim.pathfinder.get_random_navigable_point(
                    island_index=island_index
                )
            else:
                point = self.sim.pathfinder.get_random_navigable_point()
            point = np.array(point, dtype=np.float32)

            if fallback_point is None:
                fallback_point = point

            if min_clearance <= 0.0:
                return point

            clearance = float(
                self.sim.pathfinder.distance_to_closest_obstacle(
                    point, max_search_radius=2.0
                )
            )
            if clearance >= min_clearance:
                return point

        if fallback_point is not None:
            return fallback_point

        raise RuntimeError("Failed to sample a navigable point.")

    # ------------------------------------------------------------------
    def _configure_follower(self) -> None:
        """Create a path follower that walks the agent along the navmesh."""
        if not self.sim.pathfinder.is_loaded:
            return

        self.follower = habitat_sim.GreedyGeodesicFollower(
            self.sim.pathfinder,
            self.agent,
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )

    # ------------------------------------------------------------------
    def _is_goal_novel_enough(self, goal: np.ndarray) -> bool:
        """Avoid immediately re-sampling goals that are too close to recent ones."""
        goal = np.array(goal, dtype=np.float32)
        for recent_goal in self.recent_goal_positions[-RECENT_GOAL_MEMORY:]:
            if float(np.linalg.norm(goal - recent_goal)) < MIN_RECENT_GOAL_SEPARATION:
                return False
        return True

    # ------------------------------------------------------------------
    def _sample_navigation_goal(
        self,
    ) -> Optional[Tuple[np.ndarray, habitat_sim.ShortestPath]]:
        """Sample a reachable goal that is far enough from the current pose."""
        if not self.sim.pathfinder.is_loaded:
            raise RuntimeError("Cannot sample a navigation goal without a loaded navmesh.")

        start = self.agent.get_state().position
        fallback_path: Optional[habitat_sim.ShortestPath] = None
        fallback_goal: Optional[np.ndarray] = None

        for _ in range(MAX_GOAL_SAMPLE_ATTEMPTS):
            goal = self._sample_nav_point(
                island_index=self.nav_island_index,
                min_clearance=GOAL_CLEARANCE,
                attempts=8,
            )
            if not self._is_goal_novel_enough(goal):
                continue
            path = habitat_sim.ShortestPath()
            path.requested_start = start
            path.requested_end = goal
            if not self.sim.pathfinder.find_path(path):
                continue
            if path.geodesic_distance >= MIN_GOAL_DISTANCE:
                return goal, path
            if fallback_path is None:
                fallback_path = path
                fallback_goal = goal

        if fallback_path is not None and fallback_goal is not None:
            return fallback_goal, fallback_path

        return None

    # ------------------------------------------------------------------
    def _bounded_meters_per_pixel(self, longest_side_pixels: int) -> float:
        """
        Compute meters-per-pixel by constraining the longest map side.

        Habitat's default helper constrains the shortest side, which can explode
        memory usage for long, thin HM3D floorplans.
        """
        lower_bound, upper_bound = self.sim.pathfinder.get_bounds()
        span_x = abs(float(upper_bound[0]) - float(lower_bound[0]))
        span_z = abs(float(upper_bound[2]) - float(lower_bound[2]))
        longest_span = max(span_x, span_z)
        if longest_span <= 0:
            raise RuntimeError("Pathfinder returned invalid bounds for top-down map generation.")
        return longest_span / float(longest_side_pixels)

    # ------------------------------------------------------------------
    def _compute_topdown_render_metadata(self) -> Dict[str, float]:
        """Compute the shared square footprint for top-down rendering and overlay."""
        lower_bound, upper_bound = self.sim.pathfinder.get_bounds()
        span_x = abs(float(upper_bound[0]) - float(lower_bound[0]))
        span_z = abs(float(upper_bound[2]) - float(lower_bound[2]))
        longest_span = max(span_x, span_z)
        if longest_span <= 0:
            raise RuntimeError("Pathfinder returned invalid bounds for rendered top-down capture.")

        padding_ratio = 0.08
        square_span = max(longest_span * (1.0 + 2.0 * padding_ratio), 1.0)
        center_x = float((lower_bound[0] + upper_bound[0]) / 2.0)
        center_z = float((lower_bound[2] + upper_bound[2]) / 2.0)
        min_x = center_x - square_span / 2.0
        max_x = center_x + square_span / 2.0
        min_z = center_z - square_span / 2.0
        max_z = center_z + square_span / 2.0
        camera_height = float(upper_bound[1]) + max(5.0, square_span * 0.35)
        far_plane_dist = max(64.0, camera_height - float(lower_bound[1]) + 5.0)

        return {
            "center_x": center_x,
            "center_z": center_z,
            "square_span": square_span,
            "min_x": min_x,
            "max_x": max_x,
            "min_z": min_z,
            "max_z": max_z,
            "camera_height": camera_height,
            "ortho_scale": 1.0 / square_span,
            "far_plane_dist": far_plane_dist,
        }

    # ------------------------------------------------------------------
    def _world_to_render_pixel(
        self,
        x: float,
        z: float,
        image_shape: Tuple[int, int, int],
    ) -> Tuple[int, int]:
        """Project a world-space x/z point into rendered top-down pixel coordinates."""
        if self.topdown_render_metadata is None:
            raise RuntimeError("Top-down render metadata is unavailable.")

        h, w = image_shape[:2]
        meta = self.topdown_render_metadata
        u = (x - meta["min_x"]) / max(meta["max_x"] - meta["min_x"], 1e-6)
        v = (z - meta["min_z"]) / max(meta["max_z"] - meta["min_z"], 1e-6)

        col = int(round(np.clip(u, 0.0, 1.0) * (w - 1)))
        row = int(round(np.clip(v, 0.0, 1.0) * (h - 1)))
        return col, row

    # ------------------------------------------------------------------
    def _path_to_render_pixels(
        self, image_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, int]]:
        """Project the recorded trajectory into rendered top-down pixels."""
        return [
            self._world_to_render_pixel(x, z, image_shape)
            for x, _, z in self.path_history
        ]

    # ------------------------------------------------------------------
    def _enhance_rgb_frame(self, image: np.ndarray) -> np.ndarray:
        """Apply mild contrast and sharpening to improve perceived clarity."""
        if image.ndim != 3 or image.shape[2] != 3:
            return image

        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0, sigmaY=1.0)
        sharpened = cv2.addWeighted(image, 1.35, blurred, -0.35, 0)

        lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    # ------------------------------------------------------------------
    def _replace_border_black_with_white(self, image: np.ndarray) -> np.ndarray:
        """
        Turn only the outer black render background white.

        We flood-fill from the image borders so genuinely dark interior content
        is preserved as much as possible.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            return image

        near_black_mask = np.all(image <= 8, axis=2).astype(np.uint8) * 255
        flood_mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
        border_mask = near_black_mask.copy()

        for seed in (
            (0, 0),
            (image.shape[1] - 1, 0),
            (0, image.shape[0] - 1),
            (image.shape[1] - 1, image.shape[0] - 1),
        ):
            if border_mask[seed[1], seed[0]] == 255:
                cv2.floodFill(border_mask, flood_mask, seedPoint=seed, newVal=128)

        whitened = image.copy()
        whitened[border_mask == 128] = 255
        return whitened

    # ------------------------------------------------------------------
    def _crop_rendered_topdown(
        self,
        image: np.ndarray,
        path_pixels: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Crop excess white border so the rendered home occupies more of the figure."""
        content_mask = np.any(image < 248, axis=2)
        if not np.any(content_mask):
            return image, path_pixels

        ys, xs = np.where(content_mask)
        min_x = int(xs.min())
        max_x = int(xs.max())
        min_y = int(ys.min())
        max_y = int(ys.max())

        if path_pixels:
            path_xs = [px for px, _ in path_pixels]
            path_ys = [py for _, py in path_pixels]
            min_x = min(min_x, min(path_xs))
            max_x = max(max_x, max(path_xs))
            min_y = min(min_y, min(path_ys))
            max_y = max(max_y, max(path_ys))

        min_x = max(0, min_x - TOPDOWN_CROP_PADDING_PX)
        min_y = max(0, min_y - TOPDOWN_CROP_PADDING_PX)
        max_x = min(image.shape[1] - 1, max_x + TOPDOWN_CROP_PADDING_PX)
        max_y = min(image.shape[0] - 1, max_y + TOPDOWN_CROP_PADDING_PX)

        cropped = image[min_y : max_y + 1, min_x : max_x + 1]
        shifted_pixels = [(px - min_x, py - min_y) for px, py in path_pixels]
        return cropped, shifted_pixels

    # ------------------------------------------------------------------
    def capture_rendered_topdown_view(self) -> np.ndarray:
        """
        Render a true bird's-eye RGB view of the scene from above.

        Unlike the navmesh map, this view contains the actual room and furniture
        appearance from the scene geometry.
        """
        saved_state = self.agent.get_state()
        render_meta = self._compute_topdown_render_metadata()
        self.topdown_render_metadata = render_meta

        topdown_sensor = self.agent.scene_node.subtree_sensors["topdown_color_sensor"]
        topdown_sensor.camera_type = habitat_sim.SensorSubType.ORTHOGRAPHIC
        topdown_sensor.near_plane_dist = 0.1
        topdown_sensor.far_plane_dist = render_meta["far_plane_dist"]
        topdown_sensor.reset_zoom()
        topdown_sensor.zoom(render_meta["ortho_scale"] / 0.1)

        render_state = habitat_sim.AgentState()
        render_state.position = np.array(
            [
                render_meta["center_x"],
                render_meta["camera_height"],
                render_meta["center_z"],
            ],
            dtype=np.float32,
        )
        render_state.rotation = quat_from_two_vectors(
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            np.array([0.0, -1.0, 0.0], dtype=np.float32),
        )
        self.agent.set_state(render_state, infer_sensor_states=True)

        observations = self.sim.get_sensor_observations()
        render_rgba = observations["topdown_color_sensor"]
        render_rgb = np.ascontiguousarray(render_rgba[:, :, :3])
        render_rgb = self._replace_border_black_with_white(render_rgb)
        render_rgb = self._enhance_rgb_frame(render_rgb)

        self.agent.set_state(saved_state, infer_sensor_states=True)
        return render_rgb

    # ------------------------------------------------------------------
    def run_navigation_loop(self) -> None:
        """
        Execute a 300-step navigation loop:
          - Move forward at every step.
          - Turn right every 20 steps to create a curved exploration path.
          - Record the agent's position at every step.
          - Capture RGB frames for the video at every step.
          - Save snapshot images periodically.
        """
        print(f"Starting {NUM_STEPS}-step navigation loop ...")
        start_time = time.time()
        stuck_steps = 0
        current_goal: Optional[np.ndarray] = None
        action_queue: List[Optional[str]] = []
        last_action: Optional[str] = None

        for step in range(NUM_STEPS):
            if self.follower is not None and not action_queue:
                goal_sample = self._sample_navigation_goal()
                if goal_sample is None:
                    print(f"  [WARN] No reachable goal found at step {step:03d}; stopping early.")
                    break
                current_goal, shortest_path = goal_sample
                current_goal = np.array(current_goal, dtype=np.float32)
                try:
                    action_queue = list(self.follower.find_path(current_goal))
                except habitat_sim.errors.GreedyFollowerError:
                    action_queue = []
                    current_goal = None
                else:
                    self.recent_goal_positions.append(np.array(current_goal, dtype=np.float32))
                    if len(self.recent_goal_positions) > RECENT_GOAL_MEMORY:
                        self.recent_goal_positions = self.recent_goal_positions[-RECENT_GOAL_MEMORY:]
                    print(
                        f"  New goal @ step {step:03d}: "
                        f"{np.round(current_goal, 3).tolist()} "
                        f"(geodesic {shortest_path.geodesic_distance:.2f} m, "
                        f"{max(len(action_queue) - 1, 0)} actions)"
                    )

            prev_pos = self.agent.get_state().position.copy()

            # --- Execute actions ---
            if action_queue:
                next_action = action_queue.pop(0)
                if next_action is not None:
                    self.agent.act(next_action)
                last_action = next_action
            else:
                self.agent.act("move_forward")
                last_action = "move_forward"
                if step % 20 == 0:
                    self.agent.act("turn_right")

            # --- Record position ---
            state = self.agent.get_state()
            pos = state.position  # numpy array: (x, y, z)
            displacement = float(np.linalg.norm(pos - prev_pos))
            if last_action == "move_forward" and displacement < STUCK_DISTANCE_EPS:
                stuck_steps += 1
                if stuck_steps % STUCK_TURN_INTERVAL == 0:
                    if self.follower is None:
                        self.agent.act("turn_left")
                        state = self.agent.get_state()
                        pos = state.position
                    else:
                        action_queue = []
                        current_goal = None
            else:
                stuck_steps = 0
            self.path_history.append((float(pos[0]), float(pos[1]), float(pos[2])))

            # --- Capture RGB observation for video ---
            obs = self.sim.get_sensor_observations()
            # The color sensor returns an RGBA image; take only RGB
            frame_rgba = obs["color_sensor"]
            frame_rgb = self._enhance_rgb_frame(np.ascontiguousarray(frame_rgba[:, :, :3]))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)

            # --- Save periodic snapshots ---
            if step % SNAPSHOT_INTERVAL == 0:
                snapshot_path = SNAPSHOT_PATTERN.format(step=step)
                cv2.imwrite(snapshot_path, frame_bgr)
                print(f"  Snapshot saved: {snapshot_path}")

            # --- Progress indicator ---
            if (step + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Step {step + 1}/{NUM_STEPS}  ({elapsed:.1f}s elapsed)")

        elapsed = time.time() - start_time
        print(f"Navigation loop complete.  Total time: {elapsed:.1f}s")
        print(f"  Average step rate: {NUM_STEPS / elapsed:.1f} steps/s")

    # ------------------------------------------------------------------
    def finalize_video(self) -> None:
        """Release the video writer."""
        self.video_writer.release()
        print(f"Egocentric video saved to '{OUTPUT_VIDEO}'.")

    # ------------------------------------------------------------------
    def generate_topdown_trajectory(
        self, map_resolution: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate a high-quality top-down map with the agent's path overlaid.

        Pipeline:
          1. Get the agent's final height (y-coordinate) for the map slice.
          2. Call maps.get_topdown_map() to obtain a raw occupancy grid
             (0 = occupied, 1 = free, 2 = border).
          3. Colorize the raw map into an RGB image using the built-in
             TOP_DOWN_MAP_COLORS lookup table.
          4. Convert each (x, z) world-coordinate waypoint into pixel
             (grid) coordinates using maps.to_grid().
          5. Draw the path onto the colorized map with maps.draw_path().

        Args:
            map_resolution: Resolution for the top-down map.
                            Defaults to the module-level MAP_RESOLUTION.

        Returns:
            The colorized map (H x W x 3 uint8 RGB array) with the path drawn.
        """
        if map_resolution is None:
            map_resolution = MAP_RESOLUTION

        print(f"Generating top-down trajectory map (resolution={map_resolution}) ...")
        if not self.sim.pathfinder.is_loaded:
            raise RuntimeError("Cannot render a top-down map because the navmesh was not loaded.")

        # --- 5a. Determine the agent's height for the map slice ---
        # Use the final position's y-coordinate so we map the correct floor.
        agent_height = self.path_history[-1][1]
        meters_per_pixel = self._bounded_meters_per_pixel(map_resolution)

        # --- 5b. Get the raw occupancy map from the pathfinder ---
        raw_map = maps.get_topdown_map(
            pathfinder=self.sim.pathfinder,
            height=agent_height,
            map_resolution=map_resolution,
            draw_border=True,
            meters_per_pixel=meters_per_pixel,
        )
        # raw_map shape: (H, W) with values 0, 1, 2

        # --- 5c. Colorize the raw map into an RGB image ---
        # TOP_DOWN_MAP_COLORS maps each integer index to an RGB triplet.
        colorized = maps.colorize_topdown_map(raw_map)
        # colorized shape: (H, W, 3), dtype=uint8, RGB order

        # --- 5d. Convert world-coordinate path to grid (pixel) coordinates ---
        # maps.to_grid(realworld_x, realworld_y, grid_resolution, pathfinder)
        #
        # IMPORTANT: Looking at the to_grid implementation:
        #   grid_x = int((realworld_x - lower_bound[2]) / grid_size[0])
        #   grid_y = int((realworld_y - lower_bound[0]) / grid_size[1])
        # where lower_bound[2] is the z-axis bound and lower_bound[0] is the
        # x-axis bound. So:
        #   realworld_x must be agent.z
        #   realworld_y must be agent.x
        grid_path: List[Tuple[int, int]] = []
        for x, y, z in self.path_history:
            gx, gy = maps.to_grid(
                realworld_x=z,       # agent.z → maps to grid_x (z-axis)
                realworld_y=x,       # agent.x → maps to grid_y (x-axis)
                grid_resolution=colorized.shape[:2],  # (H, W)
                pathfinder=self.sim.pathfinder,
            )
            grid_path.append((gx, gy))

        # --- 5e. Draw the path on the colorized map ---
        # draw_path modifies the map in-place.
        # Color index 10 corresponds to a vivid blue in TOP_DOWN_MAP_COLORS.
        maps.draw_path(
            top_down_map=colorized,
            path_points=grid_path,
            color=10,       # Blue path
            thickness=3,
        )

        print(f"  Path has {len(grid_path)} waypoints mapped to grid.")
        return colorized

    # ------------------------------------------------------------------
    def render_paper_grade_figure(
        self,
        colorized_map: np.ndarray,
        rendered_topdown: Optional[np.ndarray],
    ) -> None:
        """
        Render a publication-quality figure using matplotlib.

        This adds:
          - A clean axis with proper aspect ratio.
          - A colorbar legend explaining the map colors.
          - A title and axis labels.
          - The agent's start and end positions highlighted.
          - A scale bar (optional, based on meters_per_pixel).
          - Semantic overlay if semantic data is available.

        The result is saved to OUTPUT_TRAJECTORY.
        """
        print("Rendering paper-grade figure ...")

        scene_label = f"{self.scene_config.dataset_name}/{self.scene_config.scene_name}"

        if rendered_topdown is not None and self.topdown_render_metadata is not None:
            path_pixels = self._path_to_render_pixels(rendered_topdown.shape)
            rendered_topdown, path_pixels = self._crop_rendered_topdown(
                rendered_topdown, path_pixels
            )
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax.imshow(rendered_topdown, interpolation="nearest")

            xs = [px for px, _ in path_pixels]
            ys = [py for _, py in path_pixels]
            start_x, start_y = path_pixels[0]
            end_x, end_y = path_pixels[-1]

            ax.plot(
                xs,
                ys,
                color="#1b0a0a",
                linewidth=3.2,
                alpha=0.95,
                label="Trajectory",
            )
            ax.plot(
                start_x,
                start_y,
                "o",
                color="lime",
                markersize=14,
                markeredgecolor="white",
                markeredgewidth=2,
                label="Start",
            )
            ax.plot(
                end_x,
                end_y,
                "*",
                color="red",
                markersize=18,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="End",
            )

            h, w = rendered_topdown.shape[:2]
            scale_bar_meters = 5.0
            mpp = self.topdown_render_metadata["square_span"] / float(max(h, w))
            scale_bar_pixels = scale_bar_meters / max(mpp, 1e-6)
            bar_x = int(w * 0.05)
            bar_y = int(h * 0.95)
            ax.plot(
                [bar_x, bar_x + scale_bar_pixels],
                [bar_y, bar_y],
                color="white",
                linewidth=4,
            )
            ax.text(
                bar_x + scale_bar_pixels / 2,
                bar_y - 16,
                f"{scale_bar_meters:.0f} m",
                color="white",
                fontsize=10,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

            ax.set_title(
                f"Rendered Top-Down Trajectory Overlay ({scene_label})",
                fontsize=16,
                fontweight="bold",
            )
            ax.axis("off")
            ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
            plt.tight_layout()
            fig.savefig(OUTPUT_TRAJECTORY, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Paper-grade trajectory plot saved to '{OUTPUT_TRAJECTORY}'.")
            return

        # New output: combine a true bird's-eye scene render with the precise
        # navmesh trajectory map when possible. If the rendered top-down view
        # fails, still save the trajectory map instead of aborting the whole run.
        mpp = self._bounded_meters_per_pixel(max(colorized_map.shape[:2]))
        if rendered_topdown is not None:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(16, 10),
                gridspec_kw={"width_ratios": [1.1, 1.0]},
            )
            ax_render, ax_map = axes
            ax_render.imshow(rendered_topdown, interpolation="nearest")
            ax_render.set_title("Rendered Top-Down View", fontsize=14, fontweight="bold")
            ax_render.axis("off")
            fig.suptitle("Habitat Navigation Summary", fontsize=20, fontweight="bold")
        else:
            fig, ax_map = plt.subplots(1, 1, figsize=(12, 10))

        ax_map.imshow(colorized_map, interpolation="nearest")

        start_x, _, start_z = self.path_history[0]
        end_x, _, end_z = self.path_history[-1]
        gsx, gsy = maps.to_grid(
            start_z, start_x, colorized_map.shape[:2], pathfinder=self.sim.pathfinder
        )
        gex, gey = maps.to_grid(
            end_z, end_x, colorized_map.shape[:2], pathfinder=self.sim.pathfinder
        )

        ax_map.plot(
            gsy, gsx, "o",
            color="lime",
            markersize=14,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Start",
        )
        ax_map.plot(
            gey, gex, "*",
            color="red",
            markersize=18,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="End",
        )

        scale_bar_meters = 5.0
        scale_bar_pixels = scale_bar_meters / mpp
        h, w = colorized_map.shape[:2]
        bar_x = int(w * 0.05)
        bar_y = int(h * 0.95)
        ax_map.plot(
            [bar_x, bar_x + scale_bar_pixels],
            [bar_y, bar_y],
            color="white",
            linewidth=4,
        )
        ax_map.text(
            bar_x + scale_bar_pixels / 2,
            bar_y - 10,
            f"{scale_bar_meters:.0f} m",
            color="white",
            fontsize=10,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

        scene_label = f"{self.scene_config.dataset_name}/{self.scene_config.scene_name}"
        ax_map.set_title(
            f"Trajectory on Navigability Map ({scene_label})",
            fontsize=14,
            fontweight="bold",
        )
        ax_map.set_xlabel("Grid X (pixels)", fontsize=12)
        ax_map.set_ylabel("Grid Y (pixels)", fontsize=12)
        ax_map.legend(loc="upper right", fontsize=11, framealpha=0.9)
        ax_map.tick_params(labelbottom=False, labelleft=False)

        if rendered_topdown is not None:
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            plt.tight_layout()
        fig.savefig(OUTPUT_TRAJECTORY, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Paper-grade trajectory plot saved to '{OUTPUT_TRAJECTORY}'.")
        return

        # Compute meters-per-pixel for the scale bar
        mpp = self._bounded_meters_per_pixel(max(colorized_map.shape[:2]))

        # Create the figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Display the colorized map
        ax.imshow(colorized_map, interpolation="nearest")

        # --- Overlay semantic information if available ---
        if self.scene_config.semantic_txt_path:
            self._overlay_semantic_legend(ax, colorized_map)

        # --- Overlay start and end markers ---
        # Convert the first and last world positions to grid coordinates.
        # Same convention: to_grid(agent.z, agent.x)
        start_x, start_y, start_z = self.path_history[0]
        end_x, end_y, end_z = self.path_history[-1]
        gsx, gsy = maps.to_grid(
            start_z, start_x, colorized_map.shape[:2],
            pathfinder=self.sim.pathfinder
        )
        gex, gey = maps.to_grid(
            end_z, end_x, colorized_map.shape[:2],
            pathfinder=self.sim.pathfinder
        )

        # Start: green circle
        ax.plot(gsy, gsx, "o", color="lime", markersize=14, markeredgecolor="white",
                markeredgewidth=2, label="Start")
        # End: red star
        ax.plot(gey, gex, "*", color="red", markersize=18, markeredgecolor="white",
                markeredgewidth=1.5, label="End")

        # --- Add a scale bar ---
        scale_bar_meters = 5.0
        scale_bar_pixels = scale_bar_meters / mpp
        h, w = colorized_map.shape[:2]
        bar_x = int(w * 0.05)   # 5% from left edge
        bar_y = int(h * 0.95)   # 95% from top (near bottom)
        ax.plot([bar_x, bar_x + scale_bar_pixels], [bar_y, bar_y],
                color="white", linewidth=4)
        ax.text(bar_x + scale_bar_pixels / 2, bar_y - 10,
                f"{scale_bar_meters:.0f} m",
                color="white", fontsize=10, ha="center", va="bottom",
                fontweight="bold")

        # --- Labels and title ---
        scene_label = f"{self.scene_config.dataset_name}/{self.scene_config.scene_name}"
        ax.set_title(f"Agent Trajectory — Top-Down View ({scene_label})",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Grid X (pixels)", fontsize=12)
        ax.set_ylabel("Grid Y (pixels)", fontsize=12)
        ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

        # Remove tick labels (they are pixel indices, not meaningful)
        ax.tick_params(labelbottom=False, labelleft=False)

        # Tight layout
        plt.tight_layout()

        # Save
        fig.savefig(OUTPUT_TRAJECTORY, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Paper-grade trajectory plot saved to '{OUTPUT_TRAJECTORY}'.")

    # ------------------------------------------------------------------
    def _overlay_semantic_legend(
        self, ax: plt.Axes, colorized_map: np.ndarray
    ) -> None:
        """
        Parse semantic data and add a color-coded legend to the figure.

        Groups semantic classes by floor level and shows the most common
        structural classes (wall, floor, ceiling, door, window, etc.)
        with their corresponding colors.
        """
        txt_path = self.scene_config.semantic_txt_path
        if not txt_path or not txt_path.exists():
            return

        classes = parse_semantic_txt(txt_path)
        if not classes:
            return

        # Build a summary of structural classes (most relevant for a map)
        # We'll show unique class names with their representative colors
        structural_keywords = [
            "wall", "floor", "ceiling", "door", "window",
            "stairs", "balustrade", "pillar", "column"
        ]

        # Get unique structural classes with their first-seen color
        seen_classes: dict = {}
        for c in classes:
            name_lower = c.name.lower()
            if any(kw in name_lower for kw in structural_keywords):
                if name_lower not in seen_classes:
                    seen_classes[name_lower] = c

        if not seen_classes:
            # Fallback: show first 8 unique classes
            for c in classes:
                if c.name not in seen_classes:
                    seen_classes[c.name] = c
                if len(seen_classes) >= 8:
                    break

        # Create legend handles
        from matplotlib.patches import Patch
        legend_handles = []
        for name, sc in sorted(seen_classes.items()):
            r, g, b = sc.color_rgb
            patch = Patch(
                color=(r / 255, g / 255, b / 255),
                label=f"{sc.name} (floor {sc.floor})"
            )
            legend_handles.append(patch)

        if legend_handles:
            # Add a secondary legend for semantic classes
            legend2 = ax.legend(
                handles=legend_handles,
                loc="lower left",
                fontsize=8,
                framealpha=0.85,
                title="Semantic Classes",
                title_fontsize=9,
            )
            ax.add_artist(legend2)
            print(f"  Added semantic legend with {len(legend_handles)} classes")

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Clean up the simulator."""
        self.sim.close()
        print("Simulator closed.")


# ============================================================================
# 6. Dataset & Scene Selection
# ============================================================================

def select_dataset_interactive() -> SceneConfig:
    """
    Interactive workflow:
      1. Discover available datasets
      2. Let user select one
      3. Discover scenes in that dataset
      4. Let user select a scene
      5. Resolve all companion assets (navmesh, semantic)
      6. Return a SceneConfig
    """
    print("=" * 60)
    print("  Habitat Navigation Demo — Research-Grade Visualization")
    print("=" * 60)

    # Step 1: Discover datasets
    datasets = discover_datasets()
    if not datasets:
        print("[ERROR] No datasets found in data/scene_datasets/")
        print("  Make sure your scene data is placed in:")
        print("    data/scene_datasets/<dataset_name>/")
        sys.exit(1)

    # Step 2: Select dataset
    ds_path = select_dataset(datasets)
    if ds_path is None:
        print("Exiting.")
        sys.exit(0)

    # Step 3: Discover scenes
    scenes = discover_scenes(ds_path)
    if not scenes:
        print(f"[ERROR] No .glb scene files found in {ds_path}")
        sys.exit(1)

    # Step 4: Select scene
    scene_path = select_scene(scenes)
    if scene_path is None:
        print("Exiting.")
        sys.exit(0)

    # Step 5: Resolve all assets
    config = resolve_assets(scene_path)
    print_scene_config(config)

    # Warn if navmesh is missing (but don't crash)
    if config.navmesh_path is None:
        print("\n[WARNING] No .navmesh file found for this scene.")
        print("  The pathfinder may not work correctly.")
        print("  Navigation will proceed but the top-down map may be empty.\n")

    return config


# ============================================================================
# 7. Main Entry Point
# ============================================================================
def main():
    """
    Full pipeline:
      0. Interactive dataset/scene selection.
      1. Create the simulator.
      2. Run the navigation loop (video + snapshots + path recording).
      3. Finalize the egocentric video.
      4. Generate the top-down trajectory map.
      5. Render a publication-quality figure.
      6. Clean up.
    """
    # --- Step 0: Select dataset and scene ---
    scene_config = select_dataset_interactive()

    # --- Step 1: Create simulator ---
    sim = create_simulator(scene_config)

    # --- Step 2: Create the visualizer and run ---
    viz = NavigationVisualizer(sim, scene_config)
    viz.run_navigation_loop()

    # --- Step 3: Finalize video ---
    viz.finalize_video()

    # --- Step 4: Generate top-down trajectory map ---
    # HM3D scenes are large and may cause memory issues in WSL.
    # Try progressively lower resolutions until one works.
    resolutions_to_try = [MAP_RESOLUTION, 512, 256, 128, 64]
    colorized_map = None

    for map_resolution in resolutions_to_try:
        try:
            colorized_map = viz.generate_topdown_trajectory(map_resolution=map_resolution)
            break  # Success!
        except MemoryError as e:
            print(f"\n[WARN] Out of memory at resolution {map_resolution}.")
            if map_resolution > resolutions_to_try[-1]:
                print(f"  Trying lower resolution ({resolutions_to_try[resolutions_to_try.index(map_resolution) + 1]}) ...")
            else:
                print("  All resolutions exhausted.")
        except Exception as e:
            print(f"\n[ERROR] Failed to generate top-down map: {e}")
            break

    if colorized_map is None:
        print("\n[ERROR] Could not generate top-down map at any resolution.")
        print("  This is likely due to WSL memory limits with large HM3D scenes.")
        print("  The video and snapshots were saved successfully!")
        viz.close()
        print(f"\nPartial success! Outputs saved (except trajectory map):")
        print(f"  - Output folder:  {OUTPUT_DIR}")
        print(f"  - Video:          {OUTPUT_VIDEO}")
        print(f"  - Snapshots:      {OUTPUT_DIR}/snapshot_step_*.png")
        return

    # --- Step 5: Render paper-grade figure ---
    rendered_topdown = None
    try:
        rendered_topdown = viz.capture_rendered_topdown_view()
    except Exception as e:
        print(f"[WARN] Rendered top-down capture failed: {e}")
        print("  Falling back to trajectory-map-only output.")
    viz.render_paper_grade_figure(colorized_map, rendered_topdown)

    # --- Step 6: Clean up ---
    viz.close()

    print("\nAll outputs generated successfully!")
    print(f"  - Output folder:  {OUTPUT_DIR}")
    print(f"  - Video:          {OUTPUT_VIDEO}")
    print(f"  - Trajectory map: {OUTPUT_TRAJECTORY}")
    print(f"  - Snapshots:      {OUTPUT_DIR}/snapshot_step_*.png")


if __name__ == "__main__":
    main()
