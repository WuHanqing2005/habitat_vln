"""
map_viz.py — Semantic-Enhanced Top-Down Map Visualization for Habitat-VLN
==========================================================================

Provides utilities for:
  1. Generating top-down occupancy maps from a habitat-sim pathfinder.
  2. Overlaying semantic class information from HM3D .semantic.txt files.
  3. Color-coding semantic regions on the map.
  4. Drawing agent trajectories.

Can be used standalone (with a SceneConfig) or imported by run_demo.py.

Usage:
    python map_viz.py                          # Interactive: select dataset/scene, show map
    python map_viz.py --scene <path_to.glb>    # Direct scene path
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors


# ============================================================================
# Imports from our project
# ============================================================================
import setup_path
import habitat_sim
from habitat.utils.visualizations import maps

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
# Constants
# ============================================================================
MAP_RESOLUTION = 1024


def bounded_meters_per_pixel(
    pathfinder: habitat_sim.PathFinder, longest_side_pixels: int
) -> float:
    """Bound top-down map size by its longest side to avoid HM3D OOM errors."""
    lower_bound, upper_bound = pathfinder.get_bounds()
    span_x = abs(float(upper_bound[0]) - float(lower_bound[0]))
    span_z = abs(float(upper_bound[2]) - float(lower_bound[2]))
    longest_span = max(span_x, span_z)
    if longest_span <= 0:
        raise RuntimeError("Pathfinder returned invalid bounds for top-down map generation.")
    return longest_span / float(longest_side_pixels)


# ============================================================================
# Semantic Map Overlay
# ============================================================================

def create_semantic_color_map(
    classes: List[SemanticClass],
) -> Tuple[dict, dict]:
    """
    Build lookup dictionaries from semantic class data.

    Returns:
        color_map:  {class_name_lower: (R, G, B) tuple}
        floor_map:  {class_name_lower: floor_number}
    """
    color_map: dict = {}
    floor_map: dict = {}

    for sc in classes:
        name_lower = sc.name.lower()
        if name_lower not in color_map:
            color_map[name_lower] = sc.color_rgb
            floor_map[name_lower] = sc.floor

    return color_map, floor_map


def render_semantic_legend(
    classes: List[SemanticClass],
    ax: plt.Axes,
    max_items: int = 15,
    loc: str = "lower left",
) -> None:
    """
    Render a color-coded legend for semantic classes on a matplotlib axis.

    Shows structural classes (wall, floor, ceiling, door, window, stairs)
    prioritized, with a fallback to showing the first N unique classes.

    Args:
        classes: List of SemanticClass objects from parse_semantic_txt()
        ax: Matplotlib axis to draw on
        max_items: Maximum number of legend entries
        loc: Legend location string
    """
    if not classes:
        return

    # Priority: structural classes first
    structural_keywords = [
        "wall", "floor", "ceiling", "door", "window",
        "stairs", "balustrade", "pillar", "column",
        "railing", "step",
    ]

    # Separate structural vs. other classes
    structural: List[SemanticClass] = []
    other: List[SemanticClass] = []
    seen_names = set()

    for c in classes:
        name_lower = c.name.lower()
        if name_lower in seen_names:
            continue
        seen_names.add(name_lower)

        if any(kw in name_lower for kw in structural_keywords):
            structural.append(c)
        else:
            other.append(c)

    # Combine: structural first, then others up to max_items
    display_classes = structural + other
    display_classes = display_classes[:max_items]

    # Create legend patches
    legend_handles = []
    for sc in display_classes:
        r, g, b = sc.color_rgb
        patch = Patch(
            color=(r / 255, g / 255, b / 255),
            label=f"{sc.name} (F{sc.floor})",
        )
        legend_handles.append(patch)

    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            loc=loc,
            fontsize=7,
            framealpha=0.85,
            title="Semantic Classes",
            title_fontsize=8,
        )
        ax.add_artist(legend)


def generate_semantic_topdown_map(
    sim: habitat_sim.Simulator,
    scene_config: SceneConfig,
    map_resolution: int = MAP_RESOLUTION,
    height: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a top-down map with semantic overlay.

    If semantic data is available, the map will include color-coded
    semantic regions. Otherwise, it falls back to the standard
    occupancy grid map.

    Args:
        sim: Initialized habitat-sim Simulator
        scene_config: SceneConfig with semantic paths
        map_resolution: Resolution of the output map
        height: Height slice for the map (uses pathfinder if None)

    Returns:
        RGB numpy array (H x W x 3) of the colorized map
    """
    # Determine height
    if height is None:
        # Use the pathfinder's default navigable height
        height = sim.pathfinder.get_bounds()[1][1] / 2

    if not sim.pathfinder.is_loaded:
        raise RuntimeError("Cannot render top-down map because the navmesh is not loaded.")

    # Get raw occupancy map
    raw_map = maps.get_topdown_map(
        pathfinder=sim.pathfinder,
        height=height,
        map_resolution=map_resolution,
        draw_border=True,
        meters_per_pixel=bounded_meters_per_pixel(sim.pathfinder, map_resolution),
    )

    # Colorize
    colorized = maps.colorize_topdown_map(raw_map)

    # If semantic data exists, we could overlay it here
    # (Currently, the semantic overlay is done via the legend in matplotlib;
    #  actual pixel-level semantic overlay requires rendering the semantic mesh,
    #  which is a more advanced feature beyond the current scope.)

    return colorized


# ============================================================================
# Standalone Map Visualization
# ============================================================================

def create_simulator_for_map(scene_config: SceneConfig) -> habitat_sim.Simulator:
    """
    Create a minimal habitat-sim Simulator for map generation only.
    No camera sensors needed — just the pathfinder.
    """
    scene_path_str = str(scene_config.render_asset_path.resolve())

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path_str
    sim_cfg.gpu_device_id = -1
    if scene_config.dataset_config_path:
        sim_cfg.scene_dataset_config_file = str(
            scene_config.dataset_config_path.resolve()
        )

    # No sensors needed for map-only mode
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = []

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    try:
        sim = habitat_sim.Simulator(cfg)
    except Exception as e:
        print(f"[ERROR] Failed to create simulator: {e}")
        sys.exit(1)

    if scene_config.navmesh_path:
        try:
            sim.pathfinder.load_nav_mesh(str(scene_config.navmesh_path.resolve()))
        except Exception as e:
            print(f"[WARNING] Could not load navmesh {scene_config.navmesh_path}: {e}")

    return sim


def visualize_map(scene_config: SceneConfig) -> None:
    """
    Generate and save a standalone top-down map with semantic overlay.
    """
    print("\nGenerating standalone top-down map ...")

    # Create simulator
    sim = create_simulator_for_map(scene_config)

    # Get map
    colorized_map = generate_semantic_topdown_map(sim, scene_config)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(colorized_map, interpolation="nearest")

    # Add semantic legend if available
    if scene_config.semantic_txt_path:
        classes = parse_semantic_txt(scene_config.semantic_txt_path)
        if classes:
            print(f"  Loaded {len(classes)} semantic classes")
            render_semantic_legend(classes, ax)

    # Title
    ax.set_title(
        f"Top-Down Map — {scene_config.dataset_name}/{scene_config.scene_name}",
        fontsize=14, fontweight="bold",
    )
    ax.tick_params(labelbottom=False, labelleft=False)

    # Save
    output_path = f"Output/map_{scene_config.dataset_name}_{scene_config.scene_name}.png"
    os.makedirs("Output", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Map saved to '{output_path}'")

    # Clean up
    sim.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Interactive map visualization with semantic overlay."""
    print("=" * 60)
    print("  Habitat-VLN Semantic Map Visualizer")
    print("=" * 60)

    # Check for direct scene argument
    if len(sys.argv) > 1 and sys.argv[1] == "--scene" and len(sys.argv) > 2:
        # Direct scene path provided
        scene_path = Path(sys.argv[2])
        if not scene_path.exists():
            print(f"[ERROR] Scene file not found: {scene_path}")
            sys.exit(1)
        config = resolve_assets(scene_path)
    else:
        # Interactive selection
        datasets = discover_datasets()
        if not datasets:
            print("[ERROR] No datasets found.")
            sys.exit(1)

        ds_path = select_dataset(datasets)
        if ds_path is None:
            sys.exit(0)

        scenes = discover_scenes(ds_path)
        if not scenes:
            print(f"[ERROR] No scenes in {ds_path}")
            sys.exit(1)

        scene_path = select_scene(scenes)
        if scene_path is None:
            sys.exit(0)

        config = resolve_assets(scene_path)

    print_scene_config(config)
    visualize_map(config)


if __name__ == "__main__":
    main()
