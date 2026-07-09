"""
AI Agent 主控制器
管理导航状态、上下文维护、决策协调
"""

import logging
import time
import math
import os
import cv2
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from vln_agent.config import VLNConfig, default_config
from vln_agent.vision_client import VisionClient
from vln_agent.semantic_locator import SemanticLocator
from vln_agent.instruction_parser import parse_instruction, get_room_features
from vln_agent.terminal_printer import TerminalPrinter
from vln_agent.navigation_logger import NavigationLogger
from utils.image_utils import image_to_base64

logger = logging.getLogger(__name__)


class VLNAgent:
    """
    VLN AI Agent 主控制器
    管理整个导航过程的状态、决策和输出
    """

    def __init__(self, config: Optional[VLNConfig] = None):
        """
        初始化 VLN Agent

        Args:
            config: VLN 系统配置，如果为 None 则使用默认配置
        """
        self.config = config or default_config

        # 核心模块
        self.vision_client: Optional[VisionClient] = None
        self.semantic_locator: Optional[SemanticLocator] = None
        self.printer = TerminalPrinter()
        self.logger: Optional[NavigationLogger] = None

        # 导航状态
        self.state = {
            "goal": "",
            "goal_room": "",
            "step": 0,
            "total_steps": 0,
            "total_distance": 0.0,
            "start_time": None,
            "position": (0.0, 0.0, 0.0),
            "heading": 0.0,
            "prev_position": (0.0, 0.0, 0.0),
            "stuck_count": 0,
            "stuck_recovery_count": 0,
            "turn_count": 0,
            "action_counts": {"move_forward": 0, "turn_left": 0, "turn_right": 0, "arrived": 0},
            "history": [],
            "is_pure_vision": False,
            "arrived": False,
            "success": False,
        }

        # Habitat 仿真器引用（由外部设置）
        self.sim = None
        self.pathfinder = None

        # 输出目录
        self.output_dir = ""
        self.pictures_dir = ""
        self.video_writer = None
        self.video_path = ""
        self.trajectory_path = ""
        self.saved_images = []  # 记录已保存的图片路径
        self.trajectory_data = []  # 记录轨迹点 (x, z) 用于绘制轨迹图

    @staticmethod
    def _quat_to_heading(rotation) -> float:
        """
        将四元数转换为朝向角度（度）
        兼容 habitat-sim 0.3.3（无 quat_to_rotation_vector）
        
        Args:
            rotation: 四元数 [x, y, z, w] 或 habitat_sim 的 quaternion 对象
            
        Returns:
            朝向角度（度）
        """
        try:
            # 尝试使用 habitat_sim 的 API
            import habitat_sim.utils.common as utils
            try:
                forward = utils.quat_to_rotation_vector(rotation)
                return math.degrees(math.atan2(forward[0], forward[2]))
            except AttributeError:
                pass
            
            # 手动计算：将四元数转换为前向向量
            # 四元数格式: [x, y, z, w]
            qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
            
            # 前向向量 (0, 0, -1) 经过四元数旋转
            # 使用标准公式: v' = q * v * q_conj
            fx = 2.0 * (qx * qz + qw * qy)
            fz = 2.0 * (qy * qz - qw * qx)
            
            return math.degrees(math.atan2(fx, fz))
        except Exception:
            return 0.0

    def initialize(
        self,
        sim,
        scene_name: str,
        goal: str,
        semantic_txt_path: Optional[str] = None,
    ):
        """
        初始化 Agent 并准备导航

        Args:
            sim: Habitat 仿真器实例
            scene_name: 场景名称
            goal: 用户导航目标
            semantic_txt_path: 语义标签文件路径
        """
        self.sim = sim
        self.state["goal"] = goal

        # 解析指令
        parsed = parse_instruction(goal)
        if parsed:
            self.state["goal_room"] = parsed["goal_room"]
            logger.info(f"解析指令: 目标={parsed['goal_room']}, 置信度={parsed['confidence']:.2f}")
        else:
            self.state["goal_room"] = goal
            logger.warning(f"无法解析指令，使用原始文本作为目标: {goal}")

        # 初始化语义定位器
        self.semantic_locator = SemanticLocator(semantic_txt_path)

        # 初始化视觉客户端
        self.vision_client = VisionClient(self.config)

        # 获取 pathfinder
        try:
            self.pathfinder = sim.pathfinder
        except Exception:
            self.pathfinder = None

        # 创建输出目录（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"Output/{timestamp}"
        self.pictures_dir = os.path.join(self.output_dir, "Pictures")
        os.makedirs(self.pictures_dir, exist_ok=True)

        # 获取初始位置
        try:
            agent_state = sim.get_agent(0).get_state()
            self.state["position"] = (
                float(agent_state.position[0]),
                float(agent_state.position[1]),
                float(agent_state.position[2]),
            )
            self.state["prev_position"] = self.state["position"]

            # 计算朝向（使用兼容方法）
            self.state["heading"] = self._quat_to_heading(agent_state.rotation)
        except Exception as e:
            logger.warning(f"获取初始位置失败: {e}")

        # 初始化日志记录器
        self.logger = NavigationLogger(
            log_dir=self.output_dir,
            scene_name=scene_name,
            goal=goal,
        )

        # 确定导航模式
        is_pure_vision = not (self.semantic_locator and self.semantic_locator.is_available)
        self.state["is_pure_vision"] = is_pure_vision

        mode = "纯 AI 视觉导航" if is_pure_vision else "混合策略（语义定位 + AI 视觉确认）"

        # 打印标题头
        self.printer.print_header(scene_name, goal, mode)

        # 初始化视频录制
        self._init_video_recording()

        # 记录初始轨迹点
        self.trajectory_data.append({
            "x": self.state["position"][0],
            "z": self.state["position"][2],
            "step": 0,
            "action": "start",
        })

        return mode

    def run_semantic_phase(self) -> Optional[Dict[str, Any]]:
        """
        执行语义地图定位阶段

        Returns:
            语义定位结果字典，包含目标位置信息
            如果语义标签不可用，返回 None
        """
        if not self.semantic_locator or not self.semantic_locator.is_available:
            info = {
                "semantic_available": False,
                "goal_room": self.state["goal_room"],
                "matches": [],
            }
            self.printer.print_semantic_phase(info)
            if self.logger:
                self.logger.log_semantic_phase(info)
            return None

        goal_room = self.state["goal_room"]

        # 获取语义定位信息
        semantic_info = self.semantic_locator.get_semantic_info(goal_room)

        # 尝试估计房间位置
        position_result = self.semantic_locator.estimate_room_position(
            goal_room=goal_room,
            pathfinder=self.pathfinder,
            agent_position=self.state["position"],
        )

        if position_result:
            semantic_info.update(position_result)

        # 打印语义定位信息
        self.printer.print_semantic_phase(semantic_info)

        # 记录日志
        if self.logger:
            self.logger.log_semantic_phase(semantic_info)

        return position_result

    def execute_action(self, action: str) -> Dict[str, Any]:
        """
        在 Habitat 仿真器中执行动作

        Args:
            action: 动作名称 (move_forward / turn_left / turn_right)

        Returns:
            执行结果字典
        """
        result = {
            "action": action,
            "displacement": 0.0,
            "new_position": self.state["position"],
            "new_heading": self.state["heading"],
            "position_unchanged": True,
        }

        if not self.sim:
            return result

        try:
            agent = self.sim.get_agent(0)

            if action == "move_forward":
                # 记录移动前的位置
                prev_pos = self.state["position"]

                # 执行前进动作
                self.sim.step("move_forward")

                # 获取新位置
                new_state = agent.get_state()
                new_pos = (
                    float(new_state.position[0]),
                    float(new_state.position[1]),
                    float(new_state.position[2]),
                )

                # 计算位移
                dx = new_pos[0] - prev_pos[0]
                dz = new_pos[2] - prev_pos[2]
                displacement = math.sqrt(dx * dx + dz * dz)

                result["displacement"] = displacement
                result["new_position"] = new_pos
                result["position_unchanged"] = displacement < self.config.navigation.position_change_threshold

                # 更新状态
                self.state["position"] = new_pos
                self.state["total_distance"] += displacement

                # 卡住检测
                if result["position_unchanged"]:
                    self.state["stuck_count"] += 1
                else:
                    self.state["stuck_count"] = 0

            elif action == "turn_left":
                self.sim.step("turn_left")
                self.state["turn_count"] += 1

                # 更新朝向（使用兼容方法）
                new_state = agent.get_state()
                self.state["heading"] = self._quat_to_heading(new_state.rotation)
                result["new_heading"] = self.state["heading"]

            elif action == "turn_right":
                self.sim.step("turn_right")
                self.state["turn_count"] += 1

                # 更新朝向（使用兼容方法）
                new_state = agent.get_state()
                self.state["heading"] = self._quat_to_heading(new_state.rotation)
                result["new_heading"] = self.state["heading"]

            # 更新动作计数
            if action in self.state["action_counts"]:
                self.state["action_counts"][action] += 1

        except Exception as e:
            logger.error(f"执行动作 {action} 失败: {e}")
            result["error"] = str(e)

        return result

    def capture_image(self) -> Optional[np.ndarray]:
        """
        从仿真器捕获当前第一视角 RGB 图片

        Returns:
            RGB 3通道图片数组，失败时返回 None
        """
        if not self.sim:
            return None

        try:
            observations = self.sim.get_sensor_observations()
            rgb = observations.get("rgb")
            if rgb is None:
                return None
            
            # Habitat-Sim 返回 RGBA (4通道)，转换为 RGB (3通道)
            if rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            
            return rgb
        except Exception as e:
            logger.error(f"捕获图片失败: {e}")
            return None

    def check_stuck(self) -> bool:
        """
        检查是否卡住

        Returns:
            是否卡住
        """
        return self.state["stuck_count"] >= self.config.navigation.stuck_threshold

    def handle_stuck(self) -> str:
        """
        处理卡住情况：强制转向

        Returns:
            强制执行的转向动作
        """
        self.state["stuck_count"] = 0
        self.state["stuck_recovery_count"] += 1

        # 交替左右转，避免死循环
        if self.state["stuck_recovery_count"] % 2 == 0:
            return "turn_right"
        else:
            return "turn_left"

    def get_heading_description(self, heading_deg: float) -> str:
        """获取朝向的文字描述"""
        # 标准化到 0-360
        heading = heading_deg % 360
        if heading < 22.5 or heading >= 337.5:
            return "北"
        elif heading < 67.5:
            return "东北"
        elif heading < 112.5:
            return "东"
        elif heading < 157.5:
            return "东南"
        elif heading < 202.5:
            return "南"
        elif heading < 247.5:
            return "西南"
        elif heading < 292.5:
            return "西"
        else:
            return "西北"

    def _init_video_recording(self):
        """
        初始化 MP4 视频录制
        使用 OpenCV VideoWriter 实时录制导航过程
        """
        if not self.config.output.record_video:
            return

        try:
            # 先捕获一帧来确定视频尺寸
            sample_image = self.capture_image()
            if sample_image is None:
                logger.warning("无法获取样本图片，跳过视频录制初始化")
                return

            # 获取图片尺寸 (H, W, C)
            h, w = sample_image.shape[:2]
            
            # 视频文件路径
            self.video_path = os.path.join(self.output_dir, "navigation_video.mp4")
            
            # 使用 MP4V 编码器
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self.config.output.video_fps
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, fps, (w, h))
            
            if self.video_writer.isOpened():
                logger.info(f"视频录制已初始化: {self.video_path} (fps={fps}, size={w}x{h})")
            else:
                logger.warning("VideoWriter 打开失败，尝试其他编码器")
                # 尝试其他编码器
                self.video_writer = cv2.VideoWriter(self.video_path, fourcc, fps, (w, h))
        except Exception as e:
            logger.error(f"初始化视频录制失败: {e}")
            self.video_writer = None

    def save_step_image(self, image: np.ndarray, step: int, action: str = "") -> str:
        """
        保存当前步的图片到 Pictures/ 文件夹

        Args:
            image: RGB 图片数组
            step: 当前步数
            action: 当前执行的动作

        Returns:
            保存的图片文件路径，失败返回空字符串
        """
        if image is None:
            return ""

        try:
            # 生成文件名：step_{步数}_{动作}.jpg
            action_tag = action.replace(" ", "_") if action else "unknown"
            filename = f"step_{step:04d}_{action_tag}.jpg"
            filepath = os.path.join(self.pictures_dir, filename)

            # 保存图片（RGB -> BGR for OpenCV）
            if image.shape[2] == 3:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = image

            cv2.imwrite(filepath, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.config.image.jpeg_quality])

            # 记录已保存的图片
            self.saved_images.append(filepath)

            return filepath
        except Exception as e:
            logger.error(f"保存图片失败 (step {step}): {e}")
            return ""

    def write_video_frame(self, image: np.ndarray):
        """
        将一帧写入视频文件

        Args:
            image: RGB 图片数组
        """
        if self.video_writer is None or image is None:
            return

        try:
            # RGB -> BGR for OpenCV
            if image.shape[2] == 3:
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                frame = image

            self.video_writer.write(frame)
        except Exception as e:
            logger.error(f"写入视频帧失败: {e}")

    def generate_trajectory_map(self):
        """
        生成导航轨迹图
        使用 habitat-sim 的顶视图占用地图作为底图，
        叠加完整的运动轨迹、起点、终点、转向点等信息
        """
        if not self.trajectory_data:
            logger.warning("无轨迹数据，跳过轨迹图生成")
            return

        try:
            import matplotlib
            matplotlib.use("Agg")  # 非交互式后端
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
            from habitat.utils.visualizations import maps as habitat_maps

            # 提取轨迹点
            xs = [p["x"] for p in self.trajectory_data]
            zs = [p["z"] for p in self.trajectory_data]
            steps_list = [p["step"] for p in self.trajectory_data]
            actions_list = [p["action"] for p in self.trajectory_data]

            # 创建图形
            fig, ax = plt.subplots(1, 1, figsize=(14, 12))

            # ---- 生成顶视图占用地图作为底图 ----
            topdown_map = None
            map_resolution = 2048
            map_bounds = None  # (x_min, x_max, z_min, z_max)

            if self.pathfinder is not None and self.pathfinder.is_loaded:
                try:
                    # 获取地图边界
                    lower_bound, upper_bound = self.pathfinder.get_bounds()
                    x_min, x_max = float(lower_bound[0]), float(upper_bound[0])
                    z_min, z_max = float(lower_bound[2]), float(upper_bound[2])
                    map_bounds = (x_min, x_max, z_min, z_max)

                    # 计算合适的 height（使用 pathfinder 默认高度）
                    height = (lower_bound[1] + upper_bound[1]) / 2.0

                    # 计算 meters_per_pixel
                    span_x = abs(x_max - x_min)
                    span_z = abs(z_max - z_min)
                    longest_span = max(span_x, span_z)
                    if longest_span > 0:
                        meters_per_pixel = longest_span / float(map_resolution)
                    else:
                        meters_per_pixel = None

                    # 生成顶视图占用地图
                    raw_map = habitat_maps.get_topdown_map(
                        pathfinder=self.pathfinder,
                        height=height,
                        map_resolution=map_resolution,
                        draw_border=True,
                        meters_per_pixel=meters_per_pixel,
                    )

                    # 着色
                    topdown_map = habitat_maps.colorize_topdown_map(raw_map)

                    # 显示底图
                    ax.imshow(topdown_map, interpolation="nearest", zorder=1)
                except Exception as e:
                    logger.warning(f"生成顶视图地图失败，回退到纯轨迹图: {e}")
                    topdown_map = None

            if topdown_map is None:
                # 回退：纯轨迹图（无底图）
                ax.set_facecolor("#1a1a2e")
                ax.grid(True, alpha=0.3, color="#444444")

            # ---- 坐标转换函数 ----
            def world_to_map(x_world, z_world):
                """将世界坐标转换为地图像素坐标"""
                if map_bounds is None or topdown_map is None:
                    return x_world, z_world  # 回退到世界坐标
                x_min, x_max, z_min, z_max = map_bounds
                h, w = topdown_map.shape[:2]
                # 地图坐标系：左上角为原点，x向右，y向下
                # 世界坐标系：x向右，z向上（但地图中z向下）
                px = (x_world - x_min) / (x_max - x_min) * w
                py = (z_max - z_world) / (z_max - z_min) * h  # z轴翻转
                return px, py

            # 转换所有轨迹点到地图坐标
            map_xs = []
            map_zs = []
            for x_w, z_w in zip(xs, zs):
                px, py = world_to_map(x_w, z_w)
                map_xs.append(px)
                map_zs.append(py)

            # ---- 绘制轨迹线 ----
            ax.plot(map_xs, map_zs, "-", color="#00d4ff", linewidth=2.0,
                    alpha=0.9, zorder=4, label="运动轨迹")

            # ---- 标记起点 ----
            ax.scatter(map_xs[0], map_zs[0], c="#00ff88", s=250, marker="o",
                       edgecolors="white", linewidth=2.5, zorder=6,
                       label=f"起点 (Step 0)")

            # ---- 标记终点 ----
            end_color = "#00ff88" if self.state.get("success") else "#ff4444"
            end_label = f"终点 (Step {steps_list[-1]})"
            if self.state.get("success"):
                end_label += " ✓ 到达"
            else:
                end_label += " ✗ 未到达"
            ax.scatter(map_xs[-1], map_zs[-1], c=end_color, s=250, marker="s",
                       edgecolors="white", linewidth=2.5, zorder=6, label=end_label)

            # ---- 标记转向点 ----
            for i in range(len(map_xs)):
                action = actions_list[i]
                if action == "turn_left":
                    ax.scatter(map_xs[i], map_zs[i], c="#ff8800", s=60, marker="<",
                               alpha=0.7, zorder=3)
                elif action == "turn_right":
                    ax.scatter(map_xs[i], map_zs[i], c="#aa44ff", s=60, marker=">",
                               alpha=0.7, zorder=3)
                elif action == "arrived":
                    ax.scatter(map_xs[i], map_zs[i], c="#ffdd00", s=200, marker="*",
                               edgecolors="#ff8800", linewidth=2, zorder=7,
                               label="到达位置")

            # ---- 添加步数标注（每隔 N 步标注一个） ----
            label_interval = max(1, len(steps_list) // 25)
            for i in range(0, len(steps_list), label_interval):
                if i == 0 or i == len(steps_list) - 1:
                    continue  # 起点和终点已标注
                ax.annotate(
                    f"S{steps_list[i]}",
                    (map_xs[i], map_zs[i]),
                    fontsize=6,
                    alpha=0.8,
                    ha="center",
                    va="bottom",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#333333",
                              edgecolor="none", alpha=0.7),
                )

            # ---- 图形美化 ----
            if topdown_map is not None:
                ax.tick_params(labelbottom=False, labelleft=False)
                ax.set_xlabel("")
                ax.set_ylabel("")
            else:
                ax.set_xlabel("X 坐标 (m)", fontsize=12, color="#cccccc")
                ax.set_ylabel("Z 坐标 (m)", fontsize=12, color="#cccccc")
                ax.tick_params(colors="#cccccc")

            ax.set_title(
                f"导航轨迹图 — 顶视图地图 + 运动轨迹\n"
                f"目标: {self.state['goal']}",
                fontsize=14, fontweight="bold", color="white", pad=15,
            )
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#0d0d1a")

            # ---- 自定义图例 ----
            legend_elements = [
                Patch(facecolor="#00d4ff", edgecolor="none", label="运动轨迹"),
                plt.scatter([], [], c="#00ff88", s=100, marker="o", label="起点"),
                plt.scatter([], [], c=end_color, s=100, marker="s", label="终点"),
                plt.scatter([], [], c="#ff8800", s=60, marker="<", label="左转"),
                plt.scatter([], [], c="#aa44ff", s=60, marker=">", label="右转"),
            ]
            if any(a == "arrived" for a in actions_list):
                legend_elements.append(
                    plt.scatter([], [], c="#ffdd00", s=150, marker="*", label="到达位置")
                )

            legend = ax.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=9,
                framealpha=0.85,
                facecolor="#222244",
                edgecolor="#444466",
                labelcolor="white",
            )
            ax.add_artist(legend)

            # ---- 添加统计信息文本框 ----
            stats_text = (
                f"总步数: {self.state['step']}\n"
                f"总距离: {self.state['total_distance']:.2f}m\n"
                f"总用时: {self.state.get('elapsed_seconds', 0):.1f}s\n"
                f"结果: {'✓ 成功到达' if self.state.get('success') else '✗ 未到达'}"
            )
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                color="white",
                bbox=dict(boxstyle="round", facecolor="#222244",
                          edgecolor="#444466", alpha=0.85),
                zorder=10,
            )

            plt.tight_layout()

            # ---- 保存轨迹图 ----
            self.trajectory_path = os.path.join(self.output_dir, "trajectory_map.png")
            plt.savefig(self.trajectory_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)

            logger.info(f"轨迹图已保存: {self.trajectory_path}")

        except ImportError as e:
            logger.warning(f"缺少依赖库，跳过轨迹图生成: {e}")
        except Exception as e:
            logger.error(f"生成轨迹图失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def get_stats(self) -> Dict[str, Any]:
        """获取导航统计信息"""
        elapsed = 0.0
        if self.state["start_time"]:
            elapsed = time.time() - self.state["start_time"]

        api_stats = {}
        if self.vision_client:
            api_stats = self.vision_client.get_api_stats()

        output_files = {
            "导航日志": self.logger.log_path if self.logger and self.logger.log_path else "",
        }

        if self.saved_images:
            output_files["保存图片数"] = str(len(self.saved_images))
            output_files["图片目录"] = self.pictures_dir

        if self.video_path and os.path.exists(self.video_path):
            output_files["视频录制"] = self.video_path

        if self.trajectory_path and os.path.exists(self.trajectory_path):
            output_files["轨迹图"] = self.trajectory_path

        return {
            "goal": self.state["goal"],
            "goal_room": self.state["goal_room"],
            "step": self.state["step"],
            "total_steps": self.state["total_steps"],
            "total_distance": self.state["total_distance"],
            "elapsed_seconds": elapsed,
            "success": self.state["success"],
            "arrived": self.state["arrived"],
            "turn_count": self.state["turn_count"],
            "stuck_recovery_count": self.state["stuck_recovery_count"],
            "action_counts": self.state["action_counts"],
            "api_calls": api_stats.get("total_calls", 0),
            "total_tokens": api_stats.get("total_tokens", 0),
            "total_cost": api_stats.get("total_cost", 0.0),
            "output_files": output_files,
        }

    def cleanup(self):
        """清理资源"""
        # 关闭日志记录器
        if self.logger:
            self.logger.close()

        # 释放视频写入器
        if self.video_writer is not None:
            try:
                self.video_writer.release()
                logger.info(f"视频录制已保存: {self.video_path}")
            except Exception as e:
                logger.error(f"释放视频写入器失败: {e}")
            self.video_writer = None

        # 生成轨迹图
        self.generate_trajectory_map()
