"""
导航循环模块
实现感知-决策-行动的核心导航循环
"""

import logging
import time
import math
from typing import Optional, Dict, Any

import numpy as np

from vln_agent.agent import VLNAgent
from vln_agent.config import VLNConfig

logger = logging.getLogger(__name__)


class NavigationLoop:
    """
    导航循环控制器
    管理感知-决策-行动的循环过程
    """

    def __init__(self, agent: VLNAgent, config: VLNConfig):
        """
        初始化导航循环

        Args:
            agent: VLN Agent 实例
            config: VLN 系统配置
        """
        self.agent = agent
        self.config = config

    def run(self) -> Dict[str, Any]:
        """
        执行完整的导航循环

        Returns:
            导航结果统计字典
        """
        agent = self.agent
        nav_config = self.config.navigation
        max_steps = nav_config.max_steps

        # 记录开始时间
        agent.state["start_time"] = time.time()

        # 第一步：语义地图定位
        semantic_result = agent.run_semantic_phase()

        # 第二步：进入感知-决策-行动循环
        for step in range(1, max_steps + 1):
            agent.state["step"] = step
            agent.state["total_steps"] = max_steps

            # 检查是否已到达
            if agent.state["arrived"]:
                break

            # ---- 感知阶段 ----
            image = agent.capture_image()
            if image is None:
                agent.printer.print_error("无法捕获第一视角图片")
                continue

            # 保存图片到 Pictures/ 文件夹
            saved_image_path = agent.save_step_image(image, step, action="capture")

            # 写入视频帧（实时录制）
            agent.write_video_frame(image)

            # 图片信息
            perception_info = {
                "original_size": (image.shape[1], image.shape[0]),
                "compressed_size": f"{self.config.image.max_width}×{self.config.image.max_height}",
                "file_size_kb": 0.0,
                "api_sent": True,
                "model_name": self.config.api.openai_model,
                "saved_image": saved_image_path,
            }

            # ---- 决策阶段 ----
            ai_result = None
            api_error = False
            api_error_reason = ""
            retry_count = 0

            # 检查是否卡住
            is_stuck = agent.check_stuck()
            if is_stuck:
                # 卡住时强制转向，不调用 API
                forced_action = agent.handle_stuck()
                ai_result = {
                    "action": forced_action,
                    "reasoning": f"检测到卡住（连续{nav_config.stuck_threshold}步未移动），强制{forced_action}",
                    "scene_description": "（卡住检测，未调用 AI）",
                }
            else:
                # 正常调用 AI 分析
                for attempt in range(self.config.api.api_max_retries):
                    ai_result = agent.vision_client.analyze_navigation_step(
                        image=image,
                        user_goal=agent.state["goal"],
                        step=step,
                        distance=agent.state["total_distance"],
                        history=agent.state["history"],
                        use_fallback=(attempt > 0),
                    )

                    if ai_result is not None:
                        break
                    else:
                        api_error = True
                        api_error_reason = f"API 调用失败 (attempt {attempt + 1}/{self.config.api.api_max_retries})"
                        retry_count = attempt + 1
                        if attempt < self.config.api.api_max_retries - 1:
                            time.sleep(self.config.api.api_retry_delay)

            # 如果 AI 分析失败，使用默认动作
            if ai_result is None:
                ai_result = {
                    "action": "turn_left",
                    "reasoning": "AI 分析失败，默认左转探索",
                    "scene_description": "（AI 分析失败）",
                }

            action = ai_result["action"]

            # ---- 执行阶段 ----
            if action == "arrived":
                # 到达目标
                agent.state["arrived"] = True
                agent.state["success"] = True

                # 打印到达信息
                elapsed = time.time() - agent.state["start_time"]
                arrival_stats = {
                    "step": step,
                    "total_steps": max_steps,
                    "remaining_distance": 0.0,
                    "elapsed_seconds": elapsed,
                    "goal": agent.state["goal"],
                    "total_distance": agent.state["total_distance"],
                    "api_calls": agent.vision_client.get_api_stats().get("total_calls", 0),
                    "scene_description": ai_result.get("scene_description", ""),
                    "reasoning": ai_result.get("reasoning", ""),
                }
                agent.printer.print_arrival(arrival_stats)

                # 记录日志
                if agent.logger:
                    agent.logger.log_step(
                        step=step,
                        action="arrived",
                        position=agent.state["position"],
                        heading=agent.state["heading"],
                        scene_description=ai_result.get("scene_description", ""),
                        reasoning=ai_result.get("reasoning", ""),
                        distance=agent.state["total_distance"],
                        remaining_distance=0.0,
                        image_path=saved_image_path,
                    )
                break

            # 执行动作
            execution_result = agent.execute_action(action)

            # 计算剩余距离和进度
            remaining_distance = max(0.0, 10.0 - agent.state["total_distance"])  # 估计值
            progress = (step / max_steps) * 100
            remaining_steps = max_steps - step

            # 构建状态信息
            status_info = {
                "progress": progress,
                "remaining_steps": remaining_steps,
                "is_stuck": is_stuck,
                "stuck_count": agent.state["stuck_count"],
                "api_error": api_error,
                "api_error_reason": api_error_reason,
                "retry_count": retry_count,
                "max_retries": self.config.api.api_max_retries,
                "retry_success": api_error and ai_result is not None,
            }

            # 构建执行信息
            exec_info = {
                "action": action,
                "displacement": execution_result.get("displacement", 0.0),
                "new_position": execution_result.get("new_position", agent.state["position"]),
                "new_heading": agent.get_heading_description(agent.state["heading"]),
                "position_unchanged": execution_result.get("position_unchanged", True),
            }

            # 打印步骤信息
            elapsed = time.time() - agent.state["start_time"]
            agent.printer.print_navigation_step(
                step=step,
                total_steps=max_steps,
                remaining_distance=remaining_distance,
                elapsed_seconds=elapsed,
                perception_info=perception_info,
                ai_info=ai_result,
                execution_info=exec_info,
                status_info=status_info,
            )

            # 记录日志
            if agent.logger:
                agent.logger.log_step(
                    step=step,
                    action=action,
                    position=agent.state["position"],
                    heading=agent.state["heading"],
                    scene_description=ai_result.get("scene_description", ""),
                    reasoning=ai_result.get("reasoning", ""),
                    distance=agent.state["total_distance"],
                    remaining_distance=remaining_distance,
                    image_path=saved_image_path,
                )

            # 更新历史记录
            agent.state["history"].append({
                "step": step,
                "action": action,
                "x": agent.state["position"][0],
                "y": agent.state["position"][1],
                "z": agent.state["position"][2],
                "scene_description": ai_result.get("scene_description", ""),
                "reasoning": ai_result.get("reasoning", ""),
            })

            # 记录轨迹点（用于轨迹图）
            agent.trajectory_data.append({
                "x": agent.state["position"][0],
                "z": agent.state["position"][2],
                "step": step,
                "action": action,
            })

            # 保持历史记录在合理大小
            max_history = self.config.navigation.context_history_size
            if len(agent.state["history"]) > max_history * 2:
                agent.state["history"] = agent.state["history"][-max_history:]

        # ---- 导航结束 ----
        # 如果达到最大步数仍未到达
        if not agent.state["arrived"]:
            agent.state["success"] = False
            agent.printer.print_warning(
                f"达到最大步数 ({max_steps})，导航未完成",
                {"建议": "可以尝试重新输入指令，或选择更近的目标"},
            )

        # 打印总结
        stats = agent.get_stats()
        agent.printer.print_summary(stats)

        # 记录日志总结
        if agent.logger:
            agent.logger.log_summary(stats)

        return stats
