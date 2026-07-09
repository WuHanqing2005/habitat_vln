"""
导航日志模块
将导航过程中的所有信息同步写入日志文件
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, TextIO


class NavigationLogger:
    """
    导航日志记录器
    将终端输出的所有信息同步写入日志文件
    """

    def __init__(self, log_dir: str, scene_name: str = "", goal: str = ""):
        """
        初始化导航日志记录器

        Args:
            log_dir: 日志输出目录
            scene_name: 场景名称
            goal: 导航目标
        """
        self.log_dir = log_dir
        self.scene_name = scene_name
        self.goal = goal
        self.log_file: Optional[TextIO] = None
        self.log_path: Optional[str] = None

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"navigation_log_{timestamp}.txt")
        self.log_file = open(self.log_path, "w", encoding="utf-8")

        # 写入文件头
        self._write_header()

    def _write_header(self):
        """写入日志文件头"""
        if not self.log_file:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"{'=' * 70}\n")
        self.log_file.write(f"  Habitat-VLN AI Agent 导航日志\n")
        self.log_file.write(f"{'=' * 70}\n")
        self.log_file.write(f"\n")
        self.log_file.write(f"  开始时间: {timestamp}\n")
        self.log_file.write(f"  场景: {self.scene_name}\n")
        self.log_file.write(f"  目标: {self.goal}\n")
        self.log_file.write(f"\n")
        self.log_file.write(f"{'─' * 70}\n")
        self.log_file.write(f"\n")
        self.log_file.flush()

    def log(self, message: str):
        """
        写入日志

        Args:
            message: 日志消息
        """
        if not self.log_file:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"[{timestamp}] {message}\n")
        self.log_file.flush()

    def log_step(
        self,
        step: int,
        action: str,
        position: tuple,
        heading: Optional[float] = None,
        scene_description: str = "",
        reasoning: str = "",
        distance: float = 0.0,
        remaining_distance: float = 0.0,
        image_path: str = "",
    ):
        """
        记录导航步骤

        Args:
            step: 步数
            action: 执行的动作
            position: 位置坐标 (x, y, z)
            heading: 朝向角度
            scene_description: AI 场景描述
            reasoning: AI 推理过程
            distance: 已走距离
            remaining_distance: 剩余距离
            image_path: 保存的图片路径
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pos_str = f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"

        self.log_file.write(f"[{timestamp}] {'=' * 70}\n")
        self.log_file.write(f"[{timestamp}]  步骤 {step} | 动作: {action} | 位置: {pos_str}\n")

        if heading is not None:
            self.log_file.write(f"[{timestamp}]  朝向: {heading:.1f}°\n")

        self.log_file.write(f"[{timestamp}]  已走距离: {distance:.2f}m | 剩余距离: {remaining_distance:.2f}m\n")

        if image_path:
            self.log_file.write(f"[{timestamp}]  图片: {image_path}\n")

        if scene_description:
            self.log_file.write(f"[{timestamp}]  场景: \"{scene_description}\"\n")
        if reasoning:
            self.log_file.write(f"[{timestamp}]  推理: \"{reasoning}\"\n")

        self.log_file.write(f"[{timestamp}] {'─' * 70}\n")
        self.log_file.flush()

    def log_semantic_phase(self, info: Dict[str, Any]):
        """
        记录语义地图定位阶段信息

        Args:
            info: 语义定位信息字典
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"[{timestamp}] ┌─ 语义地图定位阶段\n")

        if info.get("semantic_available", False):
            self.log_file.write(f"[{timestamp}] 语义解析: 找到 {info.get('num_categories', 0)} 个语义类别\n")
            for match in info.get("matches", []):
                status = "✓" if match.get("found") else " "
                self.log_file.write(f"[{timestamp}]   {status} {match['name']}\n")

            self.log_file.write(
                f"[{timestamp}] 语义定位: 中心=({info['center_x']:.2f}, {info['center_z']:.2f}), "
                f"范围={info['width']:.1f}m×{info['depth']:.1f}m\n"
            )
            self.log_file.write(
                f"[{timestamp}] 路径规划: 距离={info['distance']:.1f}m, "
                f"路点={info['num_waypoints']}, 预计步数={info['estimated_steps']}\n"
            )
        else:
            self.log_file.write(f"[{timestamp}] 语义标签不可用，降级为纯视觉模式\n")

        self.log_file.write(f"[{timestamp}] └─ 语义地图定位阶段\n\n")
        self.log_file.flush()

    def log_summary(self, stats: Dict[str, Any]):
        """
        记录导航总结

        Args:
            stats: 导航统计信息
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"[{timestamp}] {'=' * 70}\n")
        self.log_file.write(f"[{timestamp}]  🎯 导航完成！\n")
        self.log_file.write(f"[{timestamp}] {'=' * 70}\n")
        self.log_file.write(f"[{timestamp}]  目标: {stats.get('goal', '')}\n")
        self.log_file.write(f"[{timestamp}]  结果: {'成功' if stats.get('success', False) else '失败'}\n")
        self.log_file.write(f"[{timestamp}]  总步数: {stats.get('step', 0)}\n")
        self.log_file.write(f"[{timestamp}]  总距离: {stats.get('total_distance', 0):.2f}m\n")
        self.log_file.write(f"[{timestamp}]  总用时: {stats.get('elapsed_seconds', 0):.1f}s\n")
        self.log_file.write(f"[{timestamp}]  API调用: {stats.get('api_calls', 0)} 次\n")
        self.log_file.write(f"[{timestamp}]  Token消耗: {stats.get('total_tokens', 0)}\n")
        self.log_file.write(f"[{timestamp}]  API成本: ${stats.get('total_cost', 0):.4f}\n")
        self.log_file.write(f"[{timestamp}] {'─' * 70}\n")
        self.log_file.flush()

    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.log_file.write(f"\n{'=' * 70}\n")
            self.log_file.write(f"  日志结束\n")
            self.log_file.write(f"{'=' * 70}\n")
            self.log_file.close()
            self.log_file = None

    def __del__(self):
        """析构时自动关闭"""
        self.close()
