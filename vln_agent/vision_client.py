"""
视觉 API 客户端模块
封装图片分析 API 调用，提供统一的视觉分析接口
"""

import logging
from typing import Optional, Dict, Any

import numpy as np

from vln_agent.config import VLNConfig
from api.openai_client import OpenAIClient
from utils.prompt_templates import (
    SYSTEM_PROMPT_NAVIGATION,
    SYSTEM_PROMPT_PURE_VISION,
    SYSTEM_PROMPT_CONFIRM_ARRIVAL,
    USER_PROMPT_NAVIGATION,
    USER_PROMPT_WITH_HISTORY,
    USER_PROMPT_CONFIRM,
    format_navigation_history,
)

logger = logging.getLogger(__name__)


class VisionClient:
    """
    视觉 API 客户端
    统一封装图片分析 API 调用，支持 GPT-4o 和 Claude
    """

    def __init__(self, config: VLNConfig):
        """
        初始化视觉客户端

        Args:
            config: VLN 系统配置
        """
        self.config = config
        self.openai_client: Optional[OpenAIClient] = None

        # 初始化 OpenAI 客户端
        if config.is_openai_configured():
            self.openai_client = OpenAIClient(config)
            logger.info("OpenAI 客户端初始化成功")
        else:
            logger.warning("未配置 OpenAI API Key，视觉分析功能不可用")

    def analyze_navigation_step(
        self,
        image: np.ndarray,
        user_goal: str,
        step: int,
        distance: float,
        history: Optional[list] = None,
        use_fallback: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        分析导航步骤的图片，获取 AI 决策

        Args:
            image: 当前第一视角 RGB 图片
            user_goal: 用户目标描述
            step: 当前步数
            distance: 已走距离
            history: 导航历史记录列表
            use_fallback: 是否使用降级模型

        Returns:
            AI 响应字典，包含 action, reasoning, scene_description
        """
        if not self.openai_client:
            logger.error("OpenAI 客户端未初始化")
            return None

        # 构建系统提示词
        system_prompt = SYSTEM_PROMPT_NAVIGATION.format(user_goal=user_goal)

        # 构建用户提示词
        if history:
            history_summary = format_navigation_history(history)
            user_prompt = USER_PROMPT_WITH_HISTORY.format(
                history_summary=history_summary,
                step=step,
                distance=distance,
                user_goal=user_goal,
            )
        else:
            user_prompt = USER_PROMPT_NAVIGATION.format(
                step=step,
                distance=distance,
                user_goal=user_goal,
            )

        # 调用 API
        return self.openai_client.analyze_image(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_fallback=use_fallback,
        )

    def analyze_pure_vision(
        self,
        images: list,
        user_goal: str,
        use_fallback: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        纯视觉模式：分析多张环视图片，判断目标方向

        Args:
            images: 多张不同角度的 RGB 图片列表
            user_goal: 用户目标描述
            use_fallback: 是否使用降级模型

        Returns:
            AI 响应字典
        """
        if not self.openai_client:
            logger.error("OpenAI 客户端未初始化")
            return None

        system_prompt = SYSTEM_PROMPT_PURE_VISION.format(user_goal=user_goal)
        user_prompt = f"这是当前场景 {len(images)} 个不同角度的图片。请分析目标 '{user_goal}' 可能在哪个方向。"

        return self.openai_client.analyze_multiple_images(
            images=images,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_fallback=use_fallback,
        )

    def confirm_arrival(
        self,
        image: np.ndarray,
        user_goal: str,
        position: tuple,
        use_fallback: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        确认是否已到达目标位置

        Args:
            image: 当前第一视角 RGB 图片
            user_goal: 用户目标描述
            position: 当前位置 (x, y, z)
            use_fallback: 是否使用降级模型

        Returns:
            AI 响应字典
        """
        if not self.openai_client:
            logger.error("OpenAI 客户端未初始化")
            return None

        system_prompt = SYSTEM_PROMPT_CONFIRM_ARRIVAL.format(user_goal=user_goal)
        user_prompt = USER_PROMPT_CONFIRM.format(
            user_goal=user_goal,
            x=position[0],
            y=position[1],
            z=position[2],
        )

        return self.openai_client.analyze_image(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_fallback=use_fallback,
        )

    def get_api_stats(self) -> Dict[str, Any]:
        """获取 API 调用统计"""
        if self.openai_client:
            return self.openai_client.get_stats()
        return {"total_calls": 0, "total_tokens": 0, "total_cost": 0.0}

    def is_available(self) -> bool:
        """检查视觉客户端是否可用"""
        return self.openai_client is not None
