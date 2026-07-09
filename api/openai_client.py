"""
OpenAI GPT-4o / GPT-4V API 客户端
封装图片分析和导航决策的 API 调用
"""

import json
import logging
import time
from typing import Optional, Dict, Any

import requests

from vln_agent.config import VLNConfig
from utils.image_utils import image_to_base64, create_image_data_url

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    OpenAI API 客户端
    支持 GPT-4o 和 GPT-4o-mini 模型
    """

    def __init__(self, config: VLNConfig):
        """
        初始化 OpenAI 客户端

        Args:
            config: VLN 系统配置
        """
        self.config = config
        self.api_key = config.api.openai_api_key
        self.base_url = config.api.openai_base_url
        self.model = config.api.openai_model
        self.fallback_model = config.api.openai_fallback_model
        self.timeout = config.api.api_timeout
        self.max_retries = config.api.api_max_retries
        self.retry_delay = config.api.api_retry_delay

        # API 调用统计
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        # 模型价格（每 1K tokens，美元）
        self.model_pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-vision-preview": {"input": 0.01, "output": 0.03},
        }

    def _get_headers(self) -> Dict[str, str]:
        """获取 API 请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        计算 API 调用成本

        Args:
            model: 模型名称
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数

        Returns:
            成本（美元）
        """
        pricing = self.model_pricing.get(model, {"input": 0.01, "output": 0.03})
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def analyze_image(
        self,
        image: "np.ndarray",
        system_prompt: str,
        user_prompt: str,
        use_fallback: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        分析图片并返回 AI 决策

        Args:
            image: RGB 图片数组
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            use_fallback: 是否使用降级模型（低成本）

        Returns:
            AI 响应字典，包含 action, reasoning, scene_description
            失败时返回 None
        """
        model = self.fallback_model if use_fallback else self.model

        # 编码图片
        base64_str, size_bytes, original_size = image_to_base64(
            image,
            max_width=self.config.image.max_width,
            max_height=self.config.image.max_height,
            quality=self.config.image.jpeg_quality,
            encode_format=self.config.image.encode_format,
        )

        image_url = create_image_data_url(
            base64_str, self.config.image.encode_format
        )

        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "auto",
                        },
                    },
                ],
            },
        ]

        # 调用 API（带重试）
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": self.config.api.max_tokens_per_response,
                        "temperature": 0.7,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                # 更新统计
                self.total_calls += 1
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                self.total_tokens += input_tokens + output_tokens
                self.total_cost += self._calculate_cost(model, input_tokens, output_tokens)

                # 解析响应
                content = result["choices"][0]["message"]["content"]
                return self._parse_response(content)

            except requests.exceptions.Timeout:
                logger.warning(
                    f"API 调用超时 (attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            except requests.exceptions.RequestException as e:
                logger.error(
                    f"API 调用失败 (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"API 响应解析失败: {e}")
                return None

        logger.error(f"API 调用在 {self.max_retries} 次重试后仍然失败")
        return None

    def analyze_multiple_images(
        self,
        images: list,
        system_prompt: str,
        user_prompt: str,
        use_fallback: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        分析多张图片（用于纯视觉模式的 360 度环视）

        Args:
            images: RGB 图片数组列表
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            use_fallback: 是否使用降级模型

        Returns:
            AI 响应字典
        """
        model = self.fallback_model if use_fallback else self.model

        # 构建多图片消息
        content_parts = [{"type": "text", "text": user_prompt}]

        for i, image in enumerate(images):
            base64_str, _, _ = image_to_base64(
                image,
                max_width=self.config.image.max_width,
                max_height=self.config.image.max_height,
                quality=self.config.image.jpeg_quality,
                encode_format=self.config.image.encode_format,
            )
            image_url = create_image_data_url(
                base64_str, self.config.image.encode_format
            )
            content_parts.append({
                "type": "text",
                "text": f"\n[图片 {i + 1}/{len(images)}]:"
            })
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": image_url, "detail": "low"},
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        # 调用 API（带重试）
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": self.config.api.max_tokens_per_response,
                        "temperature": 0.7,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                self.total_calls += 1
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                self.total_tokens += input_tokens + output_tokens
                self.total_cost += self._calculate_cost(model, input_tokens, output_tokens)

                content = result["choices"][0]["message"]["content"]
                return self._parse_response(content)

            except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError) as e:
                logger.warning(
                    f"多图片 API 调用失败 (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return None

    def _parse_response(self, content: str) -> Optional[Dict[str, Any]]:
        """
        解析 AI 响应中的 JSON 内容

        Args:
            content: API 返回的文本内容

        Returns:
            解析后的字典，包含 action, reasoning, scene_description
        """
        # 尝试提取 JSON 部分
        content = content.strip()

        # 查找 JSON 块（可能被 ```json ... ``` 包裹）
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            if json_end != -1:
                content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            if json_end != -1:
                content = content[json_start:json_end].strip()

        # 尝试解析 JSON
        try:
            result = json.loads(content)
            # 验证必要字段
            if "action" not in result:
                logger.warning(f"AI 响应缺少 action 字段: {content}")
                return None

            # 验证 action 值
            valid_actions = {"move_forward", "turn_left", "turn_right", "arrived"}
            if result["action"] not in valid_actions:
                logger.warning(f"无效的 action 值: {result['action']}")
                return None

            return {
                "action": result["action"],
                "reasoning": result.get("reasoning", ""),
                "scene_description": result.get("scene_description", ""),
            }

        except json.JSONDecodeError:
            logger.warning(f"无法解析 AI 响应为 JSON: {content[:200]}...")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """获取 API 调用统计信息"""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "model": self.model,
        }
