"""
VLN AI Agent 配置模块
管理 API 密钥、模型参数、导航参数等所有配置项
支持从 config.json 文件加载和保存配置
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# 默认配置文件路径
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")


@dataclass
class APIConfig:
    """API 相关配置"""
    # OpenAI 配置
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"  # 主模型
    openai_fallback_model: str = "gpt-4o-mini"  # 降级模型（低成本）

    # Anthropic Claude 配置（备选）
    claude_api_key: str = ""
    claude_model: str = "claude-3-sonnet-20240229"

    # API 调用参数
    api_timeout: int = 60  # API 超时时间（秒）
    api_max_retries: int = 3  # 最大重试次数
    api_retry_delay: float = 2.0  # 重试间隔（秒）

    # Token 限制
    max_tokens_per_response: int = 500
    max_context_tokens: int = 128000


@dataclass
class ImageConfig:
    """图片处理相关配置"""
    # 压缩参数
    max_width: int = 1024
    max_height: int = 768
    jpeg_quality: int = 85
    encode_format: str = "jpeg"  # jpeg / webp / png

    # 捕获参数
    rgb_sensor_name: str = "rgb"
    camera_height: float = 0.88  # 相机高度（米）


@dataclass
class NavigationConfig:
    """导航相关配置"""
    # 动作参数
    forward_step_size: float = 0.08  # 前进步长（米）
    turn_angle: float = 8.0  # 转向角度（度）

    # 循环控制
    max_steps: int = 300  # 最大步数
    max_api_calls_per_step: int = 1  # 每步最大 API 调用次数

    # 防循环机制
    stuck_threshold: int = 3  # 连续多少步位置未变化视为卡住
    stuck_turn_angle: float = 30.0  # 卡住时强制转向角度
    position_change_threshold: float = 0.01  # 位置变化阈值（米）

    # 语义定位
    semantic_check_interval: int = 5  # 每 N 步进行一次 AI 视觉确认
    min_semantic_overlap: float = 0.3  # 语义标签重叠阈值

    # 降级策略
    pure_vision_spin_steps: int = 8  # 纯视觉模式原地旋转拍照数
    pure_vision_spin_angle: float = 45.0  # 纯视觉模式每步旋转角度

    # 上下文管理
    context_history_size: int = 5  # 保留最近 N 步历史


@dataclass
class OutputConfig:
    """输出相关配置"""
    # 输出目录
    output_base_dir: str = "Output"
    create_timestamp_subdir: bool = True

    # 日志
    log_to_file: bool = True
    log_level: str = "INFO"

    # 视频录制
    record_video: bool = True
    video_fps: int = 10

    # 快照
    save_snapshots: bool = True
    snapshot_interval: int = 10  # 每 N 步保存一张快照


@dataclass
class VLNConfig:
    """VLN 系统总配置"""
    api: APIConfig = field(default_factory=APIConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # 场景配置
    scene_path: str = ""
    navmesh_path: str = ""
    semantic_txt_path: str = ""

    # 用户目标
    user_goal: str = ""

    def load_from_env(self):
        """从环境变量加载 API 密钥（作为 config.json 的补充）"""
        if not self.api.openai_api_key:
            self.api.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api.claude_api_key:
            self.api.claude_api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def save_to_json(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        将配置保存到 JSON 文件

        Args:
            config_path: JSON 配置文件路径
        """
        data = self._to_dict()
        try:
            os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到: {config_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")

    def load_from_json(self, config_path: str = DEFAULT_CONFIG_PATH) -> bool:
        """
        从 JSON 文件加载配置

        Args:
            config_path: JSON 配置文件路径

        Returns:
            是否成功加载
        """
        if not os.path.exists(config_path):
            logger.info(f"配置文件不存在: {config_path}，使用默认配置")
            return False

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._from_dict(data)
            logger.info(f"配置已从 {config_path} 加载")
            return True
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return False

    def _to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "api": {
                "openai_api_key": self.api.openai_api_key,
                "openai_base_url": self.api.openai_base_url,
                "openai_model": self.api.openai_model,
                "openai_fallback_model": self.api.openai_fallback_model,
                "claude_api_key": self.api.claude_api_key,
                "claude_model": self.api.claude_model,
                "api_timeout": self.api.api_timeout,
                "api_max_retries": self.api.api_max_retries,
                "api_retry_delay": self.api.api_retry_delay,
                "max_tokens_per_response": self.api.max_tokens_per_response,
                "max_context_tokens": self.api.max_context_tokens,
            },
            "image": {
                "max_width": self.image.max_width,
                "max_height": self.image.max_height,
                "jpeg_quality": self.image.jpeg_quality,
                "encode_format": self.image.encode_format,
                "rgb_sensor_name": self.image.rgb_sensor_name,
                "camera_height": self.image.camera_height,
            },
            "navigation": {
                "forward_step_size": self.navigation.forward_step_size,
                "turn_angle": self.navigation.turn_angle,
                "max_steps": self.navigation.max_steps,
                "max_api_calls_per_step": self.navigation.max_api_calls_per_step,
                "stuck_threshold": self.navigation.stuck_threshold,
                "stuck_turn_angle": self.navigation.stuck_turn_angle,
                "position_change_threshold": self.navigation.position_change_threshold,
                "semantic_check_interval": self.navigation.semantic_check_interval,
                "min_semantic_overlap": self.navigation.min_semantic_overlap,
                "pure_vision_spin_steps": self.navigation.pure_vision_spin_steps,
                "pure_vision_spin_angle": self.navigation.pure_vision_spin_angle,
                "context_history_size": self.navigation.context_history_size,
            },
            "output": {
                "output_base_dir": self.output.output_base_dir,
                "create_timestamp_subdir": self.output.create_timestamp_subdir,
                "log_to_file": self.output.log_to_file,
                "log_level": self.output.log_level,
                "record_video": self.output.record_video,
                "video_fps": self.output.video_fps,
                "save_snapshots": self.output.save_snapshots,
                "snapshot_interval": self.output.snapshot_interval,
            },
        }

    def _from_dict(self, data: Dict[str, Any]):
        """从字典加载配置"""
        api_data = data.get("api", {})
        self.api.openai_api_key = api_data.get("openai_api_key", self.api.openai_api_key)
        self.api.openai_base_url = api_data.get("openai_base_url", self.api.openai_base_url)
        self.api.openai_model = api_data.get("openai_model", self.api.openai_model)
        self.api.openai_fallback_model = api_data.get("openai_fallback_model", self.api.openai_fallback_model)
        self.api.claude_api_key = api_data.get("claude_api_key", self.api.claude_api_key)
        self.api.claude_model = api_data.get("claude_model", self.api.claude_model)
        self.api.api_timeout = api_data.get("api_timeout", self.api.api_timeout)
        self.api.api_max_retries = api_data.get("api_max_retries", self.api.api_max_retries)
        self.api.api_retry_delay = api_data.get("api_retry_delay", self.api.api_retry_delay)
        self.api.max_tokens_per_response = api_data.get("max_tokens_per_response", self.api.max_tokens_per_response)
        self.api.max_context_tokens = api_data.get("max_context_tokens", self.api.max_context_tokens)

        img_data = data.get("image", {})
        self.image.max_width = img_data.get("max_width", self.image.max_width)
        self.image.max_height = img_data.get("max_height", self.image.max_height)
        self.image.jpeg_quality = img_data.get("jpeg_quality", self.image.jpeg_quality)
        self.image.encode_format = img_data.get("encode_format", self.image.encode_format)
        self.image.rgb_sensor_name = img_data.get("rgb_sensor_name", self.image.rgb_sensor_name)
        self.image.camera_height = img_data.get("camera_height", self.image.camera_height)

        nav_data = data.get("navigation", {})
        self.navigation.forward_step_size = nav_data.get("forward_step_size", self.navigation.forward_step_size)
        self.navigation.turn_angle = nav_data.get("turn_angle", self.navigation.turn_angle)
        self.navigation.max_steps = nav_data.get("max_steps", self.navigation.max_steps)
        self.navigation.max_api_calls_per_step = nav_data.get("max_api_calls_per_step", self.navigation.max_api_calls_per_step)
        self.navigation.stuck_threshold = nav_data.get("stuck_threshold", self.navigation.stuck_threshold)
        self.navigation.stuck_turn_angle = nav_data.get("stuck_turn_angle", self.navigation.stuck_turn_angle)
        self.navigation.position_change_threshold = nav_data.get("position_change_threshold", self.navigation.position_change_threshold)
        self.navigation.semantic_check_interval = nav_data.get("semantic_check_interval", self.navigation.semantic_check_interval)
        self.navigation.min_semantic_overlap = nav_data.get("min_semantic_overlap", self.navigation.min_semantic_overlap)
        self.navigation.pure_vision_spin_steps = nav_data.get("pure_vision_spin_steps", self.navigation.pure_vision_spin_steps)
        self.navigation.pure_vision_spin_angle = nav_data.get("pure_vision_spin_angle", self.navigation.pure_vision_spin_angle)
        self.navigation.context_history_size = nav_data.get("context_history_size", self.navigation.context_history_size)

        out_data = data.get("output", {})
        self.output.output_base_dir = out_data.get("output_base_dir", self.output.output_base_dir)
        self.output.create_timestamp_subdir = out_data.get("create_timestamp_subdir", self.output.create_timestamp_subdir)
        self.output.log_to_file = out_data.get("log_to_file", self.output.log_to_file)
        self.output.log_level = out_data.get("log_level", self.output.log_level)
        self.output.record_video = out_data.get("record_video", self.output.record_video)
        self.output.video_fps = out_data.get("video_fps", self.output.video_fps)
        self.output.save_snapshots = out_data.get("save_snapshots", self.output.save_snapshots)
        self.output.snapshot_interval = out_data.get("snapshot_interval", self.output.snapshot_interval)

    def is_openai_configured(self) -> bool:
        """检查 OpenAI API 是否已配置"""
        return bool(self.api.openai_api_key)

    def is_claude_configured(self) -> bool:
        """检查 Claude API 是否已配置"""
        return bool(self.api.claude_api_key)

    def is_api_configured(self) -> bool:
        """检查是否有可用的 API 配置"""
        return self.is_openai_configured() or self.is_claude_configured()


# 全局默认配置实例
default_config = VLNConfig()
