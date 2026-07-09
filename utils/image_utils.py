"""
图片处理工具模块
提供图片编码、压缩、预处理等功能
"""

import base64
import io
import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def compress_image(
    image: np.ndarray,
    max_width: int = 1024,
    max_height: int = 768,
    quality: int = 85,
    encode_format: str = "jpeg",
) -> Tuple[bytes, Tuple[int, int]]:
    """
    压缩图片到指定尺寸和质量

    Args:
        image: RGB 图片数组 (H, W, 3)
        max_width: 最大宽度
        max_height: 最大高度
        quality: JPEG 压缩质量 (1-100)
        encode_format: 编码格式 (jpeg / webp / png)

    Returns:
        (压缩后的图片字节数据, (原始宽度, 原始高度))
    """
    pil_image = Image.fromarray(image)
    original_size = pil_image.size

    # 计算缩放比例，保持宽高比
    width, height = original_size
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

    # 编码为指定格式
    buffer = io.BytesIO()
    if encode_format == "jpeg":
        # JPEG 不支持 RGBA，确保是 RGB
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
    elif encode_format == "webp":
        pil_image.save(buffer, format="WEBP", quality=quality)
    else:
        pil_image.save(buffer, format="PNG", optimize=True)

    compressed_size = len(buffer.getvalue())
    logger.debug(
        f"图片压缩: {original_size[0]}x{original_size[1]} → "
        f"{pil_image.size[0]}x{pil_image.size[1]}, "
        f"大小: {compressed_size / 1024:.1f} KB"
    )

    return buffer.getvalue(), original_size


def image_to_base64(
    image: np.ndarray,
    max_width: int = 1024,
    max_height: int = 768,
    quality: int = 85,
    encode_format: str = "jpeg",
) -> Tuple[str, int, Tuple[int, int]]:
    """
    将图片编码为 Base64 字符串

    Args:
        image: RGB 图片数组 (H, W, 3)
        max_width: 最大宽度
        max_height: 最大高度
        quality: JPEG 压缩质量
        encode_format: 编码格式

    Returns:
        (Base64 编码字符串, 压缩后大小(字节), (原始宽度, 原始高度))
    """
    compressed_data, original_size = compress_image(
        image, max_width, max_height, quality, encode_format
    )
    base64_str = base64.b64encode(compressed_data).decode("utf-8")
    return base64_str, len(compressed_data), original_size


def get_image_format_from_array(image: np.ndarray) -> str:
    """
    根据图片数组推断合适的格式

    Args:
        image: 图片数组

    Returns:
        格式字符串: "jpeg" 或 "png"
    """
    # 如果有 alpha 通道，使用 PNG
    if image.shape[2] == 4:
        return "png"
    return "jpeg"


def resize_image(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """
    调整图片到指定尺寸（不保持宽高比）

    Args:
        image: 输入图片数组
        target_width: 目标宽度
        target_height: 目标高度

    Returns:
        调整后的图片数组
    """
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(pil_image)


def create_image_data_url(
    base64_str: str,
    encode_format: str = "jpeg",
) -> str:
    """
    创建 Data URL 格式的图片数据

    Args:
        base64_str: Base64 编码的图片数据
        encode_format: 图片格式

    Returns:
        Data URL 字符串
    """
    mime_type = {
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "png": "image/png",
    }.get(encode_format, "image/jpeg")

    return f"data:{mime_type};base64,{base64_str}"
