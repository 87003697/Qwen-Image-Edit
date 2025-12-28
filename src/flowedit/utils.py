"""
辅助函数模块
"""

import io
import base64

import numpy as np
import torch
from PIL import Image


def decode_base64_image(b64_str: str) -> Image.Image:
    """Base64 解码为 PIL Image"""
    image_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """PIL Image 编码为 Base64"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_if_needed(img: Image.Image, target_max: int = 1024) -> Image.Image:
    """等比缩放后居中贴到正方形画布"""
    w, h = img.size
    scale = target_max / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_max, target_max), (255, 255, 255))
    offset_x = (target_max - new_w) // 2
    offset_y = (target_max - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def encode_tensor_to_base64(tensor: torch.Tensor) -> str:
    """Tensor -> Base64 编码的 numpy 字节"""
    buffer = io.BytesIO()
    np.save(buffer, tensor.detach().cpu().numpy().astype(np.float32))
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

