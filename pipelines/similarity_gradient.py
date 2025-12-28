"""
SSIM/LPIPS 相似度计算及梯度模块
"""

import torch
import numpy as np
from PIL import Image
from pytorch_msssim import ssim
import lpips


def create_lpips_model(device: str = "cuda") -> lpips.LPIPS:
    """创建 LPIPS 模型"""
    return lpips.LPIPS(net='alex').to(device).eval()


def pil_to_tensor(img: Image.Image, device: str = "cuda") -> torch.Tensor:
    """PIL Image -> [1, 3, H, W] tensor, 范围 [0, 1]"""
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]


def compute_ssim_gradient(
    source_img: Image.Image,
    output_img: Image.Image,
    device: str = "cuda",
) -> dict:
    """
    计算 SSIM 值及其相对于 output 的梯度
    
    Returns:
        ssim: float 值 (0~1, 越高越相似)
        ssim_grad: [3, H, W] 梯度张量
    """
    src = pil_to_tensor(source_img, device)  # [1, 3, H, W]
    out = pil_to_tensor(output_img, device).requires_grad_(True)  # [1, 3, H, W]
    
    ssim_val = ssim(src, out, data_range=1.0, size_average=True)  # scalar
    ssim_loss = 1.0 - ssim_val  # scalar
    
    ssim_grad = torch.autograd.grad(ssim_loss, out)[0].squeeze(0)  # [3, H, W]
    
    return {
        "ssim": ssim_val.item(),
        "ssim_grad": ssim_grad,
    }


def compute_lpips_gradient(
    source_img: Image.Image,
    output_img: Image.Image,
    lpips_model: lpips.LPIPS,
    device: str = "cuda",
) -> dict:
    """
    计算 LPIPS 值及其相对于 output 的梯度
    
    Returns:
        lpips: float 值 (越低越相似)
        lpips_grad: [3, H, W] 梯度张量
    """
    src = pil_to_tensor(source_img, device)  # [1, 3, H, W]
    out = pil_to_tensor(output_img, device).requires_grad_(True)  # [1, 3, H, W]
    
    # LPIPS 期望 [-1, 1] 输入
    lpips_val = lpips_model(src * 2 - 1, out * 2 - 1).squeeze()  # scalar
    
    lpips_grad = torch.autograd.grad(lpips_val, out)[0].squeeze(0)  # [3, H, W]
    
    return {
        "lpips": lpips_val.item(),
        "lpips_grad": lpips_grad,
    }
