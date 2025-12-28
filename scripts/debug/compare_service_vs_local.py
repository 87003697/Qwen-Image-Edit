#!/usr/bin/env python
"""
与线上 service 返回的 SSIM/LPIPS 及梯度进行对比的调试脚本
使用与 scripts/run_flowedit_v2.sh 相同的默认输入
"""

import os
import io
import base64
import json
import numpy as np
import requests
import torch
from PIL import Image
from pytorch_msssim import ssim
import lpips


# ============ 默认配置（与 scripts/run_flowedit_v2.sh 保持一致） ============
DEFAULT_SOURCE = "@dataset/normals/02_azi-45_dst-2.png"
DEFAULT_TARGET = "@dataset/images/02.jpg"
DEFAULT_PROMPT = "Move the camera"
API_HOST = os.environ.get("API_HOST", "localhost")
API_PORT = int(os.environ.get("API_PORT", 8000))
API_URL = f"http://{API_HOST}:{API_PORT}"


# ============ 工具函数 ============
def pil_to_tensor(img: Image.Image, device: str) -> torch.Tensor:
    """PIL -> [1,3,H,W] float32 0~1"""
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_ssim_grad_local(src_img: Image.Image, out_img: Image.Image, device: str):
    """本地 SSIM 值与梯度（不依赖 pipelines.similarity_gradient）"""
    with torch.enable_grad():
        src = pil_to_tensor(src_img, device)
        out = pil_to_tensor(out_img, device).requires_grad_(True)
        val = ssim(src, out, data_range=1.0, size_average=True)  # scalar
        loss = 1.0 - val
        grad = torch.autograd.grad(loss, out)[0].squeeze(0)  # [3,H,W]
    return {"ssim": val.item(), "ssim_grad": grad}


def compute_lpips_grad_local(src_img: Image.Image, out_img: Image.Image, lpips_model, device: str):
    """本地 LPIPS 值与梯度（不依赖 pipelines.similarity_gradient）"""
    with torch.enable_grad():
        src = pil_to_tensor(src_img, device)
        out = pil_to_tensor(out_img, device).requires_grad_(True)
        val = lpips_model(src * 2 - 1, out * 2 - 1).squeeze()  # scalar
        grad = torch.autograd.grad(val, out)[0].squeeze(0)  # [3,H,W]
    return {"lpips": val.item(), "lpips_grad": grad}


def decode_img_from_b64(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def decode_grad_from_b64(b64_str: str):
    if not b64_str:
        return None
    arr = np.load(io.BytesIO(base64.b64decode(b64_str)))
    return torch.from_numpy(arr)


def norm(t: torch.Tensor):
    return t.float().norm().item() if t is not None else None


def diff_norm(a: torch.Tensor, b: torch.Tensor):
    if a is None or b is None:
        return None
    return (a.to(b.device) - b).float().norm().item()


def main():
    # 输入路径，可通过环境变量覆盖，与 run_flowedit_v2.sh 同名
    source_path = os.environ.get("SOURCE_IMAGE", DEFAULT_SOURCE)
    target_path = os.environ.get("TARGET_IMAGE", DEFAULT_TARGET)
    prompt = os.environ.get("PROMPT", DEFAULT_PROMPT)
    device_metric = os.environ.get("METRIC_DEVICE", "cuda")

    # 读取并编码图片
    with open(source_path, "rb") as f:
        source_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(target_path, "rb") as f:
        target_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 请求服务
    payload = {
        "source_image": source_b64,
        "target_image": target_b64,
        "prompt": prompt,
        "seed": 0,
        "steps": 40,
        "guidance_scale": 1.0,
        "true_cfg_scale_tgt": 15.0,
        "n_min": 0,
        "n_max": 25,
        "compute_ssim_grad": True,
        "compute_lpips_grad": True,
        "compute_latent_mse_grad": False,
    }
    print(f"Calling service {API_URL}/edit ...")
    resp = requests.post(f"{API_URL}/edit", json=payload, timeout=300)
    resp.raise_for_status()
    result = resp.json()
    print("Service call done.")

    # 解析服务输出
    out_img = decode_img_from_b64(result["image"])
    ssim_srv = result.get("ssim")
    lpips_srv = result.get("lpips")
    ssim_grad_srv = decode_grad_from_b64(result.get("ssim_grad"))
    lpips_grad_srv = decode_grad_from_b64(result.get("lpips_grad"))

    # 本地重新计算（独立实现）
    lpips_model = lpips.LPIPS(net="alex").to(device_metric).eval()
    src_img = Image.open(source_path).convert("RGB")
    tgt_img = Image.open(target_path).convert("RGB")

    ssim_local = compute_ssim_grad_local(src_img, out_img, device_metric)
    lpips_local = compute_lpips_grad_local(src_img, out_img, lpips_model, device_metric)

    # 打印对比
    print("\n==== SSIM ====")
    print("service :", ssim_srv)
    print("local   :", ssim_local['ssim'])
    print("grad‖srv‖:", norm(ssim_grad_srv))
    print("grad‖loc‖:", norm(ssim_local["ssim_grad"]))
    print("grad‖diff‖:", diff_norm(ssim_grad_srv, ssim_local["ssim_grad"]))

    print("\n==== LPIPS ====")
    print("service :", lpips_srv)
    print("local   :", lpips_local['lpips'])
    print("grad‖srv‖:", norm(lpips_grad_srv))
    print("grad‖loc‖:", norm(lpips_local["lpips_grad"]))
    print("grad‖diff‖:", diff_norm(lpips_grad_srv, lpips_local["lpips_grad"]))


if __name__ == "__main__":
    main()

