"""
无 UI 的 FlowEdit API 服务器
使用自定义 pipeline_qwenimage_edit_plus_flowedit_v2
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipelines.pipeline_qwenimage_edit_plus_flowedit_v2 import QwenImageEditPlusPipeline
from pipelines.similarity_gradient import (
    compute_ssim_gradient,
    compute_lpips_gradient,
    compute_latent_mse_gradient,
    create_lpips_model,
)
from src.flowedit.utils import (
    decode_base64_image,
    encode_image_to_base64,
    resize_if_needed,
    encode_tensor_to_base64,
)

# ============ dtype / device 配置 ============
MODEL_DTYPE = torch.bfloat16  # pipeline / VAE
METRIC_DTYPE = torch.float32  # SSIM / LPIPS
METRIC_DEVICE = "cuda"        # 同一 GPU


# ============ 全局模型实例 ============
pipe: Optional[QwenImageEditPlusPipeline] = None
lpips_model = None


def load_pipeline(device: str = "cuda"):
    """加载 Pipeline 和 LPIPS 模型"""
    global pipe, lpips_model
    if pipe is None:
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=MODEL_DTYPE,
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)
        print(f"Pipeline loaded on {device}")
    
    if lpips_model is None:
        # LPIPS 使用 fp32，放在同一 GPU 上计算相似度/梯度
        lpips_model = create_lpips_model(METRIC_DEVICE)
        print("LPIPS model loaded")
    
    return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    device = os.environ.get("DEVICE", "cuda")
    load_pipeline(device)
    yield


app = FastAPI(
    title="Qwen-Image FlowEdit API",
    description="无 UI 的 FlowEdit 图像编辑 API",
    lifespan=lifespan,
)


# ============ 请求/响应模型 ============
class EditRequest(BaseModel):
    """编辑请求"""
    source_image: str = Field(..., description="Base64 编码的源图像（用于编辑）")
    target_image: str = Field(..., description="Base64 编码的目标图像（用作 image prompt）")
    prompt: str = Field(..., description="编辑指令")
    
    # 可选参数
    seed: int = Field(default=0, description="随机种子")
    steps: int = Field(default=40, description="推理步数")
    guidance_scale: float = Field(default=1.0, description="Guidance scale")
    negative_prompt: str = Field(default="", description="负向提示词")
    
    # FlowEdit 特有参数
    true_cfg_scale_tgt: float = Field(default=15.0, description="Target 分支的 CFG Scale")
    n_min: int = Field(default=0, description="FlowEdit n_min")
    n_max: int = Field(default=25, description="FlowEdit n_max")
    
    # 梯度计算（独立控制）
    compute_ssim_grad: bool = Field(default=False, description="是否计算 SSIM 梯度")
    compute_lpips_grad: bool = Field(default=False, description="是否计算 LPIPS 梯度")
    compute_latent_mse_grad: bool = Field(default=False, description="是否计算 Latent MSE 梯度")


class EditResponse(BaseModel):
    """编辑响应"""
    image: str = Field(..., description="Base64 编码的输出图像")
    seed: int = Field(..., description="使用的随机种子")
    # 梯度相关字段
    ssim: Optional[float] = Field(default=None, description="SSIM 值")
    lpips: Optional[float] = Field(default=None, description="LPIPS 值")
    latent_mse: Optional[float] = Field(default=None, description="Latent MSE 值")
    ssim_grad: Optional[str] = Field(default=None, description="Base64 编码的 SSIM 梯度")
    lpips_grad: Optional[str] = Field(default=None, description="Base64 编码的 LPIPS 梯度")
    latent_mse_grad: Optional[str] = Field(default=None, description="Base64 编码的 Latent MSE 梯度")


# ============ API 端点 ============
@app.post("/edit", response_model=EditResponse)
async def edit_image(request: EditRequest):
    """图像编辑接口"""
    global pipe
    
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")
    
    try:
        # 1. 解码图像
        source_image = decode_base64_image(request.source_image)
        target_image = decode_base64_image(request.target_image)
        
        # 2. 预处理（等比缩放）
        source_image = resize_if_needed(source_image)
        target_image = resize_if_needed(target_image)
        
        # 3. 构建推理参数
        inputs = {
            "image_src": source_image,
            "image_tgt": target_image,
            "prompt": request.prompt,
            "generator": torch.manual_seed(request.seed),
            "negative_prompt": request.negative_prompt or " ",
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "num_images_per_prompt": 1,
            # FlowEdit 参数
            "true_cfg_scale_tgt": request.true_cfg_scale_tgt,
            "n_min": request.n_min,
            "n_max": request.n_max,
        }
        
        # 4. 执行推理
        with torch.inference_mode():
            output = pipe(**inputs)
            output_image = output.images[0]
            edited_latent = output.latents  # [B, seq_len, C] packed latent
        
        # 5. 计算梯度（独立控制）
        ssim_val, lpips_val, latent_mse_val = None, None, None
        ssim_grad_b64, lpips_grad_b64, latent_mse_grad_b64 = None, None, None
        device_model = str(next(pipe.transformer.parameters()).device)
        device_metric = METRIC_DEVICE
        
        # 推理结果 latent 与模型 dtype 对齐
        edited_latent = edited_latent.to(device_model, dtype=MODEL_DTYPE)  # [B, seq_len, C] bf16
        
        if request.compute_ssim_grad:
            # SSIM 在 GPU 上用 fp32 计算，避免与 bf16 权重混用
            ssim_result = compute_ssim_gradient(source_image, output_image, device_metric)
            ssim_val = ssim_result["ssim"]
            ssim_grad_b64 = encode_tensor_to_base64(ssim_result["ssim_grad"])
        
        if request.compute_lpips_grad:
            # LPIPS 同样在 GPU 上用 fp32 计算
            lpips_result = compute_lpips_gradient(source_image, output_image, lpips_model, device_metric)
            lpips_val = lpips_result["lpips"]
            lpips_grad_b64 = encode_tensor_to_base64(lpips_result["lpips_grad"])
        
        if request.compute_latent_mse_grad:
            # 封装 pipeline 的 preprocess、encode 和 unpack 函数
            def preprocess_fn(img: Image.Image) -> torch.Tensor:
                """使用 pipeline 的图像预处理"""
                w, h = img.size
                return pipe.image_processor.preprocess(img, h, w).to(
                    device_model, dtype=MODEL_DTYPE
                )  # [1, 3, H, W] bf16, 范围 [-1, 1]
            
            def encode_fn(img_tensor: torch.Tensor) -> torch.Tensor:
                """使用 pipeline 的 VAE 编码 + 标准化"""
                return pipe._encode_vae_image(img_tensor, generator=None)  # [B, C, 1, h, w]
            
            def unpack_fn(latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
                """使用 pipeline 的 unpack 函数"""
                return pipe._unpack_latents(latent, height, width, pipe.vae_scale_factor)  # [B, C, 1, h, w]
            
            latent_mse_result = compute_latent_mse_gradient(
                source_image, edited_latent, preprocess_fn, encode_fn, unpack_fn, device_model
            )
            latent_mse_val = latent_mse_result["mse"]
            latent_mse_grad_b64 = encode_tensor_to_base64(latent_mse_result["mse_grad"])
        
        # 6. 编码输出
        output_b64 = encode_image_to_base64(output_image)
        
        return EditResponse(
            image=output_b64,
            seed=request.seed,
            ssim=ssim_val,
            lpips=lpips_val,
            latent_mse=latent_mse_val,
            ssim_grad=ssim_grad_b64,
            lpips_grad=lpips_grad_b64,
            latent_mse_grad=latent_mse_grad_b64,
        )
        
    except Exception as e:
        import traceback
        
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "model_loaded": pipe is not None,
        "device": str(next(pipe.transformer.parameters()).device) if pipe else None,
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(app, host=host, port=port)