"""
无 UI 的 FlowEdit API 服务器
使用自定义 pipeline_qwenimage_edit_plus_flowedit_v2
"""

import io
import os
import base64
from typing import Optional
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 使用你的自定义 FlowEdit pipeline
from pipelines.pipeline_qwenimage_edit_plus_flowedit_v2 import QwenImageEditPlusPipeline


# ============ 全局模型实例 ============
pipe: Optional[QwenImageEditPlusPipeline] = None


def load_pipeline(device: str = "cuda"):
    """加载 Pipeline"""
    global pipe
    if pipe is None:
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=torch.bfloat16,
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)
        print(f"Pipeline loaded on {device}")
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


class EditResponse(BaseModel):
    """编辑响应"""
    image: str = Field(..., description="Base64 编码的输出图像")
    seed: int = Field(..., description="使用的随机种子")


# ============ 辅助函数 ============
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
        
        # 5. 编码输出
        output_b64 = encode_image_to_base64(output_image)
        
        return EditResponse(
            image=output_b64,
            seed=request.seed,
        )
        
    except Exception as e:
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