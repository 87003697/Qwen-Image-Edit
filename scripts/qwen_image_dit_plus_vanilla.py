"""
Qwen-Image-Edit-2509 示例脚本（支持单/多图输入，argparse 可配置）。
依赖：pip install git+https://github.com/huggingface/diffusers
"""

import argparse
import os
import textwrap
from io import BytesIO
from typing import List, Sequence

import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from pipelines.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline


def _load_image(path_or_url: str) -> Image.Image:
    def _to_rgb_with_white_bg(img: Image.Image) -> Image.Image:
        if img.mode == "RGBA":
            # 仅在存在透明通道时叠加白色背景
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            return Image.alpha_composite(bg, img).convert("RGB")
        return img.convert("RGB")

    if os.path.exists(path_or_url):
        return _to_rgb_with_white_bg(Image.open(path_or_url))

    resp = requests.get(path_or_url, timeout=30)
    resp.raise_for_status()
    return _to_rgb_with_white_bg(Image.open(BytesIO(resp.content)))


def _resize_if_needed(img: Image.Image, target_max: int = 1024) -> Image.Image:
    """
    等比缩放后居中贴到正方形画布（target_max x target_max）。
    - 若任一边大于 target_max：先按最长边等比缩放到 target_max
    - 若原图已不大于 target_max：直接按比例放大至最长边 = target_max
    返回：尺寸为 (target_max, target_max)，内容等比居中，无拉伸
    """
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


def _make_triptych_with_prompt(
    images: Sequence[Image.Image],
    output_image: Image.Image,
    prompt: str,
    output_path: str,
    strength: float | None = None,
) -> str | None:
    """拼接三图一行，底部附上 prompt，保存可视化。"""
    if len(images) < 2:
        return None

    def _resize_to_height(im: Image.Image, target_h: int = 512) -> Image.Image:
        w, h = im.size
        scale = target_h / h
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return im.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img_a = _resize_to_height(images[0])
    img_b = _resize_to_height(images[1])
    img_c = _resize_to_height(output_image)

    margin = 16
    font = ImageFont.load_default()

    # 预估文本高度
    prompt_text = prompt if strength is None else f"{prompt} | strength={strength:.2f}"
    wrapped_prompt = textwrap.fill(prompt_text, width=80)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10), (255, 255, 255)))
    text_bbox = dummy_draw.textbbox((0, 0), wrapped_prompt, font=font)
    text_h = (text_bbox[3] - text_bbox[1]) + margin * 2

    total_w = img_a.width + img_b.width + img_c.width + margin * 4
    total_h = img_a.height + margin * 2 + text_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    x = margin
    for im in (img_a, img_b, img_c):
        canvas.paste(im, (x, margin))
        x += im.width + margin

    draw = ImageDraw.Draw(canvas)
    text_x, text_y = margin, img_a.height + margin
    draw.text((text_x, text_y), wrapped_prompt, fill=(0, 0, 0), font=font)

    vis_path = f"{os.path.splitext(output_path)[0]}_vis.png"
    canvas.save(vis_path)
    return os.path.abspath(vis_path)


def run_edit(
    images: Sequence[Image.Image],
    prompt: str,
    output_path: str,
    steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    negative_prompt: str,
    seed: int,
    device: str,
    strength: float,
    init_image_index: int,
) -> str:
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)  # pipeline 内部模型参数张量形状保持默认
    pipe.set_progress_bar_config(disable=None)

    inputs = {
        "image": list(images),
        "prompt": prompt,
        "generator": torch.manual_seed(seed),  # 生成器不涉及张量形状
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
        "strength": strength,
        "init_image_index": init_image_index,
    }

    with torch.inference_mode():
        output = pipe(**inputs)
        output_image = output.images[0]

    output_image.save(output_path)
    vis_path = _make_triptych_with_prompt(images, output_image, prompt, output_path, strength=strength)
    if vis_path:
        print(f"visualization saved at: {vis_path}")
    return os.path.abspath(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Edit-2509 推理脚本（单/多图编辑）"
    )
    parser.add_argument(
        "-i",
        "--image",
        nargs="+",
        required=True,
        help="本地路径或 URL，可传多张图（推荐 1~3 张）",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="编辑描述",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output_image_edit_plus.png",
        help="输出文件路径",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="采样步数（num_inference_steps）",
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="true_cfg_scale",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="guidance_scale",
    )
    parser.add_argument(
        "--negative-prompt",
        default=" ",
        help="负向提示，默认空格（与 README 一致）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="运行设备，默认 cuda",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.0,
        help="img2img 强度，0 表示随机起点（不参考输入图），(0,1] 越大改动越大",
    )
    parser.add_argument(
        "--init-image-index",
        type=int,
        default=0,
        help="多图输入时，选择哪一张作为起始 latent（strength>0 生效）",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    images = [_resize_if_needed(_load_image(item)) for item in args.image]
    save_path = run_edit(
        images=images,
        prompt=args.prompt,
        output_path=args.output,
        steps=args.steps,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        device=args.device,
        strength=args.strength,
        init_image_index=args.init_image_index,
    )
    print(f"image saved at: {save_path}")


if __name__ == "__main__":
    main()
