"""
Qwen-Image-Edit-2509 FlowEdit 示例脚本。
依赖：pip install git+https://github.com/huggingface/diffusers
"""

import argparse
import os
import textwrap
from io import BytesIO

import requests
import torch
from PIL import Image, ImageDraw, ImageFont
# Import from local file
from pipelines.pipeline_qwenimage_edit_plus_flowedit_v2 import QwenImageEditPlusPipeline


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
    source_image: Image.Image,
    target_image: Image.Image,
    output_image: Image.Image,
    prompt: str,
    output_path: str,
    tgt_cfg: float = 5.5,
    n_max: int = 20,
) -> str | None:
    """拼接三图一行（source, target, output），底部附上 prompt，保存可视化。"""

    def _resize_to_height(im: Image.Image, target_h: int = 512) -> Image.Image:
        w, h = im.size
        scale = target_h / h
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return im.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Visualization: Show source image, target image (for prompt), and output
    img_a = _resize_to_height(target_image)
    img_b = _resize_to_height(source_image)
    img_c = _resize_to_height(output_image)

    margin = 16
    font = ImageFont.load_default()

    # 预估文本高度
    lines = []
    lines.extend(textwrap.wrap(f"Prompt: {prompt}", width=80))
    param_line = f"Tgt CFG: {tgt_cfg:.1f} | n_max: {n_max}"
    lines.append(param_line)
    
    final_text = "\n".join(lines)

    dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10), (255, 255, 255)))
    text_bbox = dummy_draw.textbbox((0, 0), final_text, font=font)
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
    draw.text((text_x, text_y), final_text, fill=(0, 0, 0), font=font)

    vis_path = f"{os.path.splitext(output_path)[0]}_vis.png"
    canvas.save(vis_path)
    return os.path.abspath(vis_path)


def run_edit(
    source_image: Image.Image,
    target_image: Image.Image,
    prompt: str,
    output_path: str,
    steps: int,
    guidance_scale: float,
    negative_prompt: str,
    seed: int,
    device: str,
    # FlowEdit specific
    true_cfg_scale_tgt: float,
    n_min: int,
    n_max: int,
) -> str:
    # Load pipeline from pre-trained, but we want to use the class from the local file
    # Since we imported QwenImageEditPlusPipeline from the local file, 
    # .from_pretrained will load weights into this class structure.
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)

    inputs = {
        "image_src": source_image,  # 用于编辑的源图像
        "image_tgt": target_image,  # 用作 image prompt 的目标图像
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
        # FlowEdit Params
        "true_cfg_scale_tgt": true_cfg_scale_tgt,
        "n_min": n_min,
        "n_max": n_max,
    }

    with torch.inference_mode():
        output = pipe(**inputs)
        output_image = output.images[0]

    # output_image.save(output_path)
    vis_path = _make_triptych_with_prompt(
        source_image, 
        target_image,
        output_image, 
        prompt, 
        output_path, 
        tgt_cfg=true_cfg_scale_tgt,
        n_max=n_max
    )
    if vis_path:
        print(f"visualization saved at: {vis_path}")
    return os.path.abspath(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Edit-2509 FlowEdit 推理脚本 (简化版)"
    )
    parser.add_argument(
        "--source-image",
        required=True,
        help="源图像路径或 URL (用于编辑的图片)",
    )
    parser.add_argument(
        "--target-image",
        required=True,
        help="目标图像路径或 URL (用作 image prompt 的图片)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="目标编辑描述 (Target Prompt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output_flowedit.png",
        help="输出文件路径",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="采样步数（num_inference_steps）",
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
        help="负向提示",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="运行设备",
    )
    
    # FlowEdit Arguments
    parser.add_argument(
        "--true-cfg-scale-tgt",
        type=float,
        default=5.5,
        help="Target 分支的 CFG Scale",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=0,
        help="FlowEdit n_min",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=20,
        help="FlowEdit n_max",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 加载并预处理两张图片
    source_image = _resize_if_needed(_load_image(args.source_image))
    target_image = _resize_if_needed(_load_image(args.target_image))

    save_path = run_edit(
        source_image=source_image,
        target_image=target_image,
        prompt=args.prompt,
        output_path=args.output,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        device=args.device,
        # FlowEdit
        true_cfg_scale_tgt=args.true_cfg_scale_tgt,
        n_min=args.n_min,
        n_max=args.n_max,
    )
    print(f"image saved at: {save_path}")


if __name__ == "__main__":
    main()
