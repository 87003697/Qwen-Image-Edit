"""
FlowEdit V2 API 调用客户端
"""

import argparse
import base64
import os
import sys
from datetime import datetime

import requests


def load_image_as_base64(path: str) -> str:
    """加载图像并编码为 Base64"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_edit_api(
    api_url: str,
    source_image: str,
    target_image: str,
    prompt: str,
    output_dir: str,
    seed: int = 0,
    steps: int = 40,
    guidance_scale: float = 1.0,
    true_cfg_scale_tgt: float = 15.0,
    n_min: int = 0,
    n_max: int = 25,
) -> str:
    """
    调用 FlowEdit API 进行图像编辑
    
    Args:
        api_url: API 服务地址
        source_image: 源图像路径
        target_image: 目标图像路径
        prompt: 编辑指令
        output_dir: 输出目录
        seed: 随机种子
        steps: 推理步数
        guidance_scale: Guidance scale
        true_cfg_scale_tgt: Target CFG Scale
        n_min: FlowEdit n_min
        n_max: FlowEdit n_max
    
    Returns:
        输出图像路径
    """
    # 检查文件是否存在
    if not os.path.exists(source_image):
        raise FileNotFoundError(f"Source image not found: {source_image}")
    if not os.path.exists(target_image):
        raise FileNotFoundError(f"Target image not found: {target_image}")
    
    # 加载图像
    print("  加载图像...")
    source_b64 = load_image_as_base64(source_image)
    target_b64 = load_image_as_base64(target_image)
    
    # 调用 API
    print("  发送请求...")
    response = requests.post(
        f"{api_url}/edit",
        json={
            "source_image": source_b64,
            "target_image": target_b64,
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "true_cfg_scale_tgt": true_cfg_scale_tgt,
            "n_min": n_min,
            "n_max": n_max,
        },
        timeout=300,
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"API call failed: {response.status_code}\n{response.text}")
    
    result = response.json()
    output_bytes = base64.b64decode(result["image"])
    
    # 保存输出
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"output_{timestamp}.png")
    
    with open(output_path, "wb") as f:
        f.write(output_bytes)
    
    print(f"  ✅ 成功！输出: {output_path}")
    print(f"  Seed: {result['seed']}")
    
    return output_path


def check_service(api_url: str) -> bool:
    """检查服务是否运行"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlowEdit V2 API 调用客户端")
    
    # 必需参数
    parser.add_argument("--source-image", required=True, help="源图像路径")
    parser.add_argument("--target-image", required=True, help="目标图像路径")
    parser.add_argument("--prompt", required=True, help="编辑指令")
    
    # 服务配置
    parser.add_argument("--api-url", default="http://localhost:8000", help="API 服务地址")
    parser.add_argument("--output-dir", default="outputs_api", help="输出目录")
    
    # FlowEdit 参数
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--steps", type=int, default=40, help="推理步数")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--true-cfg-scale-tgt", type=float, default=15.0, help="Target CFG Scale")
    parser.add_argument("--n-min", type=int, default=0, help="FlowEdit n_min")
    parser.add_argument("--n-max", type=int, default=25, help="FlowEdit n_max")
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    print("==========================================")
    print("FlowEdit V2 API 调用")
    print("==========================================")
    print(f"API: {args.api_url}")
    print(f"Source: {args.source_image}")
    print(f"Target: {args.target_image}")
    print(f"Prompt: {args.prompt}")
    print("==========================================")
    
    # 检查服务
    print("\n[1/2] 检查服务状态...")
    if not check_service(args.api_url):
        print("[Error] 服务未运行，请先启动服务:")
        print("  bash scripts/service_flowedit_v2.sh")
        sys.exit(1)
    print("  ✅ 服务运行中")
    
    # 调用 API
    print("\n[2/2] 调用编辑 API...")
    try:
        output_path = call_edit_api(
            api_url=args.api_url,
            source_image=args.source_image,
            target_image=args.target_image,
            prompt=args.prompt,
            output_dir=args.output_dir,
            seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            true_cfg_scale_tgt=args.true_cfg_scale_tgt,
            n_min=args.n_min,
            n_max=args.n_max,
        )
        print("\n==========================================")
        print(f"输出: {output_path}")
        print("==========================================")
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

