#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录 (scripts/)
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
# 获取项目根目录 (scripts/ 的上一级)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 设置 PYTHONPATH 包含项目根目录，以便能找到 pipelines 模块
PYTHONPATH="$PROJECT_ROOT" python "$SCRIPT_DIR/qwen_image_dit_vanilla.py" \
  --image "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png" \
  --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant colors" \
  --output "outputs/test_qwen_image_dit_vanilla.png" \
  --steps 20 \
  --true-cfg-scale 4.0 \
  --guidance-scale 1.0 \
  --negative-prompt "" \
  --seed 42 \
  --device cuda
