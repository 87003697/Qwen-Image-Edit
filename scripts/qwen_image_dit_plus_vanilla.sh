#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录 (scripts/)
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
# 获取项目根目录 (scripts/ 的上一级)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 设置 PYTHONPATH 包含项目根目录，以便能找到 pipelines 模块
PYTHONPATH="$PROJECT_ROOT" python "$SCRIPT_DIR/qwen_image_dit_plus_vanilla.py" \
  --image \
  "@dataset/normals/01_azi-0_dst-2.png" \
  "@dataset/images/01.jpg" \
  --prompt "Add the appearance of Image 2 to the normal map of Image 1" \
  --output "outputs/output_image_edit_plus.png" \
  --steps 40 \
  --true-cfg-scale 4.0 \
  --guidance-scale 1.0 \
  --negative-prompt "" \
  --seed 0 \
  --device cuda \
  --strength 0 \
  --init-image-index 0

  # 