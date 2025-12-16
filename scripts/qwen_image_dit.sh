#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录 (scripts/)
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
# 获取项目根目录 (scripts/ 的上一级)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 设置 PYTHONPATH 包含项目根目录，以便能找到 pipelines 模块
PYTHONPATH="$PROJECT_ROOT" python "$SCRIPT_DIR/qwen_image_dit.py" \
  --image \
  "/home/zhiyuan_ma/code/Qwen-Image-Edit/@dataset/normals/direct3d_s2/0.png" \
  "/home/zhiyuan_ma/code/Qwen-Image-Edit/@dataset/images/direct3d_s2/0.png" \
  --prompt "Create a concept design image for 3D modeling shape visualization, matching Figure 1 in overall appearance. Keep Figure 1's color palette and visualization camera/viewpoint, and incorporate key concept-design details from Figure 2." \
  --output "output_image_edit_plus.png" \
  --steps 40 \
  --true-cfg-scale 4.0 \
  --guidance-scale 1.0 \
  --negative-prompt " " \
  --seed 0 \
  --device cuda \
  --strength 0.4 \
  --init-image-index 0