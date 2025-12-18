#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录 (scripts/)
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
# 获取项目根目录 (scripts/ 的上一级)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 示例输入图片
# IMAGE_REF: 结构图/Normal Map/底图
IMAGE_REF="@dataset/images/01.jpg"
# IMAGE_GEN: 风格图/材质图
IMAGE_GEN="@dataset/normals/01_azi-0_dst-2.png"

# --- 定义参数变量 ---
CFG_SRC="1"
CFG_TGT="13.5"
N_MAX="30"

# --- 构建输出文件名 ---
OUT_NAME="output_Src-${CFG_SRC}_Tgt-${CFG_TGT}_Max-${N_MAX}.png"
OUTPUT_PATH="outputs/${OUT_NAME}"

# 设置 PYTHONPATH 包含项目根目录，以便能找到 pipelines 模块
PYTHONPATH="$PROJECT_ROOT" python "$SCRIPT_DIR/qwen_image_dit_plus_flowedit.py" \
  --image "$IMAGE_REF" "$IMAGE_GEN" \
  --prompt "Move the camera to a normal view" \
  --source-prompt "Reconstruct the normal map" \
  --output "$OUTPUT_PATH" \
  --steps 40 \
  --true-cfg-scale-src "$CFG_SRC" \
  --true-cfg-scale-tgt "$CFG_TGT" \
  --guidance-scale 1.0 \
  --negative-prompt "" \
  --seed 0 \
  --device cuda \
  --init-image-index 1 \
  --target-indices 0 \
  --source-indices 1 \
  --n-max "$N_MAX"
