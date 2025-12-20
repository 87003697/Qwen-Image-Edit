#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
# 获取脚本所在目录 (scripts/)
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
# 获取项目根目录 (scripts/ 的上一级)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 示例输入图片
# SOURCE_IMAGE: 用于编辑的源图像
SOURCE_IMAGE="@dataset/normals/02_azi-45_dst-2.png"
# TARGET_IMAGE: 用作 image prompt 的目标图像
TARGET_IMAGE="@dataset/images/02.jpg"

# --- 定义参数变量 ---
CFG_TGT="15"
N_MAX="25"

# --- 构建输出文件名 ---
OUT_NAME="output_Tgt-${CFG_TGT}_Max-${N_MAX}.png"
OUTPUT_PATH="outputs_v2-3/${OUT_NAME}"

# 设置 PYTHONPATH 包含项目根目录，以便能找到 pipelines 模块
PYTHONPATH="$PROJECT_ROOT" python "$SCRIPT_DIR/qwen_image_dit_plus_flowedit_v2.py" \
  --source-image "$SOURCE_IMAGE" \
  --target-image "$TARGET_IMAGE" \
  --prompt "Move the camera" \
  --output "$OUTPUT_PATH" \
  --steps 40 \
  --true-cfg-scale-tgt "$CFG_TGT" \
  --guidance-scale 1.0 \
  --negative-prompt "" \
  --seed 0 \
  --device cuda \
  --n-max "$N_MAX"
