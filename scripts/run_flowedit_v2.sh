#!/usr/bin/env bash
# FlowEdit V2 服务调用脚本
# 使用方法: bash scripts/run_flowedit_v2.sh [options]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认参数
SOURCE_IMAGE="${SOURCE_IMAGE:-@dataset/normals/02_azi-45_dst-2.png}"
TARGET_IMAGE="${TARGET_IMAGE:-@dataset/images/02.jpg}"
PROMPT="${PROMPT:-Move the camera}"

API_HOST="${API_HOST:-localhost}"
API_PORT="${API_PORT:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs_api}"

# FlowEdit 参数
SEED="${SEED:-0}"
STEPS="${STEPS:-40}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
TRUE_CFG_SCALE_TGT="${TRUE_CFG_SCALE_TGT:-15.0}"
N_MIN="${N_MIN:-0}"
N_MAX="${N_MAX:-25}"
# 梯度计算开关
COMPUTE_SSIM_GRAD="${COMPUTE_SSIM_GRAD:-true}"
COMPUTE_LPIPS_GRAD="${COMPUTE_LPIPS_GRAD:-true}"
COMPUTE_LATENT_MSE_GRAD="${COMPUTE_LATENT_MSE_GRAD:-true}"

cd "$PROJECT_ROOT"

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qwen-image-edit

# 调用 Python API 客户端
ARGS=(
    --source-image "$SOURCE_IMAGE"
    --target-image "$TARGET_IMAGE"
    --prompt "$PROMPT"
    --api-url "http://${API_HOST}:${API_PORT}"
    --output-dir "$OUTPUT_DIR"
    --seed "$SEED"
    --steps "$STEPS"
    --guidance-scale "$GUIDANCE_SCALE"
    --true-cfg-scale-tgt "$TRUE_CFG_SCALE_TGT"
    --n-min "$N_MIN"
    --n-max "$N_MAX"
)

# 梯度相关开关
shopt -s nocasematch
if [[ "$COMPUTE_SSIM_GRAD" == "true" ]]; then
    ARGS+=(--compute-ssim-grad)
fi
if [[ "$COMPUTE_LPIPS_GRAD" == "true" ]]; then
    ARGS+=(--compute-lpips-grad)
fi
if [[ "$COMPUTE_LATENT_MSE_GRAD" == "true" ]]; then
    ARGS+=(--compute-latent-mse-grad)
fi
shopt -u nocasematch

python src/flowedit/flowedit_v2_api.py "${ARGS[@]}"
