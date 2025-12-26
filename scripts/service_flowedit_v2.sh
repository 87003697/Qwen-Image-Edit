#!/usr/bin/env bash
# FlowEdit V2 服务后台启动脚本
# 使用方法: bash scripts/run_flowedit_v2.sh

set -euo pipefail

# 获取项目根目录
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 配置参数
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export DEVICE="${DEVICE:-cuda}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/flowedit_service_gpu${CUDA_VISIBLE_DEVICES}.log"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "FlowEdit V2 服务启动"
echo "=========================================="
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Device: $DEVICE"
echo "Address: http://$HOST:$PORT"
echo "Log: $LOG_FILE"
echo "=========================================="

cd "$PROJECT_ROOT"

# 激活 conda 环境并启动服务
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qwen-image-edit

PYTHONPATH="$PROJECT_ROOT" nohup python src/flowedit/flowedit_v2_service.py > "$LOG_FILE" 2>&1 &

echo ""
echo "✅ 服务已后台启动 (PID: $!)"
echo ""
echo "查看日志:  tail -f $LOG_FILE"
echo "健康检查:  curl http://localhost:$PORT/health"
echo "停止服务:  pkill -f flowedit_v2_service.py"
echo ""

