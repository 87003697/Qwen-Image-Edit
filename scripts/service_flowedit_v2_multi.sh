#!/usr/bin/env bash
# FlowEdit V2 多 GPU 服务后台启动脚本
# 使用方法: bash scripts/service_flowedit_v2_multi.sh

set -euo pipefail

# 获取项目根目录
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 配置参数
GPUS=(4 5 6 7)           # 使用的 GPU 列表
BASE_PORT=8001           # 起始端口，GPU 0 用 8001，GPU 1 用 8002，以此类推
export DEVICE="${DEVICE:-cuda}"
export HOST="${HOST:-0.0.0.0}"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "FlowEdit V2 多 GPU 服务启动"
echo "=========================================="
echo "GPUs: ${GPUS[*]}"
echo "端口: $((BASE_PORT + GPUS[0])) - $((BASE_PORT + GPUS[-1]))"
echo "=========================================="

cd "$PROJECT_ROOT"

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qwen-image-edit

# 启动各 GPU 实例
for gpu in "${GPUS[@]}"; do
    port=$((BASE_PORT + gpu))
    log_file="$LOG_DIR/flowedit_service_gpu${gpu}.log"
    
    CUDA_VISIBLE_DEVICES="$gpu" HOST="$HOST" PORT="$port" PYTHONPATH="$PROJECT_ROOT" \
        nohup python src/flowedit/flowedit_v2_service.py > "$log_file" 2>&1 &
    
    echo "✅ GPU $gpu 已启动 (PID: $!, 端口: $port)"
done

echo ""
echo "=========================================="
echo "查看日志:  tail -f $LOG_DIR/flowedit_service_gpu*.log"
echo "健康检查:  curl http://localhost:8001/health"
echo "停止服务:  pkill -f flowedit_v2_service.py"
echo ""

