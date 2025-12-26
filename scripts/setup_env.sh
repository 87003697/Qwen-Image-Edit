#!/usr/bin/env bash
# Qwen-Image-Edit 环境安装脚本
# 使用方法: conda activate qwen-image-edit && bash scripts/setup_env.sh

set -euo pipefail

echo "=== Qwen-Image-Edit 环境安装 ==="

# PyTorch (CUDA 12.4)
echo "[1/4] 安装 PyTorch..."
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124 -q

# 核心依赖
echo "[2/4] 安装核心依赖..."
pip install transformers>=4.51.3 -q
pip install git+https://github.com/huggingface/diffusers -q

# FlowEdit API 服务依赖
echo "[3/4] 安装 API 服务依赖..."
pip install fastapi uvicorn requests -q

# flash-attn (可选，加速注意力计算)
echo "[4/4] 安装 flash-attn..."
pip install ninja -q
pip install flash-attn --no-build-isolation -q

# 验证
echo ""
echo "=== 验证安装 ==="
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import diffusers; print('diffusers:', diffusers.__version__)"
python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)" 2>/dev/null || echo "flash-attn: 未安装"

echo ""
echo "✅ 安装完成！"
