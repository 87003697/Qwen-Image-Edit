#!/usr/bin/env bash
# Qwen-Image-Edit 环境安装脚本
# 使用方法: bash scripts/setup_env.sh

set -euo pipefail

echo "=== Qwen-Image-Edit 环境安装 ==="

# 创建并激活 conda 环境
ENV_NAME="qwen-image-edit"
PYTHON_VERSION="3.10"

echo "[0/4] 创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)..."

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh

# 检查环境是否已存在
if conda env list | grep -q "^$ENV_NAME "; then
    echo "环境 $ENV_NAME 已存在，跳过创建"
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# 激活环境
conda activate "$ENV_NAME"
echo "已激活环境: $CONDA_DEFAULT_ENV"
echo "Python 路径: $(which python)"
echo "Pip 路径: $(which pip)"

# PyTorch (CUDA 12.4)
echo "[1/5] 安装 PyTorch..."
python -m pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124 -q

# 核心依赖
echo "[2/5] 安装核心依赖..."
python -m pip install "transformers>=4.51.3" -q
python -m pip install git+https://github.com/huggingface/diffusers -q
python -m pip install accelerate -q

# 相似度/梯度计算依赖
echo "[3/5] 安装相似度梯度依赖 (SSIM/LPIPS)..."
python -m pip install lpips pytorch-msssim -q

# FlowEdit API 服务依赖
echo "[4/5] 安装 API 服务依赖..."
python -m pip install fastapi uvicorn requests -q

# flash-attn (可选，加速注意力计算)
echo "[5/5] 安装 flash-attn..."
python -m pip install ninja -q
python -m pip install flash-attn --no-build-isolation -q

# 验证
echo ""
echo "=== 验证安装 ==="
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import diffusers; print('diffusers:', diffusers.__version__)"
python -c "import accelerate; print('accelerate:', accelerate.__version__)"
python -c "import lpips; print('lpips:', lpips.__version__)"
python -c "from pytorch_msssim import __version__ as msssim_version; print('pytorch-msssim:', msssim_version)"
python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)" 2>/dev/null || echo "flash-attn: 未安装"

echo ""
echo "✅ 安装完成！"
