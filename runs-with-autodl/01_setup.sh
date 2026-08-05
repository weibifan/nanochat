#!/bin/bash
# 步骤1：环境配置（在 AutoDL 服务器上执行）
# 用法: bash 01_setup.sh
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== [1/3] 克隆 nanochat 代码 ==="
cd /root/autodl-tmp
if [ ! -d "nanochat" ]; then
    git clone https://github.com/karpathy/nanochat.git
fi
cd nanochat

echo "=== [2/3] 放宽 torch 版本约束 ==="
# AutoDL 预装 torch 2.8，pyproject.toml 锁定 2.9.1 会触发降级
sed -i 's/torch==2\.9\.1/torch>=2.8.0,<2.10.0/g' pyproject.toml
grep -n 'torch' pyproject.toml | head -3

echo "=== [3/3] 用 pip 安装依赖（不用 uv，AutoDL 下会卡死）==="
pip install huggingface_hub tiktoken numpy rustbpe filelock kernels psutil pyarrow wandb -q

echo ""
echo "=== 验证 ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo "✅ 步骤1 完成"