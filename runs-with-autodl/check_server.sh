#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== 系统 ==="
cat /etc/os-release | head -2
uname -r

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""
echo "=== CPU / 内存 ==="
nproc
free -h

echo ""
echo "=== CUDA 版本 ==="
nvcc --version | tail -2

echo ""
echo "=== Python ==="
python --version
which python

echo ""
echo "=== PyTorch (从 Python 里检测) ==="
python -c "import torch; print('torch:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('GPU 名:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "=== 磁盘 ==="
df -h /root/autodl-tmp /root/.cache

echo ""
echo "=== 关键 Python 包（本轮实验后续要用，此时可能尚未安装）==="
python -c "import huggingface_hub; print('huggingface_hub:', huggingface_hub.__version__)" 2>&1 | head -1
python -c "import tiktoken; print('tiktoken:', tiktoken.__version__)" 2>&1 | head -1
python -c "import numpy; print('numpy:', numpy.__version__)" 2>&1 | head -1
python -c "import rustbpe; print('rustbpe: OK')" 2>&1 | head -1
python -c "import filelock; print('filelock: OK')" 2>&1 | head -1
python -c "import pyarrow; print('pyarrow:', pyarrow.__version__)" 2>&1 | head -1
