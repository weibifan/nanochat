#!/usr/bin/env bash
# linux_env_check.sh —— 检查一台 Linux 机器的系统/CPU/内存/磁盘/GPU/Python 环境。
# 纯 shell 实现，不依赖 Python：即使 Python 没装、GPU 没驱动，也能照常输出系统信息。
#
# 用法：bash linux_env_check.sh          （无 bash 权限时用 sh）
#       ./linux_env_check.sh            （需先 chmod +x）

echo "=== 系统 ==="
if [ -f /etc/os-release ]; then
  . /etc/os-release
  echo "发行版: $PRETTY_NAME"
else
  echo "发行版: (无法读取 /etc/os-release)"
fi
echo "内核: $(uname -r)"
echo "架构: $(uname -m)"
echo "主机名: $(hostname)"

echo ""
echo "=== CPU ==="
echo "逻辑核心数: $(nproc)"
if command -v lscpu >/dev/null 2>&1; then
  lscpu | grep -E "Model name|^CPU\(s\)|Architecture|CPU MHz" || true
fi

echo ""
echo "=== 内存 ==="
if command -v free >/dev/null 2>&1; then
  free -h | head -3
else
  grep -E "MemTotal|MemFree|MemAvailable" /proc/meminfo || echo "(无法读取 /proc/meminfo)"
fi

echo ""
echo "=== 磁盘 ==="
df -h / 2>/dev/null
for d in /root/autodl-tmp /data /mnt/data /workspace /home; do
  [ -d "$d" ] && df -h "$d" 2>/dev/null
done

echo ""
echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
else
  echo "nvidia-smi 不存在（无 NVIDIA GPU 或驱动未装）"
  if command -v lspci >/dev/null 2>&1; then
    lspci | grep -i -E "vga|3d controller|display" || echo "(未发现显卡)"
  fi
fi

echo ""
echo "=== CUDA 编译器 ==="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | grep release
else
  echo "nvcc 未安装（正常：PyTorch 自带 CUDA 运行时，训练不依赖 nvcc）"
fi

echo ""
echo "=== C/C++ 工具链（torch.compile 需要）==="
for c in cc gcc g++ clang make ninja; do
  if command -v "$c" >/dev/null 2>&1; then
    v="$("$c" --version 2>/dev/null | head -1)"
    echo "$c: $v"
  else
    echo "$c: 未安装"
  fi
done

echo ""
echo "=== Python ==="
PY=""
if command -v python3 >/dev/null 2>&1; then
  echo "python3: $(python3 --version)  @ $(command -v python3)"
  PY="$(command -v python3)"
fi
if command -v python >/dev/null 2>&1; then
  echo "python:  $(python --version)  @ $(command -v python)"
  [ -z "$PY" ] && PY="$(command -v python)"
fi
if [ -z "$PY" ]; then
  echo "未安装 Python（先装：Ubuntu/Debian 用 apt install python3，或用 conda/micromamba）"
fi
if command -v conda >/dev/null 2>&1; then
  echo "conda: $(conda --version)  @ $(command -v conda)"
fi

echo ""
echo "=== PyTorch / 关键依赖（需要 Python）==="
if [ -n "$PY" ]; then
  "$PY" -c "import torch; print('torch:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('GPU 名:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1 | head -3
  "$PY" -c "import numpy; print('numpy:', numpy.__version__)" 2>&1 | head -1
  "$PY" -c "import rustbpe; print('rustbpe: OK')" 2>&1 | head -1
  "$PY" -c "import filelock; print('filelock: OK')" 2>&1 | head -1
  "$PY" -c "import tiktoken; print('tiktoken:', tiktoken.__version__)" 2>&1 | head -1
  "$PY" -c "import pyarrow; print('pyarrow:', pyarrow.__version__)" 2>&1 | head -1
else
  echo "未找到 Python，跳过依赖检测。"
fi
