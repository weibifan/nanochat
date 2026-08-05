#!/bin/bash
# 步骤4：SFT 微调（完整 1 epoch，约 1868 步 / 25 分钟）
# 用法: bash 04_sft.sh
# 前置：本脚本会把修改版 tasks/common.py 复制到 nanochat 仓库（新 clone 的原版走 HF API 会 403）
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/nanochat

echo "=== 覆盖修改版 tasks/common.py（读本地缓存，不再走 HF API）==="
if [ -f /root/tasks_common.py ]; then
    cp /root/tasks_common.py tasks/common.py
    echo "已覆盖 tasks/common.py"
fi

if [ -f "/root/.cache/nanochat/chatsft_checkpoints/d4/model_001868.pt" ]; then
    echo "SFT checkpoint 已存在，跳过"
else
    export HF_ENDPOINT=https://hf-mirror.com

    echo "=== 设置 HF_ENDPOINT + 确认加载本地缓存数据 ==="
    echo "数据缓存: /root/.cache/nanochat/task_data/ (若存在则跳过下载)"

    echo "=== SFT 微调 (d4) ==="
    python -m scripts.chat_sft \
        --run="dummy" \
        --device-batch-size=4 \
        --num-iterations=-1 \
        --model-tag="d4" \
        --chatcore-every=-1
fi

echo "=== 检查产物 ==="
ls -la /root/.cache/nanochat/chatsft_checkpoints/d4/
echo "✅ 步骤4 完成"