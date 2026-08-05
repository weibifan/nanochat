#!/bin/bash
# 步骤3：d4 玩具预训练（500 步）
# 用法: bash 03_pretrain.sh
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/nanochat

if [ -f "/root/.cache/nanochat/base_checkpoints/d4/model_000500.pt" ]; then
    echo "base checkpoint 已存在，跳过"
else
    echo "=== d4 预训练 (500 步) ==="
    python -m scripts.base_train \
        --depth=4 --device-batch-size=8 \
        --num-iterations=500 --run="dummy" \
        --save-every=500 --model-tag="d4"
fi

echo "=== 检查产物 ==="
ls -la /root/.cache/nanochat/base_checkpoints/d4/
echo "✅ 步骤3 完成"