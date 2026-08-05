#!/bin/bash
# 步骤5：推理对话 + 评估（MMLU / GSM8K）
# 用法: bash 05_eval.sh
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/nanochat

echo "=== 交互式对话（仅支持英文，Ctrl-C 或输入退出）==="
echo "问: What is the capital of France?"
python -m scripts.chat_cli -g d4 -p "What is the capital of France?"

echo ""
echo "=== 评估 MMLU / GSM8K（各 50 题样本）==="
python -m scripts.chat_eval -i sft -g d4 -a "MMLU|GSM8K" -x 50

echo "✅ 步骤5 完成"