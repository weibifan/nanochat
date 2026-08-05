#!/bin/bash
# 步骤2：下载 ClimbMix-400B 预训练数据 + 训练 Tokenizer（词表 32768）
# 用法: bash 02_data_tokenizer.sh
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/nanochat

# ClimbMix 预训练数据目录（tokenizer 与预训练共用）
CLIMBMIX_DIR="$HOME/.cache/nanochat/base_data_climbmix"
mkdir -p "$CLIMBMIX_DIR"

echo "=== [1/2] 确保 ClimbMix-400B 分片已就绪（缺则从 hf-mirror 下载，至少 8 个 shard）==="
shard_count=$(ls "$CLIMBMIX_DIR"/*.parquet 2>/dev/null | wc -l)
echo "当前已有 $shard_count 个分片"
if [ "$shard_count" -lt 8 ]; then
    echo "分片不足，从 hf-mirror 下载 shard_00000~00007 ..."
    for i in 00000 00001 00002 00003 00004 00005 00006 00007; do
        f="shard_${i}.parquet"
        if [ ! -f "$CLIMBMIX_DIR/$f" ]; then
            wget -q --show-progress -O "$CLIMBMIX_DIR/$f" \
                "https://hf-mirror.com/datasets/karpathy/climbmix-400b-shuffle/resolve/main/${f}"
            echo "  已下载 $f"
        fi
    done
fi
echo "ClimbMix 分片：$(ls "$CLIMBMIX_DIR"/*.parquet 2>/dev/null | wc -l) 个，$(du -sh "$CLIMBMIX_DIR" 2>/dev/null | cut -f1)"

echo ""
echo "=== [2/2] 用 ClimbMix-400B 数据训练 Tokenizer（词表 32768）==="
# 直接用项目官方 tok_train.py：其内部经 nanochat.dataset 的 parquets_iter_batched("train")
# 读取 base_data_climbmix/ 下的所有分片，训练 BPE tokenizer 并保存到 ~/.cache/nanochat/tokenizer/
python -m scripts.tok_train

echo ""
echo "=== 验证产物 ==="
ls -la "$HOME/.cache/nanochat/tokenizer/"
python -c "from nanochat.tokenizer import RustBPETokenizer; import os; t=RustBPETokenizer.from_directory(os.path.expanduser('~/.cache/nanochat/tokenizer')); print('vocab =', t.get_vocab_size())"
echo "✅ 步骤2 完成"