"""
gpustep2：训练 BPE tokenizer（对应 speedrun.sh 的 tok_train 段）。

用 ClimbMix 语料训练 BPE tokenizer，产物：
  data/gpu/tokenizer/tokenizer.pkl + token_bytes.pt

词表 / 语料规模由配置决定：
  fast: NANOCHAT_GPU_VOCAB_SIZE=8192  NANOCHAT_GPU_MAX_CHARS=120000000（~1-2 min）
  full: NANOCHAT_GPU_VOCAB_SIZE=32768 NANOCHAT_GPU_MAX_CHARS=2000000000（~30-40 min）

rustbpe 训练是纯 CPU 计算（不涉及 CUDA），AutoDL 机器 CPU 核数越多越快。

用法：
  python runs-in-zh-en/gpustep2_tok_train.py
  NANOCHAT_GPU_VOCAB_SIZE=16384 python runs-in-zh-en/gpustep2_tok_train.py
"""

import os
import sys

import _gpu_common as g
from _common import timed, guard, run_in_process


@timed
def main():
    g.print_profile()
    vocab = g.cfg("vocab_size")
    max_chars = g.cfg("max_chars")
    print("=" * 66)
    print(f"gpustep2：训练 tokenizer（vocab={vocab}，max_chars={max_chars}）")
    print("=" * 66)

    guard(g.count_shards() >= 1, "data/gpu/base_data_climbmix/ 里没有分片，请先运行 gpustep1_prepare_data.py。")

    run_in_process(
        "scripts/tok_train.py",
        ["--vocab-size", str(vocab), "--max-chars", str(max_chars)],
        note="tok_train.py（进程内，CPU 线程数设满）",
    )
    guard(g.has_tokenizer(), "tokenizer 未生成: data/gpu/tokenizer/")

    print("\n  tokenizer 产物：")
    for f in sorted(os.listdir(g.data_dir("tokenizer"))):
        print(f"    {f}  ({os.path.getsize(os.path.join(g.data_dir('tokenizer'), f)):,} bytes)")

    print("\ngpustep2 完成！下一步：python runs-in-zh-en/gpustep3_tok_eval.py")


if __name__ == "__main__":
    sys.exit(main())
