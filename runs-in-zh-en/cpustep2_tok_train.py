"""
cpustep2：训练 BPE tokenizer（对应 runcpu.sh 的 `python -m scripts.tok_train --max-chars=2000000000`）。

用 ClimbMix 语料训练 BPE tokenizer，产物：
  data/zh-en/tokenizer/tokenizer.pkl + token_bytes.pt

练习版默认（合计 ~5-7 min，配合其它步骤总预算 ≤30 min）：
  NANOCHAT_MAX_CHARS=120000000  （120M 字符，足够训小词表）
  NANOCHAT_VOCAB_SIZE=8192       （小词表 → BPE merge 阶段线性变快）

完整版（runcpu.sh 官方值，~25-30 min 单步）：
  NANOCHAT_MAX_CHARS=2000000000  NANOCHAT_VOCAB_SIZE=32768

进程内运行（run_in_process）以使用 CPU 线程数（rustbpe 训练同样走线程）。

用法：
  python runs-in-zh-en/cpustep2_tok_train.py
  NANOCHAT_MAX_CHARS=2000000000 NANOCHAT_VOCAB_SIZE=32768 python runs-in-zh-en/cpustep2_tok_train.py   # 完整版
"""

import os
import sys

from _common import timed, data_dir, has_tokenizer, count_shards, guard, run_in_process

VOCAB_SIZE = os.environ.get("NANOCHAT_VOCAB_SIZE", "8192")
MAX_CHARS = os.environ.get("NANOCHAT_MAX_CHARS", "120000000")

# 预计耗时（16 核 CPU，实测）：merge 阶段 ≈ vocab_size×0.4ms，chars 1M≈0.08min
_EST = {
    "v8192/ch120M": "~0.1 min",
    "v32768/ch2B": "~35 min",
}


@timed
def main():
    print("=" * 66)
    print(f"cpustep2：训练 tokenizer（vocab={VOCAB_SIZE}，max_chars={MAX_CHARS}）")
    print("=" * 66)

    guard(count_shards() >= 1, "data/zh-en/base_data_climbmix/ 里没有分片，请先运行 cpustep1_prepare_data.py。")

    run_in_process(
        "scripts/tok_train.py",
        ["--vocab-size", VOCAB_SIZE, "--max-chars", MAX_CHARS],
        note="tok_train.py（进程内，CPU 线程数设满）",
    )
    guard(has_tokenizer(), "tokenizer 未生成: data/zh-en/tokenizer/")

    print("\n  tokenizer 产物：")
    for f in sorted(os.listdir(data_dir("tokenizer"))):
        print(f"    {f}  ({os.path.getsize(os.path.join(data_dir('tokenizer'), f)):,} bytes)")

    print("\ncpustep2 完成！下一步：python runs-in-zh-en/cpustep3_tok_eval.py")


if __name__ == "__main__":
    sys.exit(main())