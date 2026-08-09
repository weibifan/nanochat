"""
step2：训练 BPE 分词器（中文）。

用 step1 重打包好的中文语料训练 BPE，并把 tokenizer + token_bytes 保存到
data/zh-ch/tokenizer/。之后所有训练/评测脚本都会通过 get_tokenizer() 读取。

对应原版 runcpu.sh：
  python -m scripts.tok_train --max-chars=2000000000

可通过环境变量调整：
  NANOCHAT_MAX_CHARS   参与 BPE 训练的字符数（本机默认 300M，见下）
  NANOCHAT_VOCAB_SIZE  词表大小（默认 32768 = nanochat 原版默认）

为什么默认值改小了（实测结论，详见 step2 之后的分析）：
- rustbpe 会把语料切出的 unique 序列全部驻留内存算 pair 计数：
    2B 字符(~1.2 亿 unique) 在本机(32GB)直接 OOM；
    1B 字符(~6400 万 unique) pair-count 就要 ~49 分钟，且 merge 阶段极慢；
    100M 字符 ~6 分钟。
- 词表大小只影响 merge 阶段（32768 的 merge 数是 65536 的一半）；
  pair-count 阶段与词表无关、只随字符数线性增长。
- 因此这里默认 300M 字符 + 32768 词表（预计 ~15 分钟）。BPE 学词表
  几亿字符已足够；大内存机器想保真可调回 1B~2B 字符 / 65536 词表。
"""

import os
import sys

from _common import data_dir, run_module, guard, has_shards


def main():
    max_chars = os.environ.get("NANOCHAT_MAX_CHARS", "300000000")
    vocab_size = os.environ.get("NANOCHAT_VOCAB_SIZE", "32768")

    print("=" * 70)
    print("step2：训练中文 BPE 分词器")
    print(f"  max_chars = {int(max_chars):,}  |  vocab_size = {vocab_size}")
    print("=" * 70)

    guard(has_shards(), "data/zh-ch/base_data_climbmix 里还没有中文预训练 shard，请先运行 step1_prepare_data.py。")

    run_module([
        "scripts.tok_train",
        "--max-chars", max_chars,
        "--vocab-size", vocab_size,
    ])

    tokenizer_dir = data_dir("tokenizer")
    print(f"\nstep2 完成！tokenizer 已保存到 {tokenizer_dir}")
    print("下一步：python runs-in-ch/step3_tok_eval.py")


if __name__ == "__main__":
    sys.exit(main())
