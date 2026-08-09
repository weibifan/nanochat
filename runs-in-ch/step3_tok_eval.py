"""
step3：评估分词器压缩率（中文 vs GPT-2 / GPT-4 词表）。

读取 step2 训练好的 tokenizer，对比 gpt2 / cl100k_base 的压缩率。
重点看：
- "climbmix-train" / "climbmix-val" 是中文语料，Ours 应该明显优于 GPT-2；
- korean / news 等非中英文本可作参考。

对应原版 runcpu.sh：
  python -m scripts.tok_eval
"""

import sys

from _common import run_module, guard, has_tokenizer


def main():
    print("=" * 70)
    print("step3：评估分词器压缩率")
    print("=" * 70)

    guard(has_tokenizer(), "data/zh-ch/tokenizer/tokenizer.pkl 不存在，请先运行 step2_tok_train.py。")

    run_module(["scripts.tok_eval"])

    print("\nstep3 完成！下一步：python runs-in-ch/step4_base_train.py")


if __name__ == "__main__":
    sys.exit(main())
