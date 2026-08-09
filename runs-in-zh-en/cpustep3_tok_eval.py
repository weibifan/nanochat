"""
cpustep3：tokenizer 压缩率评估（对应 runcpu.sh 的 `python -m scripts.tok_eval`）。

在固定文本样本上验证 bytes/token，比较 BPE 词表的压缩能力。
tok_eval.py 无 CLI 参数，直接运行即可，本地读数据不出网。

用法：
  python runs-in-zh-en/cpustep3_tok_eval.py
"""

import sys

from _common import timed, run_module, guard, has_tokenizer


@timed
def main():
    print("=" * 66)
    print("cpustep3：tokenizer 评估（tok_eval）")
    print("=" * 66)

    guard(has_tokenizer(), "data/zh-en/tokenizer/tokenizer.pkl 不存在，请先运行 cpustep2_tok_train.py。")

    run_module(["scripts.tok_eval"])
    print("\ncpustep3 完成！下一步：python runs-in-zh-en/cpustep4_base_train.py")


if __name__ == "__main__":
    sys.exit(main())