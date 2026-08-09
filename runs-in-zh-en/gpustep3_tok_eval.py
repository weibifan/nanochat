"""
gpustep3：tokenizer 压缩率评估（对应 speedrun.sh 的 tok_eval 段）。

在固定文本样本上验证 bytes/token。tok_eval.py 无 CLI 参数，本地读数据不出网。

用法：
  python runs-in-zh-en/gpustep3_tok_eval.py
"""

import sys

import _gpu_common as g
from _common import timed, run_module, guard


@timed
def main():
    g.print_profile()
    print("=" * 66)
    print("gpustep3：tokenizer 评估（tok_eval）")
    print("=" * 66)

    guard(g.has_tokenizer(), "data/gpu/tokenizer/tokenizer.pkl 不存在，请先运行 gpustep2_tok_train.py。")

    run_module(["scripts.tok_eval"])
    print("\ngpustep3 完成！下一步：python runs-in-zh-en/gpustep4_base_train.py")


if __name__ == "__main__":
    sys.exit(main())
