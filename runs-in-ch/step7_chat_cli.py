"""
step7：与训练好的中文模型对话（chat_cli）。

交互式模式：
  python runs-in-ch/step7_chat_cli.py

单次问答（非交互）：
  python runs-in-ch/step7_chat_cli.py "中国的首都是哪个城市？"

加载 data/zh-ch/chatsft_checkpoints/ 里最新的 SFT 模型。
进程内运行（run_module_inproc）以利用 16 线程加速生成。
"""

import sys

from _common import run_module_inproc, guard, has_checkpoints


def main():
    print("=" * 70)
    print("step7：中文对话（chat_cli）")
    print("=" * 70)

    guard(has_checkpoints("chatsft_checkpoints"), "data/zh-ch/chatsft_checkpoints/ 里还没有 SFT 模型，请先运行 step6_chat_sft.py。")

    args = []
    if len(sys.argv) > 1:
        args += ["-p", sys.argv[1]]
    run_module_inproc("scripts/chat_cli.py", args, note="torch 线程数 = 逻辑核数（16）")

    print("\nstep7 完成！")


if __name__ == "__main__":
    sys.exit(main())
