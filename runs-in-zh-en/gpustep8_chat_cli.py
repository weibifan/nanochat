"""
gpustep8：与训练好的模型对话（对应 speedrun.sh 注释中的 chat_cli 段）。

交互式模式：
  python runs-in-zh-en/gpustep8_chat_cli.py
单次问答（非交互）：
  python runs-in-zh-en/gpustep8_chat_cli.py "What is the capital of France?"

加载 data/gpu/chatsft_checkpoints/ 里最新的 SFT 模型（GPU 推理）。
"""

import os
import sys
import runpy

import _gpu_common as g
from _common import timed, PROJECT_ROOT, guard


@timed
def main():
    g.print_profile()
    print("=" * 66)
    print("gpustep8：对话（chat_cli，加载 SFT 模型，GPU 推理）")
    print("=" * 66)

    guard(g.has_checkpoints("chatsft_checkpoints"),
          "data/gpu/chatsft_checkpoints/ 里还没有 SFT 模型，请先运行 gpustep6_chat_sft.py。")

    args = []
    if len(sys.argv) > 1:
        args += ["-p", sys.argv[1]]

    import torch

    torch.set_num_threads(os.cpu_count() or 4)
    chat_cli_path = os.path.join(PROJECT_ROOT, "scripts", "chat_cli.py")
    print(f"\n[step] $ python scripts/chat_cli.py {' '.join(args)}\n", flush=True)
    sys.argv = ["scripts/chat_cli.py", *args]
    runpy.run_path(chat_cli_path, run_name="__main__")

    print("\ngpustep8 完成！全流程结束。")


if __name__ == "__main__":
    sys.exit(main())
