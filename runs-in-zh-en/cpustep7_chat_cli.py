"""
cpustep7：与训练好的模型对话（对应 runcpu.sh 注释中的 `chat_cli` 段）。

交互式模式：
  python runs-in-zh-en/cpustep7_chat_cli.py

单次问答（非交互）：
  python runs-in-zh-en/cpustep7_chat_cli.py "What is the capital of France?"

加载 data/zh-en/chatsft_checkpoints/ 里最新的 SFT 模型。
进程内运行（runpy）以复用 torch 线程数设置。
"""

import os
import sys
import runpy

from _common import timed, PROJECT_ROOT, guard, has_checkpoints


@timed
def main():
    print("=" * 66)
    print("cpustep7：对话（chat_cli，加载 SFT 模型）")
    print("=" * 66)

    guard(has_checkpoints("chatsft_checkpoints"),
          "data/zh-en/chatsft_checkpoints/ 里还没有 SFT 模型，请先运行 cpustep6_chat_sft.py。")

    args = []
    if len(sys.argv) > 1:
        args += ["-p", sys.argv[1]]

    import torch

    torch.set_num_threads(os.cpu_count() or 4)
    chat_cli_path = os.path.join(PROJECT_ROOT, "scripts", "chat_cli.py")
    print(f"\n[step] $ python scripts/chat_cli.py {' '.join(args)}\n", flush=True)
    sys.argv = ["scripts/chat_cli.py", *args]
    runpy.run_path(chat_cli_path, run_name="__main__")

    print("\ncpustep7 完成！完整版请把各步 NANOCHAT_*_ITERS 调高（参考 README）。")


if __name__ == "__main__":
    sys.exit(main())