"""
gpustep7：SFT 后的 ChatCORE 评估（对应 speedrun.sh 的 chat_eval 段，本地数据）。

用 scripts/chat_eval_zh.py 跑 ARC-Easy / ARC-Challenge / MMLU / GSM8K / HumanEval，
数据来自本地 <repo>/eval_data/（install_chatcore_datasets 装进 task_data 缓存），
不再走海外。最后输出 ChatCORE 指标。

   python -m scripts.chat_eval_zh -i sft -b <batch> [-x <max_problems>]

若仓库未带 eval_data，会直接给出提示并跳过（避免 load_hub_dataset 缺数据报错）。

用法：
  python runs-in-zh-en/gpustep7_chat_eval.py
  NANOCHAT_GPU_CHAT_EVAL_BATCH=32 python runs-in-zh-en/gpustep7_chat_eval.py
"""

import sys

import _gpu_common as g
from _common import timed, run_module, guard


@timed
def main():
    g.print_profile()
    print("=" * 66)
    print("gpustep7：chat_eval_zh（SFT 模型 ChatCORE）")
    print("=" * 66)

    guard(g.has_checkpoints("chatsft_checkpoints"),
          "data/gpu/chatsft_checkpoints/ 里还没有 SFT 模型，请先运行 gpustep6_chat_sft.py。")
    guard(g.has_chatcore_source(),
          "仓库 eval_data/ 未带 ChatCORE 数据（ARC/MMLU/GSM8K/HumanEval），本步无法评估。"
          "请把数据放回 eval_data/ 后重试，或跳过本步。")

    args = [
        "scripts.chat_eval_zh",
        "-i", "sft",
        "-b", str(g.cfg("chat_eval_batch")),
    ]
    max_problems = int(g.cfg("chat_eval_max_problems"))
    if max_problems > 0:
        args += ["-x", str(max_problems)]

    run_module(args)
    print("\ngpustep7 完成！下一步：python runs-in-zh-en/gpustep8_chat_cli.py")


if __name__ == "__main__":
    sys.exit(main())
