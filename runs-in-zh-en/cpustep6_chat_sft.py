"""
cpustep6：SFT 微调（对应 runcpu.sh 的 `chat_sft` 段，_zh 版启用 ChatCORE）。

与 runs-in-ch/step6 思路一致：不修改 scripts/chat_sft.py，而是用
`_common.install_zh_modules()` 把 sys.modules 里
  tasks.common     -> tasks/common_zh.py   （读本地 task_data 缓存，不联网）
  nanochat.dataset -> nanochat/dataset_zh.py（ModelScope 底座）
并运行 scripts/chat_sft_zh.py（_zh 拷贝）：
  - ChatCORE 数据来自本地 <repo>/eval_data/（install_chatcore_datasets 装进
    task_data 缓存），--chatcore-every 由 NANOCHAT_CHATCORE_EVERY 控制
    （练习版默认 25，完整版默认 200；设 -1 可关闭）。
  - ChatCORE 分类任务样本数由 NANOCHAT_CHATCORE_MAX_CAT 控制
    （练习版默认 200，完整版默认 -1=全部），生成任务默认 24。

SFT 任务数据（SmolTalk / MMLU / GSM8K）由 cpustep1 预下载到 data/zh-en/task_data/，
这里缺失时会自动再跑一次（幂等，已有则跳过）。

超参对应 runcpu.sh：--eval-every=200 --eval-tokens=524288 --num-iterations=1500。
练习版默认 NANOCHAT_SFT_ITERS=25（配合其它步骤总预算 ≤30 min；完整版设 1500）。

实测耗时（本机 16 核 CPU，按 runs-in-ch 单步 ~11-26 s 外推）：
  练习版 25 迭代 ≈ 5-10 min；完整版 1500 迭代 ≈ 4-5 h。

用法：
  python runs-in-zh-en/cpustep6_chat_sft.py
  NANOCHAT_SFT_ITERS=1500 python runs-in-zh-en/cpustep6_chat_sft.py   # 完整版（runcpu.sh 数值）
  NANOCHAT_CHATCORE_EVERY=-1 python runs-in-zh-en/cpustep6_chat_sft.py  # 关闭 ChatCORE
"""

import os
import sys
import runpy

import _common
from _common import timed, guard, has_checkpoints, has_sft_data, data_dir, PROJECT_ROOT


def _ensure_task_data():
    if has_sft_data():
        print("  data/zh-en/task_data/ 已存在，跳过预下载。")
        return
    print("  data/zh-en/task_data/ 缺失，先下载 SFT 数据（ModelScope 镜像）……")
    _common.run_module(["tasks.download_sft_data_zh"], check=False)
    guard(has_sft_data(), "SFT 任务数据仍未就绪（data/zh-en/task_data/ 缺失）。请检查网络后重试。")


@timed
def main():
    iters = int(os.environ.get("NANOCHAT_SFT_ITERS", "25"))
    full = iters >= 1500
    chatcore_every = os.environ.get("NANOCHAT_CHATCORE_EVERY", "200" if full else "25")
    chatcore_max_cat = os.environ.get("NANOCHAT_CHATCORE_MAX_CAT", "-1" if full else "200")
    print("=" * 66)
    print(f"cpustep6：chat_sft_zh（练习版 {iters} 迭代，ChatCORE every {chatcore_every}）")
    print("=" * 66)

    guard(has_checkpoints("base_checkpoints"),
          "data/zh-en/base_checkpoints/ 里还没有模型，请先运行 cpustep4_base_train.py。")
    _ensure_task_data()

    args = [
        "--eval-every=200" if full else "--eval-every=25",
        "--eval-tokens=524288" if full else "--eval-tokens=32768",
        "--num-iterations={}".format(iters),
        "--chatcore-every={}".format(chatcore_every),
        "--chatcore-max-cat={}".format(chatcore_max_cat),
        "--run=dummy",  # 等同 runcpu.sh 的 WANDB_RUN=dummy
    ]

    _common.install_zh_modules()
    import torch

    torch.set_num_threads(os.cpu_count() or 4)

    chat_sft_path = os.path.join(PROJECT_ROOT, "scripts", "chat_sft_zh.py")
    print(f"\n[step] $ python scripts/chat_sft_zh.py {' '.join(args)}\n", flush=True)
    sys.argv = ["scripts/chat_sft_zh.py", *args]
    runpy.run_path(chat_sft_path, run_name="__main__")

    guard(has_checkpoints("chatsft_checkpoints"), "SFT 结束但 data/zh-en/chatsft_checkpoints/ 仍为空。")
    print(f"  SFT 检查点：{data_dir('chatsft_checkpoints')}")
    print("\ncpustep6 完成！下一步：python runs-in-zh-en/cpustep7_chat_cli.py")


if __name__ == "__main__":
    sys.exit(main())