"""
gpustep6：SFT 微调（GPU，对应 speedrun.sh 的 chat_sft 段，_zh 版启用 ChatCORE）。

与 cpustep6 同思路：用 _common.install_zh_modules() 把 sys.modules 重定向到
_zh 版本后，进程内运行 scripts/chat_sft_zh.py：
  - ChatCORE 数据来自本地 <repo>/eval_data/（install_chatcore_datasets 装进
    data/gpu/task_data/ 缓存）；若克隆的仓库没带这些数据，自动关闭 ChatCORE
    （--chatcore-every=-1）只做 SFT。
  - SFT 任务数据（SmolTalk / MMLU / GSM8K）由 gpustep1 预下载，缺失时自动补。

迭代数由配置决定（NANOCHAT_GPU_SFT_ITERS 覆盖）：
  fast: 1000（~2-4 min）   full: -1=完整 epoch（~1-2 h，总 batch 继承 base）

用法：
  python runs-in-zh-en/gpustep6_chat_sft.py
  NANOCHAT_GPU_SFT_ITERS=200 python runs-in-zh-en/gpustep6_chat_sft.py
"""

import os
import sys
import runpy

import _gpu_common as g
import _common
from _common import timed, guard, PROJECT_ROOT


def _ensure_task_data():
    if g.has_sft_data():
        print("  data/gpu/task_data/ 已存在，跳过预下载。")
        return
    print("  data/gpu/task_data/ 缺失，先下载 SFT 数据（ModelScope 镜像）……")
    _common.run_module(["tasks.download_sft_data_zh"], check=False)
    guard(g.has_sft_data(), "SFT 任务数据仍未就绪（data/gpu/task_data/ 缺失）。请检查网络后重试。")


@timed
def main():
    g.print_profile()
    print("=" * 66)
    print("gpustep6：chat_sft_zh（单卡 GPU）")
    print("=" * 66)

    guard(g.has_checkpoints("base_checkpoints"),
          "data/gpu/base_checkpoints/ 里还没有模型，请先运行 gpustep4_base_train.py。")
    _ensure_task_data()

    sft_iters = int(g.cfg("sft_iters"))
    chatcore_every = int(g.cfg("chatcore_every"))
    if not g.has_chatcore_source():
        print("[sft] 仓库 eval_data/ 未带 ChatCORE 数据，本次关闭 ChatCORE 评测。")
        chatcore_every = -1

    args = [
        "--eval-every={}".format(g.cfg("sft_eval_every")),
        "--eval-tokens={}".format(g.cfg("sft_eval_tokens")),
        "--num-iterations={}".format(sft_iters),
        "--chatcore-every={}".format(chatcore_every),
        "--chatcore-max-cat={}".format(g.cfg("chatcore_max_cat")),
        "--run=dummy",  # 等同 speedrun.sh 的 WANDB_RUN=dummy
    ]

    _common.install_zh_modules()
    import torch

    torch.set_num_threads(os.cpu_count() or 4)

    chat_sft_path = os.path.join(PROJECT_ROOT, "scripts", "chat_sft_zh.py")
    print(f"\n[step] $ python scripts/chat_sft_zh.py {' '.join(args)}\n", flush=True)
    sys.argv = ["scripts/chat_sft_zh.py", *args]
    runpy.run_path(chat_sft_path, run_name="__main__")

    guard(g.has_checkpoints("chatsft_checkpoints"), "SFT 结束但 data/gpu/chatsft_checkpoints/ 仍为空。")
    print(f"  SFT 检查点：{g.data_dir('chatsft_checkpoints')}")
    print("\ngpustep6 完成！下一步：python runs-in-zh-en/gpustep7_chat_eval.py")


if __name__ == "__main__":
    sys.exit(main())
