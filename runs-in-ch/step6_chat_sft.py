"""
step6：中文 SFT（chat_sft）。

不修改 scripts/chat_sft.py。它在文件顶部硬编码了四个任务模块 import：
  from tasks.common import TaskMixture
  from tasks.gsm8k import GSM8K
  from tasks.mmlu import MMLU
  from tasks.smoltalk import SmolTalk

本脚本在进程内运行 chat_sft.py 之前，先把这四个模块在 sys.modules 里
重定向到中文版 *_ch 模块：
  tasks.common   -> tasks.common_ch   （TaskMixture 增加 eval_type）
  tasks.smoltalk -> tasks.smoltalk_ch （SmolTalkCh：smoltalk-chinese）
  tasks.mmlu     -> tasks.mmlu_ch     （MMLUCh：C-Eval dev+val）
  tasks.gsm8k    -> tasks.gsm8k_ch    （GSM8KCh：GSM8K_zh）

这样 chat_sft.py 内部 "from tasks.xxx import ..." 拿到的就是中文版类，
任务数据全部走本地 data/zh-ch/task_data/，不改任何原文件。

超参对应 runcpu.sh：
  --eval-every=200 --eval-tokens=524288 --num-iterations=1500 --run=dummy
额外加 --chatcore-every=-1：跳过 ChatCORE 评测（它会跑英文 ARC/HumanEval，
且 C-Eval/GSM8K_zh 的格式会把 ChatCORE 的 json 解析打爆）。

“练习版”配置：num_iterations=50（约 10 分钟跑完），完整版把 argv 里的 50
改回 1500（这台机器约需 4~5 小时）。

注意：进程内运行时先把 torch 线程设满（16），torch.compile 已在 _common 里
退化为 eager（本机没有 MSVC cl）。

检查点保存到 data/zh-ch/chatsft_checkpoints/。
"""

import os
import sys
import runpy

from _common import PROJECT_ROOT, data_dir, guard, has_checkpoints


def main():
    import torch

    iters = int(os.environ.get("NANOCHAT_SFT_ITERS", "50"))
    print("=" * 70)
    print(f"step6：中文 SFT（chat_sft，练习版 {iters} 迭代）")
    print("=" * 70)

    guard(has_checkpoints("base_checkpoints"), "data/zh-ch/base_checkpoints/ 里还没有模型，请先运行 step4_base_train.py。")

    torch.set_num_threads(os.cpu_count() or 4)

    # 先正常导入中文版 *_ch 模块（此刻 sys.modules["tasks.common"] 还是原版，
    # 因此 common_ch 内部 "from tasks.common import ..." 拿到的还是原版基类）
    import tasks.common_ch  # noqa: F401
    import tasks.smoltalk_ch  # noqa: F401
    import tasks.mmlu_ch  # noqa: F401
    import tasks.gsm8k_ch  # noqa: F401

    # 重定向：让 chat_sft.py 的 from-import 拿到中文版
    sys.modules["tasks.common"] = sys.modules["tasks.common_ch"]
    sys.modules["tasks.smoltalk"] = sys.modules["tasks.smoltalk_ch"]
    sys.modules["tasks.mmlu"] = sys.modules["tasks.mmlu_ch"]
    sys.modules["tasks.gsm8k"] = sys.modules["tasks.gsm8k_ch"]

    # 组装 chat_sft 的 argv 并在进程内运行
    args = [
        "--eval-every=25",
        "--eval-tokens=16384",
        "--num-iterations={}".format(iters),
        "--chatcore-every=-1",
        "--run=dummy",
    ]
    sys.argv = ["scripts/chat_sft.py", *args]
    chat_sft_path = os.path.join(PROJECT_ROOT, "scripts", "chat_sft.py")
    runpy.run_path(chat_sft_path, run_name="__main__")

    print(f"\nstep6 完成！SFT 检查点已保存到 {data_dir('chatsft_checkpoints')}")
    print("下一步：python runs-in-ch/step7_chat_cli.py")


if __name__ == "__main__":
    sys.exit(main())
