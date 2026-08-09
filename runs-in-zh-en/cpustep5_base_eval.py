"""
cpustep5：基座评估（对应 runcpu.sh 的 `base_eval` 段，_zh 版启用 CORE）。

用 scripts/base_eval_zh.py（CORE 数据本地优先）跑 core / bpb / sample：
  python -m scripts.base_eval_zh --device-batch-size=1 --split-tokens=16384 \
      --max-per-task=16 --eval=core,bpb,sample

- core：CORE 22 任务。数据优先来自本地 <repo>/eval_data/eval_bundle.zip
  （解压到 data/zh-en/eval_bundle，不再走海外 S3）；--max-per-task 默认 16
  用于 CPU 上限制每个任务的样本数（runcpu.sh 同值）。
- bpb / sample：与之前一致。

用法：
  python runs-in-zh-en/cpustep5_base_eval.py
  NANOCHAT_EVAL_MAX_PER_TASK=100 python runs-in-zh-en/cpustep5_base_eval.py  # 提高样本数
"""

import os
import sys

from _common import timed, run_module, guard, has_checkpoints

MAX_PER_TASK = os.environ.get("NANOCHAT_EVAL_MAX_PER_TASK", "16")


@timed
def main():
    print("=" * 66)
    print("cpustep5：base_eval_zh（core + bpb + sample，本地 eval_data）")
    print("=" * 66)

    guard(has_checkpoints("base_checkpoints"),
          "data/zh-en/base_checkpoints/ 里还没有模型，请先运行 cpustep4_base_train.py。")

    run_module([
        "scripts.base_eval_zh",
        "--device-batch-size=1",
        "--split-tokens=16384",
        "--eval=core,bpb,sample",
        "--max-per-task={}".format(MAX_PER_TASK),
    ])
    print("\ncpustep5 完成！下一步：python runs-in-zh-en/cpustep6_chat_sft.py")


if __name__ == "__main__":
    sys.exit(main())