"""
gpustep5：基座评估（对应 speedrun.sh 的 base_eval 段，_zh 版启用 CORE）。

用 scripts/base_eval_zh.py（CORE 数据本地优先）跑 core / bpb / sample：
  python -m scripts.base_eval_zh --device-batch-size=16 --split-tokens=<cfg> \
      --max-per-task=<cfg> --eval=core,bpb,sample

- core：CORE 22 任务。数据来自本地 <repo>/eval_data/eval_bundle.zip（解压到
  data/gpu/eval_bundle，不再走海外 S3）。若仓库里没有该 zip（克隆时被忽略），
  自动退化为只跑 bpb + sample，并给出提示。
- bpb / sample：与之前一致。

用法：
  python runs-in-zh-en/gpustep5_base_eval.py
  NANOCHAT_GPU_BASE_EVAL_MAX_PER_TASK=1000 python runs-in-zh-en/gpustep5_base_eval.py
"""

import os
import sys

import _gpu_common as g
from _common import timed, run_module, guard, has_checkpoints

EVAL_BUNDLE = os.path.join(g.PROJECT_ROOT, "eval_data", "eval_bundle.zip")


@timed
def main():
    g.print_profile()
    print("=" * 66)
    print("gpustep5：base_eval_zh（core + bpb + sample，本地 eval_data）")
    print("=" * 66)

    guard(g.has_checkpoints("base_checkpoints"),
          "data/gpu/base_checkpoints/ 里还没有模型，请先运行 gpustep4_base_train.py。")

    has_bundle = os.path.exists(EVAL_BUNDLE)
    eval_modes = "core,bpb,sample" if has_bundle else "bpb,sample"
    if not has_bundle:
        print(f"[eval] 未找到 {EVAL_BUNDLE}，本次跳过 CORE（仅 bpb+sample）。")
        print("[eval] 可选：把 eval_bundle.zip 手动放进仓库 eval_data/ 后再跑本步即可恢复 CORE。")

    run_module([
        "scripts.base_eval_zh",
        "--device-batch-size={}".format(g.cfg("base_eval_batch")),
        "--split-tokens={}".format(g.cfg("base_eval_tokens")),
        "--eval={}".format(eval_modes),
        "--max-per-task={}".format(g.cfg("base_eval_max_per_task")),
    ])
    print("\ngpustep5 完成！下一步：python runs-in-zh-en/gpustep6_chat_sft.py")


if __name__ == "__main__":
    sys.exit(main())
