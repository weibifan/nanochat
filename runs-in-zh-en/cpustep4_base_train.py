"""
cpustep4：基座预训练（对应 runcpu.sh 的 `base_train` 段）。

与 runcpu.sh 完全一致的模型/超参（6 层 / head-dim 64 / max-seq-len 512 /
batch 32 / total-batch 16384），仅把官方 5000 迭代换成可调的练习值。

参数（== runcpu.sh 官方值，练习版只改如下四项）：
  --depth=6 --head-dim=64 --window-pattern=L --max-seq-len=512
  --device-batch-size=32 --total-batch-size=16384
  --eval-every=100        # 练习版 50
  --eval-tokens=524288    # 练习版 32768
  --core-metric-every     # 训练中 CORE：本地 eval_bundle 已解压时开启（见下）
  --sample-every=100      # 练习版 50
  --num-iterations=5000   # 练习版 190（NANOCHAT_PRETRAIN_ITERS 覆盖）
  --run=dummy

训练中 CORE 评测：
  base_train 里 evaluate_core 先查本地 <base_dir>/eval_bundle/，只有缺失才去
  海外 S3。本步骤运行前先调用 ensure_core_eval_bundle() 把本地
  eval_data/eval_bundle.zip 解压到 data/zh-en/eval_bundle/，随后 CORE 全程本地。
  频率 NANOCHAT_CORE_METRIC_EVERY 默认 2000（练习版 190 迭代只在最后一步触发，
  CPU 开销 ~1 min 可忽略；完整版每 2000 步跑一次），每个任务样本数
  NANOCHAT_CORE_METRIC_MAX_PER_TASK 练习版默认 50 / 完整版 500。
  仅当仓库缺 eval_bundle.zip 时才回退 --core-metric-every=-1。

进程内运行（run_in_process）以设 CPU 线程数；torch.compile 已在 _common
退化为 eager（本机可能无 MSVC cl）。

实测耗时（本机 16 核 CPU，190 步实测 17.3 min，其中训练循环 16.0 min
≈ 0.084 min/步、固定开销 ~1.3 min）：
  练习版 190 迭代 ≈ 17-18 min，整链除下载外合计 ≈ 25 min。
  完整版 5000 迭代 ≈ 15-16 h（官方估算，含多轮 eval，纯训练按实测速率约 7 h）。

检查点保存到 data/zh-en/base_checkpoints/。

用法：
  python runs-in-zh-en/cpustep4_base_train.py
  NANOCHAT_PRETRAIN_ITERS=40 python runs-in-zh-en/cpustep4_base_train.py    # 快速版
  NANOCHAT_PRETRAIN_ITERS=5000 python runs-in-zh-en/cpustep4_base_train.py  # 完整版
  NANOCHAT_CORE_METRIC_EVERY=-1 python runs-in-zh-en/cpustep4_base_train.py # 关训练中 CORE
"""

import os
import sys

from _common import timed, run_in_process, guard, has_tokenizer


@timed
def main():
    guard(has_tokenizer(), "data/zh-en/tokenizer/tokenizer.pkl 不存在，请先运行 cpustep2_tok_train.py。")

    iters = int(os.environ.get("NANOCHAT_PRETRAIN_ITERS", "190"))
    full = iters >= 5000
    print("=" * 66)
    print(f"cpustep4：base_train（6 层小模型，{'完整版' if full else '练习版'} {iters} 迭代）")
    print("=" * 66)

    # 训练前先解压本地 CORE bundle（eval_data/eval_bundle.zip -> data/zh-en/eval_bundle），
    # 让 base_train 训练中的 evaluate_core 走本地，不触海外 S3。
    from nanochat.eval_data_zh import ensure_core_eval_bundle

    has_bundle = ensure_core_eval_bundle()
    core_metric_every = os.environ.get("NANOCHAT_CORE_METRIC_EVERY", "2000")
    core_metric_max_per_task = os.environ.get("NANOCHAT_CORE_METRIC_MAX_PER_TASK", "500" if full else "50")
    if has_bundle:
        print("[train] 本地 CORE bundle 已就绪，训练中开启 CORE 评测。")
        core_args = [
            "--core-metric-every={}".format(core_metric_every),
            "--core-metric-max-per-task={}".format(core_metric_max_per_task),
        ]
    else:
        print("[train] 未找到 eval_data/eval_bundle.zip，训练中跳过 CORE（cpustep5 亦会降级）。")
        core_args = ["--core-metric-every=-1"]

    args = [
        "--depth=6",
        "--head-dim=64",
        "--window-pattern=L",
        "--max-seq-len=512",
        "--device-batch-size=32",
        "--total-batch-size=16384",
        "--eval-every=100" if full else "--eval-every=50",
        "--eval-tokens=524288" if full else "--eval-tokens=32768",
        "--sample-every=100" if full else "--sample-every=50",
        "--num-iterations={}".format(iters),
        "--run=dummy",  # 等同 runcpu.sh 的 WANDB_RUN=dummy
        *core_args,
    ]
    run_in_process("scripts/base_train.py", args, note="torch 线程数 = 逻辑核数；torch.compile 已退化")

    print("\ncpustep4 完成！检查点已保存到 data/zh-en/base_checkpoints/")
    print("下一步：python runs-in-zh-en/cpustep5_base_eval.py")


if __name__ == "__main__":
    sys.exit(main())