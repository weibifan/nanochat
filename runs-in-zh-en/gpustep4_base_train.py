"""
gpustep4：基座预训练（GPU，对应 speedrun.sh 的 base_train 段）。

单卡 RTX 4090D：直接进程内运行 scripts/base_train.py（不启动 torchrun，
单进程 DDP 即可；nanochat 自动 bf16——SM 8.9 >= 8，不支持 FP8 故不开 --fp8）。

训练中 CORE 评测：
  base_train 里 evaluate_core 先查本地 <base_dir>/eval_bundle/，只有缺失才去
  海外 S3。因此本步骤先调用 ensure_core_eval_bundle() 把本地
  eval_data/eval_bundle.zip 解压到 data/gpu/eval_bundle/，随后 CORE 全程本地
  运行（NANOCHAT_GPU_CORE_METRIC_EVERY 控制频率，NANOCHAT_GPU_CORE_METRIC_MAX_PER_TASK
  控制每个任务样本数）。仅当仓库缺 eval_bundle.zip 时才回退 --core-metric-every=-1。

参数来自 _gpu_common 配置（NANOCHAT_CONFIG 决定，NANOCHAT_GPU_* 可覆盖）：
  fast: depth=6 / head-dim=64 / seq=512 / total-batch=16384 / 3000 迭代（~8-12 min）
  full: depth=24 / head-dim=128 / seq=1024 / total-batch=262144 / 7000 迭代（~15-18 h）

- 检查点保存到 data/gpu/base_checkpoints/。

用法：
  python runs-in-zh-en/gpustep4_base_train.py
  NANOCHAT_CONFIG=full python runs-in-zh-en/gpustep4_base_train.py
  NANOCHAT_GPU_BASE_ITERS=4000 python runs-in-zh-en/gpustep4_base_train.py
"""

import sys

import _gpu_common as g
from _common import timed, run_in_process, guard


@timed
def main():
    g.print_profile()
    print("=" * 66)
    print("gpustep4：base_train（单卡 GPU）")
    print("=" * 66)

    guard(g.has_tokenizer(), "data/gpu/tokenizer/tokenizer.pkl 不存在，请先运行 gpustep2_tok_train.py。")
    guard(g.count_shards() >= 1, "data/gpu/base_data_climbmix/ 里没有分片，请先运行 gpustep1_prepare_data.py。")

    # 训练前先解压本地 CORE bundle（eval_data/eval_bundle.zip -> data/gpu/eval_bundle），
    # 让 base_train 训练中的 evaluate_core 走本地，不触海外 S3。
    from nanochat.eval_data_zh import ensure_core_eval_bundle

    has_bundle = ensure_core_eval_bundle()
    if has_bundle:
        print("[train] 本地 CORE bundle 已就绪，训练中开启 CORE 评测。")
        core_args = [
            "--core-metric-every={}".format(g.cfg("core_metric_every")),
            "--core-metric-max-per-task={}".format(g.cfg("core_metric_max_per_task")),
        ]
    else:
        print("[train] 未找到 eval_data/eval_bundle.zip，训练中跳过 CORE（gpustep5 亦会降级）。")
        core_args = ["--core-metric-every=-1"]

    args = [
        "--depth={}".format(g.cfg("base_depth")),
        "--head-dim={}".format(g.cfg("base_head_dim")),
        "--window-pattern=L",
        "--max-seq-len={}".format(g.cfg("base_seq_len")),
        "--device-batch-size={}".format(g.cfg("base_device_batch")),
        "--total-batch-size={}".format(g.cfg("base_total_batch")),
        "--num-iterations={}".format(g.cfg("base_iters")),
        "--eval-every={}".format(g.cfg("base_eval_every")),
        "--eval-tokens={}".format(g.cfg("base_eval_tokens")),
        "--sample-every={}".format(g.cfg("base_sample_every")),
        "--save-every={}".format(g.cfg("base_save_every")),
        "--device-type=cuda",
        "--run=dummy",                 # 等同 speedrun.sh 的 WANDB_RUN=dummy
        *core_args,
    ]
    run_in_process("scripts/base_train.py", args, note="单卡 GPU 训练（bf16，SDPA 注意力）")

    guard(g.has_checkpoints("base_checkpoints"),
          "训练结束但 data/gpu/base_checkpoints/ 仍为空，请检查日志。")

    print("\ngpustep4 完成！检查点已保存到 data/gpu/base_checkpoints/")
    print("下一步：python runs-in-zh-en/gpustep5_base_eval.py")


if __name__ == "__main__":
    sys.exit(main())
