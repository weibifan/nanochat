"""
step5：评测 base 模型（bpb + 采样）。

与 runcpu.sh 相同的参数（--device-batch-size=1 --split-tokens=16384 --max-per-task=16），
但把 eval 模式限为 bpb,sample：
  - bpb    ：中文 train/val 上的 bits-per-byte（最有意义的指标）
  - sample ：从模型采样几段文本做 sanity check
  - （省略 core：CORE 是英文 ICL 基准，且需要从海外 S3 下载 eval_bundle，
    对中文模型意义不大，故跳过）

对应原版 runcpu.sh：
  python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16
"""

import sys

from _common import run_module_inproc, guard, has_checkpoints


def main():
    print("=" * 70)
    print("step5：评测 base 模型（bpb + sample）")
    print("=" * 70)

    guard(has_checkpoints("base_checkpoints"), "data/zh-ch/base_checkpoints/ 里还没有模型，请先运行 step4_base_train.py。")

    run_module_inproc(
        "scripts/base_eval.py",
        [
            "--device-batch-size=1",
            "--split-tokens=16384",
            "--max-per-task=16",
            "--eval=bpb,sample",
        ],
        note="torch 线程数 = 逻辑核数（16）",
    )

    print("\nstep5 完成！下一步：python runs-in-ch/step6_chat_sft.py")


if __name__ == "__main__":
    sys.exit(main())
