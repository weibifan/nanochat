"""
step4：CPU 预训练（base_train）。

与 runcpu.sh 完全相同的模型/超参（6 层小模型、max-seq-len=512、batch=32、
total-batch-size=16384），只是数据换成了中文 shard（data/zh-ch/base_data_climbmix）。
数据加载链路复用原版：
base_train → nanochat/dataloader.py → nanochat.dataset.list_parquet_files
→ data/zh-ch/base_data_climbmix/，因此零代码修改。

注意：本机是 Windows CPU（i5-14400，无 AVX512）：
  - torch.compile 无法用（没有 MSVC cl），_common 已设 TORCH_COMPILE_DISABLE=1
    退化为 eager；
  - 用 run_module_inproc 在进程内跑并设满 16 线程（默认 10 线程要慢 ~3 倍）。

“练习版”配置：num_iterations=50（实测 ~10s/step，约 9 分钟跑完，只验证整条
流水线跑通 + 损失下降）。
  runcpu.sh 的正式配置是 --num-iterations=5000（这台机器约需 15~16 小时），
  想跑完整版把下面 argv 里的 50 改回 5000 即可。

对应原版 runcpu.sh：
  python -m scripts.base_train --depth=6 --head-dim=64 --window-pattern=L \
      --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 \
      --eval-every=100 --eval-tokens=524288 --core-metric-every=-1 \
      --sample-every=100 --num-iterations=5000 --run=dummy

检查点保存到 data/zh-ch/base_checkpoints/。
"""

import sys

from _common import run_module_inproc, guard, has_tokenizer


def main():
    print("=" * 70)
    print("step4：CPU 预训练（6 层小模型，练习版 50 迭代）")
    print("=" * 70)

    guard(has_tokenizer(), "data/zh-ch/tokenizer/tokenizer.pkl 不存在，请先运行 step2_tok_train.py。")

    run_module_inproc(
        "scripts/base_train.py",
        [
            "--depth=6",
            "--head-dim=64",
            "--window-pattern=L",
            "--max-seq-len=512",
            "--device-batch-size=32",
            "--total-batch-size=16384",
            "--eval-every=25",
            "--eval-tokens=32768",
            "--core-metric-every=-1",
            "--sample-every=25",
            "--num-iterations=50",
            "--run=dummy",
        ],
        note="torch 线程数 = 逻辑核数（16），torch.compile 已退化为 eager",
    )

    print("\nstep4 完成！检查点已保存到 data/zh-ch/base_checkpoints/")
    print("下一步：python runs-in-ch/step5_base_eval.py")


if __name__ == "__main__":
    sys.exit(main())
