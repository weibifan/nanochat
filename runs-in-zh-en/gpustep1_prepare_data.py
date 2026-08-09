"""
gpustep1：准备数据（对应 speedrun.sh 的 data 段，ModelScope 镜像）。

- ClimbMix 预训练数据：`python -m nanochat.dataset_zh -n <shards>`
  （-n 由配置决定：fast=8、full=170；下载到 data/gpu/base_data_climbmix/，
   校验 shard 用于评估，恒会一并下载）。
- SFT 数据：`python -m tasks.download_sft_data_zh`（SmolTalk / MMLU / GSM8K，
  装进 data/gpu/task_data/，供 chat_sft 与 ChatCORE 使用，不再走海外 HF）。

全程走 ModelScope，无需科学上网。重复运行幂等（已下载的 shard 会跳过）。

用法：
  python runs-in-zh-en/gpustep1_prepare_data.py
  NANOCHAT_CONFIG=full python runs-in-zh-en/gpustep1_prepare_data.py
  NANOCHAT_GPU_NUM_SHARDS=20 python runs-in-zh-en/gpustep1_prepare_data.py  # 只下 20 个
"""

import sys

import _gpu_common as g
from _common import timed, run_module, guard


@timed
def main():
    g.print_profile()
    print("=" * 66)
    print(f"gpustep1：下载数据（ClimbMix {g.cfg('num_shards')} 个 shard + SFT 数据，ModelScope）")
    print("=" * 66)

    n_shards = int(g.cfg("num_shards"))
    run_module(["nanochat.dataset_zh", "-n", str(n_shards), "-w", "8"])

    got = g.count_shards()
    guard(got >= n_shards,
          f"ClimbMix 仅下载到 {got} 个 shard（需要 {n_shards}）。请检查网络后重跑。")
    print(f"  ClimbMix 已就绪：{got} 个 train shard（不含校验 shard）。")

    print("\n下载 SFT 数据（SmolTalk / MMLU / GSM8K，ModelScope）……")
    run_module(["tasks.download_sft_data_zh"], check=False)
    guard(g.has_sft_data(), "SFT 任务数据仍未就绪（data/gpu/task_data/ 缺失）。请检查网络后重试。")

    print("\ngpustep1 完成！下一步：python runs-in-zh-en/gpustep2_tok_train.py")


if __name__ == "__main__":
    sys.exit(main())
