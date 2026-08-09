"""
cpustep1：准备数据（对应 runcpu.sh 的 `python -m nanochat.dataset -n 8`）。

两类数据，全部走 ModelScope 镜像（本地可直连）：
1. ClimbMix 预训练语料 → data/zh-en/base_data_climbmix/（≥8 片，末端自动含 val）
   - 用 nanochat/dataset_zh.py（_zh 重定向版，BASE_URL=ModelScope）下载，
     等价 runcpu.sh 的 `python -m nanochat.dataset -n 8`。
2. SFT 任务数据（SmolTalk / MMLU / GSM8K）→ data/zh-en/task_data/
   - tasks/download_sft_data_zh.py（ModelScope 预下载，含 manifest）。
     chat_sft 通过 tasks/common_zh.py 读本地缓存，全程不联网。

本步幂等：分片数 ≥ NANOCHAT_NUM_SHARDS 时跳过下载；task_data/ 已存在则跳过。

用法：
  python runs-in-zh-en/cpustep1_prepare_data.py
  NANOCHAT_NUM_SHARDS=8 python runs-in-zh-en/cpustep1_prepare_data.py

超参：
  runcpu.sh 用 -n 8（8 片，约 800MB）。
"""

import os
import sys

from _common import timed, data_dir, count_shards, guard, has_sft_data, run_in_process, run_module

NUM_SHARDS = int(os.environ.get("NANOCHAT_NUM_SHARDS", "8"))


def climbmix():
    print("\n[1/2] ClimbMix 分片（ModelScope）")
    n = count_shards()
    print(f"  base_data_climbmix/ 现有 {n} 个分片")
    if n >= NUM_SHARDS:
        print(f"  ≥ {NUM_SHARDS} 片，已够用，跳过下载。")
    else:
        run_in_process(
            "nanochat/dataset_zh.py", ["-n", str(NUM_SHARDS), "-w", "2"],
            note="dataset_zh.py（ModelScope 镜像）下载 ClimbMix shard",
        )
    n = count_shards()
    guard(n >= 1, "base_data_climbmix/ 里没有任何分片，请检查网络或镜像可达性。")
    print(f"  现在共 {n} 个分片。")


def sft_data():
    print("\n=== [2/2] SFT 任务数据（SmolTalk / MMLU / GSM8K，ModelScope）")
    if has_sft_data():
        print("  data/zh-en/task_data/ 已存在，跳过预下载。")
        return
    run_module(["tasks.download_sft_data_zh"], check=False)
    guard(has_sft_data(), "SFT 任务数据未就绪（data/zh-en/task_data/）。请检查网络。")
    print("  SFT 任务数据就绪。")


@timed
def main():
    print("=" * 66)
    print("cpustep1：准备数据（ClimbMix + SFT 任务数据，ModelScope 镜像）")
    print(f"  NANOCHAT_BASE_DIR = {data_dir()}")
    print("=" * 66)
    climbmix()
    sft_data()
    print("\ncpustep1 完成！下一步：python runs-in-zh-en/cpustep2_tok_train.py")


if __name__ == "__main__":
    sys.exit(main())