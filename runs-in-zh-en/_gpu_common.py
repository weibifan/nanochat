"""
runs-in-zh-en 的 GPU（gpustepX）套件共享工具。

与 _common.py 的区别：
- 数据目录独立：默认 <项目根>/data/gpu（与 CPU 的 data/zh-en 隔离，互不覆盖），
  可用环境变量 NANOCHAT_BASE_DIR 覆盖。
- 配置 profile：NANOCHAT_CONFIG=fast|full 两套速跑/长跑参数，每个数值都可用
  NANOCHAT_GPU_* 环境变量单独覆盖（详见各 gpustepX 脚本的用法说明）。
- GPU 步骤直接以 cuda 运行，不设 CPU 线程数；torch.compile 保持开启（Linux
  CUDA + Triton 原生支持，无需 MSVC）。

导入顺序注意：_common 在 import 时会无条件把 NANOCHAT_BASE_DIR 指到 data/zh-en，
所以这里必须在 import _common 之前保存用户的显式设置，import 后再改回 GPU 目录。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))   # runs-in-zh-en/
PROJECT_ROOT = os.path.dirname(_HERE)                # nanochat repo 根

# 在 import _common 之前记下用户显式设置的 base dir（_common 会把它冲掉）
_USER_BASE_DIR = os.environ.get("NANOCHAT_BASE_DIR")

import _common  # noqa: E402  # 复用 timed/guard/run_module/install_zh_modules 等

# 恢复到 GPU 目录（默认 data/gpu；用户显式设置优先）
GPU_DATA_DIR = _USER_BASE_DIR or os.path.join(PROJECT_ROOT, "data", "gpu")
os.environ["NANOCHAT_BASE_DIR"] = GPU_DATA_DIR

# ---------------------------------------------------------------------------
# 配置 profile：fast（约 30 分钟全流程） / full（约 20 小时全流程）
# 所有数值可用 NANOCHAT_GPU_<KEY> 覆盖，例如：
#   NANOCHAT_GPU_BASE_ITERS=5000   # 覆盖 base 迭代数
# ---------------------------------------------------------------------------

CONFIG = os.environ.get("NANOCHAT_CONFIG", "fast").strip().lower()

PROFILES = {
    "fast": {
        # 数据规模：ClimbMix 8 个 shard（约 0.8 GB）+ SFT 数据（SmolTalk/MMLU/GSM8K）
        "num_shards": 8,
        "vocab_size": 8192,
        "max_chars": 120_000_000,
        # base_train：6 层 / head-dim 64 / seq 512（与 cpustep4 同规模）
        "base_depth": 6,
        "base_head_dim": 64,
        "base_seq_len": 512,
        "base_device_batch": 32,
        "base_total_batch": 16384,
        "base_iters": 3000,
        "base_eval_every": 500,
        "base_eval_tokens": 16384,
        "base_sample_every": 500,
        "base_save_every": 1000,
        # 训练中 CORE 评测（base_train --core-metric-every；本地 eval_bundle 已解压时启用）
        "core_metric_every": 2000,
        "core_metric_max_per_task": 200,
        # 单独跑 base_eval（gpustep5）
        "base_eval_batch": 16,
        "base_eval_max_per_task": 100,
        # chat_sft：1000 迭代（总 batch 继承 base=16384），ChatCORE 本地
        "sft_iters": 1000,
        "sft_eval_every": 250,
        "sft_eval_tokens": 16384,
        "chatcore_every": 250,
        "chatcore_max_cat": 200,
        # 单独跑 chat_eval（gpustep7）；样本数不宜过大：生成式 GSM8K/HumanEval
        # 每题需自回归生成，1000 题在 fast 预算内太慢
        "chat_eval_batch": 8,
        "chat_eval_max_problems": 100,
    },
    "full": {
        # 数据规模：ClimbMix 170 个 shard（约 17 GB）+ SFT 数据
        "num_shards": 170,
        "vocab_size": 32768,
        "max_chars": 2_000_000_000,
        # base_train：24 层 / head-dim 128 / seq 1024（单卡 24G 显存安全余量）
        "base_depth": 24,
        "base_head_dim": 128,
        "base_seq_len": 1024,
        "base_device_batch": 16,
        "base_total_batch": 262144,
        "base_iters": 7000,
        "base_eval_every": 1000,
        "base_eval_tokens": 262144,
        "base_sample_every": 1000,
        "base_save_every": 1000,
        # 训练中 CORE 评测（base_train --core-metric-every；本地 eval_bundle 已解压时启用）
        "core_metric_every": 2000,
        "core_metric_max_per_task": 2000,
        "base_eval_batch": 16,
        "base_eval_max_per_task": 500,
        # chat_sft：-1 = 完整 epoch（数据驱动），ChatCORE 本地
        "sft_iters": -1,
        "sft_eval_every": 500,
        "sft_eval_tokens": 131072,
        "chatcore_every": 500,
        "chatcore_max_cat": 500,
        "chat_eval_batch": 16,
        "chat_eval_max_problems": -1,
    },
}

if CONFIG not in PROFILES:
    print(f"[gpu] 未知的 NANOCHAT_CONFIG={CONFIG!r}，可用：{'/'.join(PROFILES)}。回退 fast。")
    CONFIG = "fast"


def cfg(key):
    """取当前 profile 的某个配置值，NANOCHAT_GPU_<KEY> 环境变量优先。"""
    val = os.environ.get("NANOCHAT_GPU_" + key.upper())
    return PROFILES[CONFIG][key] if val is None else type(PROFILES[CONFIG][key])(val)


def print_profile():
    p = PROFILES[CONFIG]
    print("=" * 66)
    print(f"当前配置：NANOCHAT_CONFIG={CONFIG}"
          f"{'（约 30 分钟全流程）' if CONFIG == 'fast' else '（约 20 小时全流程）'}")
    print(f"  数据目录: {GPU_DATA_DIR}")
    print(f"  数据: {p['num_shards']} shard / 词表 {p['vocab_size']}")
    print(f"  base: depth {p['base_depth']} / head-dim {p['base_head_dim']} / seq {p['base_seq_len']}"
          f" / total-batch {p['base_total_batch']} / {p['base_iters']} 迭代")
    print(f"  sft : {p['sft_iters'] if p['sft_iters'] > 0 else '完整 epoch'} 迭代 / ChatCORE every {p['chatcore_every']}")
    print("  可用环境变量 NANOCHAT_GPU_* 逐项覆盖，例如 NANOCHAT_GPU_BASE_ITERS=4000。")
    print("=" * 66)
    print()


# ---------------------------------------------------------------------------
# 与 _common 同名的数据/检查函数，但指向 GPU 数据目录
# ---------------------------------------------------------------------------

def data_dir(*parts):
    return os.path.join(GPU_DATA_DIR, *parts)


def has_tokenizer():
    return os.path.exists(data_dir("tokenizer", "tokenizer.pkl"))


def has_checkpoints(rel_dir):
    d = data_dir(rel_dir)
    return os.path.isdir(d) and any(
        os.path.isdir(os.path.join(d, f)) for f in os.listdir(d)
    )


def has_sft_data():
    return os.path.isdir(data_dir("task_data"))


def count_shards(rel_dir="base_data_climbmix"):
    d = data_dir(rel_dir)
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.endswith(".parquet") and not f.endswith(".tmp"))


def has_chatcore_source():
    """repo 的 eval_data/ 里是否带有 ChatCORE 数据（chat_sft/chat_eval 用）。

    返回 True 表示 install_chatcore_datasets 有东西可装；False 时步骤应关闭
    ChatCORE 评测（--chatcore-every=-1），否则 load_hub_dataset 会报缺数据。
    """
    from nanochat.eval_data_zh import EVAL_DATA_DIR, CHATCORE_DATASETS

    for _, _, _, rel in CHATCORE_DATASETS:
        d = os.path.join(EVAL_DATA_DIR, rel)
        if os.path.isdir(d) and any(
            f.endswith(".parquet") for f in os.listdir(d)
        ):
            return True
    return False
