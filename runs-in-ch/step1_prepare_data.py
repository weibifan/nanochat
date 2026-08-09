"""
step1：准备中文数据集（下载 + 重打包 + 任务数据）。

本步骤做三件事，全部幂等（已存在的文件自动跳过）：
  1) 抽样下载 Ultra-FineWeb-zh 的 4 个分片（默认隔片取样，~5GB）并重打包成
     ClimbMix 格式 → data/zh-ch/base_data_climbmix/shard_*.parquet
     （最后一个 shard 自动成为验证集，与 dataset.py 的约定一致）
  2) 下载 SFT 对话数据 → data/zh-ch/task_data/smoltalk-chinese/*.parquet
     （opencsg/smoltalk-chinese，19 个类目）
  3) 下载数学与多选题数据：
     - data/zh-ch/task_data/gsm8k_zh/GSM8K_zh.json （testUser/GSM8K_zh）
     - data/zh-ch/task_data/ceval/ceval-exam.zip → 解压 dev/val/test
       （opencompass/ceval-exam）

用法：
  python runs-in-ch/step1_prepare_data.py [--parts 1,65,129,193]

选项：
  --parts      指定下载的 Ultra-FineWeb-zh 分片编号（1~256，逗号分隔）。
               默认隔片取 4 片，保证来源多样（不要只取前 N 片）。
  --num-parts  若要改下载片数（默认 4），用这个参数自动隔片取样。
"""

import os
import sys
import time
import zipfile

import requests

from _common import data_dir

# 让 `python runs-in-ch/step1_prepare_data.py` 能 import nanochat 包
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nanochat.dataset_ch import (  # noqa: E402
    download_parts,
    repackage,
    default_parts,
    RAW_DIR,
    SHARD_DIR,
)

# opencsg/smoltalk-chinese 的 19 个类目文件（不含 old/ 目录下的历史文件）
SMOLTALK_CATEGORIES = [
    "advice-seeking",
    "brainstorming",
    "coding",
    "creative-writing",
    "data-analysis",
    "document-qa",
    "editing",
    "everyday",
    "format-constrain",
    "information-seeking",
    "math",
    "math23k_zh_fixed",
    "planning",
    "reasoning",
    "rewrite",
    "role-playing",
    "safe",
    "summary",
    "translate",
]

SMOLTALK_BASE = "https://modelscope.cn/datasets/opencsg/smoltalk-chinese/resolve/master"
GSM8K_URL = "https://modelscope.cn/datasets/testUser/GSM8K_zh/resolve/master/GSM8K_zh.json"
CEVAL_URL = "https://modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip"


def _download(url, filepath, retries=5, timeout=120):
    """流式下载单个文件到 filepath，先写 .tmp 再原子改名；已存在则跳过。"""
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        print(f"  skip {os.path.basename(filepath)} (already exists)")
        return True
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = filepath + ".tmp"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, filepath)
            print(f"  ok {os.path.basename(filepath)} ({os.path.getsize(filepath)/1e6:.1f} MB)")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  attempt {attempt}/{retries} failed for {url}: {e}")
            if os.path.exists(filepath + ".tmp"):
                try:
                    os.remove(filepath + ".tmp")
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(2 ** attempt)
    print(f"  FAILED: {url}")
    return False


def download_smoltalk():
    print("\n[step1.2] 下载 smoltalk-chinese（SFT 中文对话）...")
    d = data_dir("task_data", "smoltalk-chinese")
    os.makedirs(d, exist_ok=True)
    ok = 0
    for category in SMOLTALK_CATEGORIES:
        filepath = os.path.join(d, f"{category}.parquet")
        if _download(f"{SMOLTALK_BASE}/{category}.parquet", filepath, timeout=180):
            ok += 1
    print(f"  smoltalk-chinese: {ok}/{len(SMOLTALK_CATEGORIES)} 类目就绪")


def download_gsm8k():
    print("\n[step1.3] 下载 GSM8K_zh（中文数学题）...")
    d = data_dir("task_data", "gsm8k_zh")
    os.makedirs(d, exist_ok=True)
    _download(GSM8K_URL, os.path.join(d, "GSM8K_zh.json"), timeout=180)


def download_ceval():
    print("\n[step1.4] 下载并解压 C-Eval（中文多选题）...")
    d = data_dir("task_data", "ceval")
    os.makedirs(d, exist_ok=True)
    if os.path.isdir(os.path.join(d, "dev")):
        print("  skip ceval (already extracted)")
        return
    zip_path = os.path.join(d, "ceval-exam.zip")
    if _download(CEVAL_URL, zip_path, timeout=180):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(d)
        os.remove(zip_path)
        print("  已解压: dev/ val/ test/")
    else:
        print("  FAILED: ceval-exam.zip 下载失败")


def main():
    parts = None
    num_parts = 4
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--parts":
            parts = [int(x) for x in argv[i + 1].split(",")]
            i += 2
        elif argv[i] == "--num-parts":
            num_parts = int(argv[i + 1])
            i += 2
        else:
            print(f"unknown arg: {argv[i]}")
            sys.exit(1)
    if parts is None:
        parts = default_parts(num_parts)

    print("=" * 70)
    print("step1：准备中文数据集")
    print(f"  NANOCHAT_BASE_DIR = {data_dir()}")
    print(f"  Ultra-FineWeb-zh 分片: {parts}")
    print("=" * 70)

    # 1) 预训练语料：下载分片 + 重打包成 ClimbMix 格式
    print("\n[step1.1] 下载 Ultra-FineWeb-zh 分片并重打包...")
    download_parts(parts, workers=4)
    repackage(parts)
    shards = sorted(
        f for f in os.listdir(SHARD_DIR)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    print(f"\n  预训练 shard 数量: {len(shards)}（最后 1 个为验证集）")
    print(f"  目录: {SHARD_DIR}")
    total_bytes = sum(os.path.getsize(os.path.join(SHARD_DIR, f)) for f in shards)
    print(f"  总大小: {total_bytes/1e9:.2f} GB")

    # 2) SFT 任务数据
    download_smoltalk()
    download_gsm8k()
    download_ceval()

    print("\n" + "=" * 70)
    print("step1 完成！下一步：python runs-in-ch/step2_tok_train.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
