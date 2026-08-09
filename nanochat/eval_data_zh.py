"""
本地 eval_data 评测数据安装/查找工具（_zh 版）。

背景：CORE / ChatCORE 评测原版代码分别从海外 S3 / HuggingFace 下载数据，
国内网络连不上。用户已把评测数据手工下载到 <repo>/eval_data/：

  eval_bundle.zip                        CORE 22 任务 bundle
    （内含 eval_bundle/core.yaml + eval_meta_data.csv + eval_data/*.jsonl）
  ai2_arc/ARC-Easy/test/...parquet        ChatCORE ARC-Easy
  ai2_arc/ARC-Challenge/test/...parquet   ChatCORE ARC-Challenge
  cais-mmlu/all/test/...parquet           ChatCORE MMLU
  gsm8k/main/test/...parquet              ChatCORE GSM8K
  openai_humaneval/openai_humaneval/test/ ChatCORE HumanEval

本模块让运行时**优先在本地 eval_data/ 查找**，找不到才回退到原版下载：

- ensure_core_eval_bundle()：把 eval_data/eval_bundle.zip 解压到
  <base_dir>/eval_bundle（与 base_eval.py 期望的目录一致，解压后原版
  evaluate_core 检测到目录存在即跳过 S3 下载）。
- install_chatcore_datasets()：把 eval_data/ 下的 parquet 复制进
  <base_dir>/task_data/<slug>/<subset>/<split>/ 并写 manifest.json
  （与 load_hub_dataset 的本地缓存布局一致，之后 tasks/* 直接读本地）。
  幂等：目标已有 manifest 则跳过。

两个函数都只依赖 get_base_dir()（尊重 NANOCHAT_BASE_DIR 环境变量）。
"""

import os
import json
import shutil
import zipfile
import tempfile

from nanochat.common import get_base_dir

# <repo>/eval_data（本文件位于 <repo>/nanochat/eval_data_zh.py）
EVAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_data"
)


# ChatCORE 5 任务的本地目录 -> nanochat 任务缓存布局映射。
# (repo_id, subset, split, 相对 eval_data 的源目录)
CHATCORE_DATASETS = [
    ("cais/mmlu", "all", "test", os.path.join("cais-mmlu", "all", "test")),
    ("allenai/ai2_arc", "ARC-Easy", "test", os.path.join("ai2_arc", "ARC-Easy", "test")),
    ("allenai/ai2_arc", "ARC-Challenge", "test", os.path.join("ai2_arc", "ARC-Challenge", "test")),
    ("openai/gsm8k", "main", "test", os.path.join("gsm8k", "main", "test")),
    ("openai/openai_humaneval", "openai_humaneval", "test",
     os.path.join("openai_humaneval", "openai_humaneval", "test")),
]


def ensure_core_eval_bundle():
    """
    确保 <base_dir>/eval_bundle 存在：优先解压本地 eval_data/eval_bundle.zip，
    本地没有时返回 False（由调用方回退到原版 S3 下载）。
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if os.path.exists(eval_bundle_dir):
        return True
    zip_path = os.path.join(EVAL_DATA_DIR, "eval_bundle.zip")
    if not os.path.exists(zip_path):
        return False
    print(f"[eval_data_zh] 解压本地 {zip_path} -> {eval_bundle_dir}")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        if not os.path.isdir(extracted_bundle_dir):
            raise RuntimeError(f"eval_bundle.zip 顶层目录不是 eval_bundle/: {extracted_bundle_dir}")
        os.makedirs(base_dir, exist_ok=True)
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print(f"[eval_data_zh] CORE eval_bundle 就绪: {eval_bundle_dir}")
    return True


def install_chatcore_datasets():
    """
    把 eval_data/ 下 5 个 ChatCORE 数据集的 parquet 安装进
    <base_dir>/task_data/<slug>/<subset>/<split>/（load_hub_dataset 的本地缓存布局）。
    幂等：目标已有 manifest.json 则跳过。返回本次新安装的任务数。
    """
    base_dir = get_base_dir()
    installed = 0
    for repo_id, subset, split, rel in CHATCORE_DATASETS:
        slug = repo_id.replace("/", "--")
        dest = os.path.join(base_dir, "task_data", slug, subset, split)
        manifest = os.path.join(dest, "manifest.json")
        if os.path.exists(manifest):
            print(f"[eval_data_zh] skip {slug}/{subset}/{split} (已安装)")
            continue
        src = os.path.join(EVAL_DATA_DIR, rel)
        if not os.path.isdir(src):
            print(f"[eval_data_zh] WARN {slug}/{subset}/{split}: 本地源目录不存在 {src}")
            continue
        parquets = sorted(
            f for f in os.listdir(src)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        )
        if not parquets:
            print(f"[eval_data_zh] WARN {slug}/{subset}/{split}: {src} 里没有 parquet")
            continue
        os.makedirs(dest, exist_ok=True)
        for f in parquets:
            shutil.copy2(os.path.join(src, f), os.path.join(dest, f))
        with open(manifest, "w", encoding="utf-8") as fh:
            json.dump(parquets, fh)
        print(f"[eval_data_zh] installed {slug}/{subset}/{split} <- {src} ({len(parquets)} parquet)")
        installed += 1
    return installed
