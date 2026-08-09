"""
SFT 数据预下载脚本（ModelScope 镜像版，zh-en）。

把 nanochat SFT 所需的三个英文数据集从 **ModelScope 镜像**下载到本地缓存
（国内可直连，见 `英文数据集国内访问情况.md`），供 `tasks/common_zh.py`
的 `load_hub_dataset` 直接读取，全程不联网 HuggingFace：

| Task | repo_id | ModelScope 镜像 | 下载文件 |
|------|---------|----------------|----------|
| SmolTalk | `HuggingFaceTB/smol-smoltalk` | `HuggingFaceTB/smol-smoltalk` | `data/train-0000{0..3}-of-00004.parquet`，test 1 片 |
| MMLU   | `cais/mmlu` | `cais/mmlu` | `all/auxiliary_train-*.parquet` + `all/test-*.parquet`（SFT 混料只取 all） |
| GSM8K  | `openai/gsm8k` | `AI-ModelScope/gsm8k` | `main/train-00000-of-00001.parquet` + `main/test-00000-of-00001.parquet` |

落地布局与官方 `load_hub_dataset` 一致：
`<base_dir>/task_data/<slug>/<subset>/<split>/<idx:05d>.parquet` + `manifest.json`，
其中 slug = repo_id.replace("/", "--")。因此 `tasks/` 下的三个任务文件无需修改。

本脚本幂等：已下载且已写入 manifest 的 shard 自动跳过。
用法：
  python tasks/download_sft_data_zh.py
运行时要求 NANOCHAT_BASE_DIR 已指向目标数据目录（默认 <repo>/data）。
本文件放在 repo 根的 tasks/ 包下（与 *_ch 一致），由 step2 以
`python tasks/download_sft_data_zh.py` 运行。
"""

import os
import sys
import json
import time
import urllib.request

# 让脚本在 repo 内任意 cwd 下都能 import nanochat（向上找到含 pyproject.toml 的目录）
HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = HERE
while True:
    if os.path.exists(os.path.join(_PROBE, "pyproject.toml")):
        if _PROBE not in sys.path:
            sys.path.insert(0, _PROBE)
        break
    _parent = os.path.dirname(_PROBE)
    if _parent == _PROBE:
        break
    _PROBE = _parent

MODELSCOPE_BASE = "https://www.modelscope.cn/datasets"

# (repo_id, subset, split) -> [(镜象 scope 仓库, 远端 hf 风格路径)]
TASKS = {
    "HuggingFaceTB/smol-smoltalk": {
        "default": {
            "train": ("HuggingFaceTB/smol-smoltalk", [
                "data/train-00000-of-00004.parquet",
                "data/train-00001-of-00004.parquet",
                "data/train-00002-of-00004.parquet",
                "data/train-00003-of-00004.parquet",
            ]),
            "test": ("HuggingFaceTB/smol-smoltalk", [
                "data/test-00000-of-00001.parquet",
            ]),
        },
    },
    "cais/mmlu": {
        "all": {
            "auxiliary_train": ("cais/mmlu", [
                "all/auxiliary_train-00000-of-00001.parquet",
            ]),
            "test": ("cais/mmlu", [
                "all/test-00000-of-00001.parquet",
            ]),
        },
    },
    "openai/gsm8k": {
        "main": {
            "train": ("AI-ModelScope/gsm8k", [
                "main/train-00000-of-00001.parquet",
            ]),
            "test": ("AI-ModelScope/gsm8k", [
                "main/test-00000-of-00001.parquet",
            ]),
        },
    },
}


def _download(url, filepath, retries=5, timeout=120):
    """流式下载单个文件；先写 .tmp 再原子改名。已存在且非空则跳过。"""
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        print(f"  skip {os.path.basename(filepath)} (already exists)")
        return True
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp = filepath + ".tmp"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            os.replace(tmp, filepath)
            print(f"  ok {os.path.basename(filepath)} ({os.path.getsize(filepath)/1e6:.1f} MB)")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt}/{retries} failed: {e}")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(2 ** attempt)
    print(f"  FAILED: {url}")
    return False


def download_task(repo_id, task_cfg):
    """下载单个 task 的所有 split；返回 (ok_count, total)."""
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    slug = repo_id.replace("/", "--")
    ok = 0
    total = 0
    for subset, splits in task_cfg.items():
        for split, (scope_repo, remote_paths) in splits.items():
            out_dir = os.path.join(base_dir, "task_data", slug, subset, split)
            os.makedirs(out_dir, exist_ok=True)
            manifest_path = os.path.join(out_dir, "manifest.json")
            existing = []
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    existing = json.load(f)
            shards = []
            for shard_index, remote in enumerate(sorted(remote_paths)):
                total += 1
                fname = f"{shard_index:05d}.parquet"
                if fname in existing:
                    shards.append(fname)
                    ok += 1
                    continue
                url = f"{MODELSCOPE_BASE}/{scope_repo}/resolve/master/{remote}"
                local = os.path.join(out_dir, fname)
                if _download(url, local, timeout=600):
                    ok += 1
                    existing.append(fname)
                    shards.append(fname)
            with open(manifest_path, "w") as f:
                json.dump(sorted(set(existing)), f)
            print(f"  [{repo_id} {subset}/{split}] {len(existing)} shards in {out_dir}")
    return ok, total


def main():
    print("=" * 70)
    print("预下载 SFT 数据（ModelScope 镜像）→ 本地 task_data/")
    from nanochat.common import get_base_dir
    print(f"  NANOCHAT_BASE_DIR = {get_base_dir()}")
    print("=" * 70)
    for repo, cfg in TASKS.items():
        print(f"[{repo}]")
        ok, total = download_task(repo, cfg)
        print(f"  done {ok}/{total} 个文件\n")
    print("SFT 数据预下载完成！")


if __name__ == "__main__":
    sys.exit(main())