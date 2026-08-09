"""
中文版预训练数据模块（dataset_ch.py）。

对应 中文数据集选型.md §4 对 nanochat/dataset.py 的改造思路：
不修改原文件，而是生成一个独立的 _ch 版本。

职责与 dataset.py 一一对应：
- 原始 dataset.py：从 HF 按需下载 karpathy/climbmix-400b-shuffle 的 shard，迭代 parquet。
- 本文件 dataset_ch.py：从 ModelScope 按需下载 OpenBMB/Ultra-FineWeb 的 zh 分片，
  重打包成与 ClimbMix 完全相同的格式（单列 "text"、~250M 字符/shard、zstd），
  再提供与 dataset.py 同签名的 list_parquet_files / parquets_iter_batched。

为什么输出目录叫 base_data_climbmix？
  原版 nanochat/dataset.py 把 DATA_DIR 硬编码为 base_dir/base_data_climbmix。
  当我们用 NANOCHAT_BASE_DIR=<data 绝对路径> 启动时，原版脚本读的就是
  <data>/base_data_climbmix。把中文 shard 写进同名目录，就能让
  tok_train / base_train / tok_eval / base_eval 一行不改地消费中文数据。

数据来源：OpenBMB/Ultra-FineWeb 的 zh split
  - 位置：data/ultrafineweb_zh/，共 256 个 parquet
  - 命名：ultrafineweb-zh-part-{NNN}-of-256.parquet（NNN = 001..256）
  - 每片约 1.21GB（zstd 压缩）/ 约 1.2G 字符
  - 列：content(string) / score(float) / source(string)
  - 抽样：不需要下载全部 256 片，按需下载若干片（隔片抽样）即可
"""

import os
import time
import argparse
import random
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# 数据源（与 dataset.py 的 BASE_URL 对应）

MODELSCOPE_BASE = "https://modelscope.cn/datasets/OpenBMB/Ultra-FineWeb/resolve/master"
HF_BASE = "https://huggingface.co/datasets/openbmb/Ultra-FineWeb/resolve/main"
ZH_SUBDIR = "data/ultrafineweb_zh"

ZH_PARTS = 256  # 一共 256 片，编号 001..256
part_filename = lambda index: f"ultrafineweb-zh-part-{index:03d}-of-256.parquet"

# 本地目录（get_base_dir 尊重 NANOCHAT_BASE_DIR 环境变量）
base_dir = get_base_dir()
RAW_DIR = os.path.join(base_dir, "ultrafineweb_zh")       # 下载的原始 zh 分片
SHARD_DIR = os.path.join(base_dir, "base_data_climbmix")  # 重打包后的 shard（与 dataset.py 同名）

# 重打包参数（与 dev/repackage_data_reference.py 保持一致）
CHARS_PER_SHARD = 250_000_000  # 每 shard ~250M 字符
ROW_GROUP_SIZE = 1024          # 行组大小，2 的倍数，便于分布式读取
SHUFFLE_SEED = 42


# -----------------------------------------------------------------------------
# 下载

def download_single_file(url, filepath, retries=5):
    """下载单个文件，带重试与指数退避；先写 .tmp 再原子改名。"""
    if os.path.exists(filepath):
        return True
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  ok {os.path.basename(filepath)} ({os.path.getsize(filepath)/1e6:.1f} MB)")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  attempt {attempt}/{retries} failed for {url}: {e}")
            for p in [filepath + ".tmp", filepath]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            if attempt < retries:
                time.sleep(2 ** attempt)
    print(f"  FAILED: {url}")
    return False


def download_parts(parts, workers=4):
    """
    下载指定编号的 zh 分片到 RAW_DIR。primary 用 ModelScope，失败回退 HF。
    已存在的文件自动跳过。
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    indices = [int(p) for p in parts]
    # 每个 index 准备两份 URL（ModelScope 优先，HF 兜底）
    jobs = []
    for index in indices:
        filename = part_filename(index)
        filepath = os.path.join(RAW_DIR, filename)
        if os.path.exists(filepath):
            print(f"Skipping {filename} (already exists)")
            continue
        jobs.append((filename, filepath, index))

    if not jobs:
        print("All raw parts already on disk.")
        return

    print(f"Downloading {len(jobs)} zh parts ({sum(1 for _ in jobs)} files) to {RAW_DIR} ...")
    t0 = time.time()
    with Pool(processes=workers) as pool:
        results = pool.map(_download_job, jobs)
    ok = sum(1 for r in results if r)
    print(f"Done! {ok}/{len(jobs)} parts downloaded in {time.time()-t0:.0f}s")
    if ok < len(jobs):
        raise RuntimeError("Some parts failed to download. Check network and retry.")


def _download_job(job):
    filename, filepath, index = job
    ms_url = f"{MODELSCOPE_BASE}/{ZH_SUBDIR}/{filename}"
    hf_url = f"{HF_BASE}/{ZH_SUBDIR}/{filename}"
    print(f"Downloading {filename} ...")
    if not download_single_file(ms_url, filepath):
        print(f"  ModelScope failed for {filename}, trying HF fallback ...")
        if not download_single_file(hf_url, filepath):
            return False
    return True


def default_parts(n):
    """在 001..256 上均匀隔片抽样，保证来源多样（不要只取前 N 片）。"""
    assert 1 <= n <= ZH_PARTS
    if n == 1:
        return [1]
    return [round(1 + k * (ZH_PARTS - 1) / (n - 1)) for k in range(n)]


# -----------------------------------------------------------------------------
# 重打包

def repackage(parts, out_dir=SHARD_DIR, chars_per_shard=CHARS_PER_SHARD,
              row_group_size=ROW_GROUP_SIZE, seed=SHUFFLE_SEED, force=False):
    """
    把下载好的 zh 分片重打包成 ClimbMix 格式：
      content/score/source 三列 -> 只留 content 改名 text
      每片内部洗牌（seed 派生） -> 累积到 ~250M 字符 -> 写一个 shard
    最后一个 shard 自动成为验证集（dataset.py 用最后一个文件做 val）。

    注意：为了控制内存，这里"逐片处理 + 片内洗牌"，不做跨片全局洗牌。
    抽样时已隔片选取，因此整体来源分布足够多样（对演示足够）。
    """
    if not force and os.path.isdir(out_dir) and any(f.endswith(".parquet") and not f.endswith(".tmp")
                                                     for f in os.listdir(out_dir)):
        print(f"Shards already exist in {out_dir}, skipping repackage (use --force to redo).")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"Repackaging parts {parts} -> {out_dir} ...")

    buffer_docs = []       # 累积的文档
    buffer_chars = 0       # 已累积字符
    shard_index = 0
    total_chars = 0
    t0 = time.time()

    def flush_shard():
        nonlocal buffer_docs, buffer_chars, shard_index
        if not buffer_docs:
            return
        shard_path = os.path.join(out_dir, f"shard_{shard_index:05d}.parquet")
        table = pa.Table.from_pydict({"text": buffer_docs})
        pq.write_table(
            table, shard_path,
            row_group_size=row_group_size,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        print(f"  Wrote {shard_path} | docs={len(buffer_docs)} chars={buffer_chars:,}")
        buffer_docs = []
        buffer_chars = 0
        shard_index += 1

    for index in parts:
        part_path = os.path.join(RAW_DIR, part_filename(index))
        if not os.path.exists(part_path):
            print(f"WARNING: missing {part_path}, skipping")
            continue
        print(f"  Reading {os.path.basename(part_path)} ...")
        table = pq.read_table(part_path, columns=["content"])
        docs = table.column("content").to_pylist()
        # 片内洗牌（派生种子，保证可复现）
        rng = random.Random(seed + index)
        rng.shuffle(docs)
        for doc in docs:
            buffer_docs.append(doc)
            buffer_chars += len(doc)
            total_chars += len(doc)
            if buffer_chars >= chars_per_shard:
                flush_shard()
    flush_shard()  # 收尾

    # 保证至少 2 个 shard（一个 train 用 + 一个 val 用）
    shards = [f for f in os.listdir(out_dir) if f.endswith(".parquet") and not f.endswith(".tmp")]
    print(f"Repackage done: {len(shards)} shards, {total_chars/1e9:.2f}B chars, {time.time()-t0:.0f}s")
    if len(shards) < 2:
        raise RuntimeError("Need at least 2 shards (train + val), got {len(shards)}.")


# -----------------------------------------------------------------------------
# 迭代接口（与 nanochat/dataset.py 同签名，便于对照阅读）

def list_parquet_files(data_dir=SHARD_DIR):
    """返回目录下所有 parquet 的完整路径。"""
    parquet_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    )
    return [os.path.join(data_dir, f) for f in parquet_files]


def parquets_iter_batched(split, start=0, step=1):
    """
    迭代数据，按行组分批返回文档列表。
    split="train": 取除最后一个外的所有 shard；split="val": 只取最后一个。
    start/step 用于 DDP 跳过行组。
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts


# -----------------------------------------------------------------------------
# CLI：python -m nanochat.dataset_ch --parts 1 86 171 256

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and repackage Chinese pretraining data (Ultra-FineWeb-zh)")
    parser.add_argument("--parts", type=int, nargs="*", default=None,
                        help="part indices to download (1..256). Default: evenly-spaced default_parts(4)")
    parser.add_argument("-n", "--num-parts", type=int, default=4, help="number of evenly-spaced parts if --parts not given")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="parallel download workers")
    parser.add_argument("--force", action="store_true", help="force repackage even if shards exist")
    args = parser.parse_args()

    parts = args.parts if args.parts else default_parts(args.num_parts)
    print(f"Selected parts: {parts}")
    download_parts(parts, workers=args.num_workers)
    repackage(parts, force=args.force)
