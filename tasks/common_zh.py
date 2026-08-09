"""
Base class for all Tasks zh-en (local-cache _zh version).

This is the `_zh` variant of `tasks/common.py`. 与 runs-in-ch 的思路一致：
**不改原文件**，而是提供一个读**本地缓存**的 `load_hub_dataset`。

差异（仅一处）：
- `load_hub_dataset()`：原版用 `huggingface_hub` 在线 list/download parquet；
  本版**只读本地磁盘** `task_data/{slug}/{subset}/{split}/`（数据已由
  `download_sft_data_zh.py` 从 ModelScope 预下载），不产生任何网络请求。
  本地目录缺失时给出明确提示，指向 step2 的 SFT 预下载。
- 其余 `HubDataset / Task / TaskMixture / TaskSequence / render_mc`
  与官方逐字一致。

注意：chat_sft/chat_eval 内部 do `from tasks.common import ...`，
本套件通过 `_common.py` 的 `install_zh_modules()` 把 `tasks.common` 在
`sys.modules` 里重定向到本模块，因此训练进程自动读到本地缓存。
"""

import os
import json
import pickle
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir


class HubDataset:
    """
    Minimal stand-in for a HuggingFace datasets Dataset: wraps a pyarrow
    Table and offers lazy row access and a seeded shuffle.
    """

    def __init__(self, table, permutation=None):
        self.table = table
        self.permutation = permutation

    def __len__(self):
        return self.table.num_rows

    def shuffle(self, seed):
        # matches datasets.Dataset.shuffle(seed=seed) exactly, row order comes out identical
        permutation = np.random.default_rng(seed).permutation(len(self))
        return HubDataset(self.table, permutation)

    def __getitem__(self, index):
        physical_index = index if self.permutation is None else int(self.permutation[index])
        row = {column: self.table[column][physical_index].as_py() for column in self.table.column_names}
        return row


def load_hub_dataset(repo_id, subset="default", split="train"):
    """
    load_hub_dataset 的 zh-en 版：只读本地缓存，不联网。

    数据由 `download_sft_data_zh.py` 预下载到
    `<NANOCHAT_BASE_DIR>/task_data/<slug>/<subset>/<split>/`
    布局与官方 load_hub_dataset 的落盘完全一致（`*.parquet` + `manifest.json`），
    因此 mmlu/gsm8k/smoltalk 一行不改即可读到。

    若缓存缺失，抛出带修复提示的 FileNotFoundError。
    """
    base_dir = get_base_dir()
    slug = repo_id.replace("/", "--")
    shards_dir = os.path.join(base_dir, "task_data", slug, subset, split)
    manifest_path = os.path.join(shards_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"\nSFT 任务数据缓存不存在: {manifest_path}\n"
            f"请先运行 runs-in-zh-en/step2_data_tokenizer.py 的 SFT 数据预下载 "
            f"(smol-smoltalk / MMLU / GSM8K 从 ModelScope 镜像落到本地 task_data/)。"
        )
    with open(manifest_path, "r") as f:
        filenames = json.load(f)
    shard_paths = [os.path.join(shards_dir, filename) for filename in filenames]
    tables = [pq.read_table(path) for path in shard_paths]
    table = pa.concat_tables(tables)
    return HubDataset(table)


class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.
    """

    def __init__(self, start=0, stop=None, step=1):
        # allows a lightweight logical view over a dataset
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop  # could be None here
        self.step = step

    @property
    def eval_type(self):
        # one of 'generative' | 'categorical'
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step  # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}"  # prevent footguns
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):
    """
    For SFT Training it becomes useful to train on a mixture of datasets.
    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        # tasks is a list of Task objects
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        # Build list of all (task_idx, local_idx) pairs
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # Deterministically shuffle to mix tasks throughout training
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # Note: this is not the most elegant or best solution, but it's ok for now

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        """
        Access conversations according to a deterministic shuffle of all examples.
        This ensures tasks are mixed throughout training, regardless of dataset size.
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for mixture with {self.num_conversations} conversations"
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """
    For SFT Training sometimes we want to sequentially train on a list of tasks.
    This is useful for cases that require a training curriculum.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for sequence with {self.num_conversations} conversations"
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length


def render_mc(question, letters, choices):
    """
    The common multiple choice rendering format we will use.

    Note two important design decisions:
    1)
    Bigger models don't care as much, but smaller models prefer to have
    the letter *after* the choice, which results in better binding.
    2)
    There is no whitespace between the delimiter (=) and the letter.
    This is actually critical because the tokenizer has different token ids
    for " A" vs. "A". The assistant responses will be just the letter itself,
    i.e. "A", so it is important that here in the prompt it is the exact same
    token, i.e. "A" with no whitespace before it. Again, bigger models don't care
    about this too much, but smaller models do care about some of these details.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


def filter_conversations_zh(ds, max_tokens=512, cache_path=None, conversation_builder=None):
    """
    返回 ds（HubDataset）中"可训练"的对话索引：
    - 渲染后 token 数 <= max_tokens（必须能塞进 SFT 的 max_seq_len 窗口）；
    - 至少有 1 个 assistant 监督 token（否则整 batch 无监督 → loss=NaN）。

    conversation_builder(index) 返回 render_conversation 接受的对话 dict：
      {"messages": [...]}
    缺省时直接用 ds[i]["messages"]（适用于 smol-smoltalk 这类有 messages 列的）。

    为什么要过滤：8192 小词表下对话普遍偏长（实测 smol-smoltalk train 68.7%、
    MMLU auxiliary_train 35.5% 超过 512 token），而 SFT 的 bestfit 打包器遇到
    超过行容量的对话会整行填 padding、mask 全为 0，一旦整个 batch 都这样，
    cross_entropy(ignore_index=-1) 会 0/0 得到 NaN。过滤后每个对话都能放进
    窗口，从根上杜绝 NaN。（runs-in-ch 的中文套件有同样的 filter_conversations。）

    首次计算较慢，结果缓存到 cache_path 的 pickle；换 tokenizer 后请删除缓存。
    """
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from nanochat.tokenizer import get_tokenizer
    tok = get_tokenizer()
    valid = []
    for i in range(len(ds)):
        if conversation_builder is not None:
            conversation = conversation_builder(i)
        else:
            conversation = {"messages": ds[i]["messages"]}
        try:
            ids, mask = tok.render_conversation(conversation)
        except (AssertionError, ValueError, TypeError, IndexError):
            continue
        if len(ids) > max_tokens or sum(mask) < 1:
            continue
        valid.append(i)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(valid, f)
    return valid


if __name__ == "__main__":
    # very lightweight test of slicing (needs pre-downloaded local cache)
    ds = load_hub_dataset("openai/gsm8k", "main", "train")
    print("Local GSM8K rows: ", len(ds))