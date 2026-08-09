"""
中文版任务工具模块（common_ch.py）。

对应 中文数据集选型.md §4.1 中对 tasks/common.py 的修改思路：
不修改原文件，而是新增本 _ch 版本，只补三样东西：

1. TaskMixture 的子类：新增 eval_type 属性（避免任何代码访问 TaskMixture.eval_type 时
   触发基类 raise NotImplementedError）。
2. render_mc_zh()：中文多选题的 prompt 渲染（原 render_mc 是英文）。
3. load_data_dir_parquet() / load_data_dir_json()：从本地 base_dir/task_data/ 加载
   中文任务数据（C-Eval、smoltalk-chinese、GSM8K_zh 等下载好的本地文件），
   替代原版从 HuggingFace 在线下载的 load_hub_dataset()。

其余的 Task / TaskSequence / HubDataset / render_mc / load_hub_dataset
直接复用原版 tasks/common.py 的实现（原版不可修改，这里只是引用）。
"""

import os
import json
import pickle

import pyarrow as pa
import pyarrow.parquet as pq

from tasks.common import (  # noqa: F401  (re-export 原版符号)
    Task,
    TaskMixture as _TaskMixture,
    TaskSequence,
    HubDataset,
    load_hub_dataset,
    render_mc,
)
from nanochat.common import get_base_dir


class TaskMixture(_TaskMixture):
    """
    与原版 tasks.common.TaskMixture 相同的混料逻辑，额外暴露一个 eval_type。
    chat_eval 会读取 task_object.eval_type 来决定走生成式还是分类式评测；
    原版 TaskMixture 继承自 Task 且未覆盖 eval_type，一旦被访问就会抛 NotImplementedError。
    这里覆盖成 'generative'（对 SFT 数据混料阶段无害）。
    """

    @property
    def eval_type(self):
        return "generative"


def render_mc_zh(question, letters, choices):
    """
    中文版 render_mc：C-Eval / CMMLU 这类中文多选题的统一渲染格式。

    与英文版保持相同的关键设计决策：
    1) 字母跟在选项内容之后（- 选项=字母），小模型对这种绑定更友好；
    2) 分隔符（=）与字母之间没有空格，保证助手回复的单个字母
       （如 "B"）与 prompt 里的字母 token 完全一致。
    """
    query = f"以下是一道单项选择题：\n{question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\n请直接回复正确选项的字母。"
    return query


def load_data_dir_parquet(rel_dir, shuffle_seed=None):
    """
    从本地 base_dir/task_data/<rel_dir>/ 加载所有 *.parquet 并拼接成一个 HubDataset。
    - rel_dir：相对 task_data 的子目录，例如 "smoltalk-chinese"、"ceval"。
    - shuffle_seed：若给出，则用该种子 shuffle（与原版 load_hub_dataset 一致的 seed=42 语义）。
    """
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "task_data", rel_dir)
    assert os.path.isdir(data_dir), (
        f"task data directory not found: {data_dir}。请先运行 runs-in-ch/step1_prepare_data.py 下载。"
    )
    parquet_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    assert parquet_files, f"no parquet files found in {data_dir}"
    tables = [pq.read_table(os.path.join(data_dir, f)) for f in parquet_files]
    # smoltalk-chinese 的各 category 文件列不完全一致（有的多 difficulty/score/classify），
    # 用 permissive 把缺失列补成 null 再拼接。
    table = pa.concat_tables(tables, promote_options="permissive")
    ds = HubDataset(table)
    if shuffle_seed is not None:
        ds = ds.shuffle(shuffle_seed)
    return ds


def filter_conversations(ds, max_tokens=512, cache_path=None):
    """
    返回 ds（HubDataset，列里有 conversations）中“可训练”的对话索引：
    - 能被 render_conversation 严格校验（user/assistant 交替）；
    - 渲染后 token 数 <= max_tokens（必须能塞进 SFT 的 max_seq_len 窗口）；
    - 至少有 1 个 assistant 监督 token。

    为什么要过滤：smoltalk-chinese 的中文多轮对话偏长（实测 51.8% 超过 513
    token），而 SFT 的 bestfit 打包器遇到超过行容量的对话会整行填 padding、
    mask 全为 0，一旦整个 batch 都这样，cross_entropy(ignore_index=-1) 会
    0/0 得到 NaN。过滤后每个对话都能放进窗口，从根上杜绝 NaN。
    （英文原版 smol-smoltalk 对话短，所以原版 loader 没有这道过滤。）

    首次计算较慢（要对几十万条对话做渲染），结果缓存到 cache_path 的 pickle，
    之后直接读取。注意：换了 tokenizer 后请删除该缓存。
    """
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from nanochat.tokenizer import get_tokenizer
    tok = get_tokenizer()
    valid = []
    for i in range(len(ds)):
        raw = ds[i]["conversations"]
        msgs = _normalize_messages(raw)
        if msgs is None:
            continue
        # 字符数粗筛：>900 字符在 1.49 字符/token 的中文密度下一定 >512 token，
        # 不用渲染直接排除（只做排除，不会误收超长对话）。
        if sum(len(m["content"]) for m in msgs) > 900:
            continue
        ids, mask = tok.render_conversation({"messages": msgs})
        if len(ids) > max_tokens or sum(mask) < 1:
            continue
        valid.append(i)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(valid, f)
    return valid


def _normalize_messages(msgs):
    """把 conversations 归一化成合法的 messages 列表；不合法返回 None。"""
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    out = []
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return None
        if not isinstance(m["content"], str):
            return None
        out.append({"role": m["role"], "content": m["content"]})
    # 可选的 system 前缀（与 render_conversation 的处理一致）
    if out[0]["role"] == "system":
        out = out[1:]
    if len(out) < 2 or out[0]["role"] != "user":
        return None
    for i, m in enumerate(out):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected_role:
            return None
    return out


def load_data_dir_json(rel_dir, filename):
    """
    从本地 base_dir/task_data/<rel_dir>/<filename> 加载一个 JSON list 并返回。
    （GSM8K_zh.json 就是这种格式：一个 dict 列表）
    """
    base_dir = get_base_dir()
    path = os.path.join(base_dir, "task_data", rel_dir, filename)
    assert os.path.exists(path), (
        f"task data file not found: {path}。请先运行 runs-in-ch/step1_prepare_data.py 下载。"
    )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), f"expected a JSON list of records, got {type(data)}"
    return data
