"""
SmolTalk by HuggingFace. Good "general" conversational dataset.
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk

_zh 版与 runs-in-ch/step6 思路一致：不修改原文件，而是在原版 SmolTalk 基础上
多一道过滤（filter_conversations_zh）：把渲染后 > 512 token（超过 SFT 的
max_seq_len 窗口）或没有 assistant 监督 token 的对话删掉。

为什么必需：8192 小词表下英文对话普遍偏长（实测 train 68.7% 超过 513 token），
bestfit 打包器遇到超长对话会整行填充 padding，整 batch 无监督 token 时
cross_entropy(ignore_index=-1) 得 0/0 = NaN，污染权重。过滤后从根上避开。
过滤结果缓存到 data/zh-en/task_data/HuggingFaceTB--smol-smoltalk/_valid_indices.pkl，
换 tokenizer 后记得删除该缓存。
"""

import os

from tasks.common_zh import Task, load_hub_dataset, filter_conversations_zh
from nanochat.common import get_base_dir


class SmolTalk(Task):
    """ smol-smoltalk dataset. train is 460K rows + filter, test is 24K rows + filter. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        self.ds = load_hub_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        cache_root = os.path.join(get_base_dir(), "task_data", "HuggingFaceTB--smol-smoltalk")
        self._valid = filter_conversations_zh(
            self.ds,
            max_tokens=512,
            cache_path=os.path.join(cache_root, f"_valid_indices_{split}.pkl"),
        )
        # 原版不设 length，用 len(ds)；现在只取过滤后的对话
        self.length = len(self._valid)
        assert self.length > 0, "smol-smoltalk 过滤后为空，检查 tokenizer 或 max_tokens"

    def num_examples(self):
        return self.length

    def get_example(self, index):
        physical_index = self._valid[index]
        row = self.ds[physical_index]
        return {"messages": row["messages"]}