"""
中文版 SmolTalk 任务（smoltalk_ch.py）。

原版 tasks/smoltalk.py 从 HuggingFace 加载 HuggingFaceTB/smol-smoltalk；
中文版改从 ModelScope 的 opencsg/smoltalk-chinese 下载到本地
base_dir/task_data/smoltalk-chinese/*.parquet 后读取。

smoltalk-chinese 与 smol-smoltalk 类型 1:1 对应（同为 Magpie 流程生成的多轮
中文对话），但列结构不同：
- 原版：messages（List[{role, content}]，可含首条 system）
- 中文：conversations（List[{role, content}]，user/assistant 严格交替，无 system 列）
       + system_prompt_key / magpie_model / n_turn / text_len

本 loader 把 conversations 归一化成 render_conversation 需要的 messages 结构，
train/test 用确定性的末尾 5% 留出（smoltalk-chinese 没有官方 test split）。

重要：smoltalk-chinese 的中文多轮对话偏长（实测 51.8% 超过 513 token），
而我们 SFT 的 max_seq_len=512（继承自预训练）。超过窗口的对话会被 bestfit
打包器整行丢弃（全 padding），甚至整 batch 无监督 token 导致 loss=NaN。
因此用 common_ch.filter_conversations 过滤出能放进窗口的对话（缓存到 pickle，
换 tokenizer 后需删除缓存）。
"""

import os

from tasks.common_ch import Task, load_data_dir_parquet, filter_conversations
from nanochat.common import get_base_dir


class SmolTalkCh(Task):
    """opencsg/smoltalk-chinese。train/test 用确定性的末尾 5% 留出。"""

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalkCh split must be train|test"
        self.split = split
        ds = load_data_dir_parquet("smoltalk-chinese", shuffle_seed=42)
        self.ds = ds
        cache_path = os.path.join(
            get_base_dir(), "task_data", "smoltalk-chinese", "_valid_indices.pkl"
        )
        self._valid = filter_conversations(ds, max_tokens=512, cache_path=cache_path)
        assert self._valid, "no renderable conversations found in smoltalk-chinese dataset"
        n = len(self._valid)
        self.test_size = max(1, int(0.05 * n))
        if split == "train":
            self.length = n - self.test_size
            self.offset = 0
        else:
            self.length = self.test_size
            self.offset = n - self.test_size

    @staticmethod
    def _normalize(msgs):
        """兼容旧的 _normalize 调用（返回 messages 或 None）。"""
        from tasks.common_ch import _normalize_messages
        return _normalize_messages(msgs)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        physical = self._valid[self.offset + index]
        messages = self._normalize(self.ds[physical]["conversations"])
        return {"messages": messages}


# 兼容别名：step6 重定向 sys.modules["tasks.smoltalk"] 后，
# chat_sft.py 里的 "from tasks.smoltalk import SmolTalk" 拿到的就是这个类。
SmolTalk = SmolTalkCh
