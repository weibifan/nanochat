"""
The MMLU dataset.
https://huggingface.co/datasets/cais/mmlu

_zh 版与 tasks/smoltalk_zh.py 同理：在原版 MMLU 上多一道过滤
（filter_conversations_zh），把渲染后 > 512 token 的题删掉。

为什么必需：8192 小词表下 MMLU 的"题面 + 4 个选项"渲染后偏长
（实测 auxiliary_train 35.5% 超过 512 token），bestfit 打包器遇到超长
对话整行填 padding，一旦整 batch 无监督 token 就 cross_entropy 0/0=NaN。
（GSM8K 实测只有 0.04% 超长，暂不过滤。）

过滤结果缓存到 data/zh-en/task_data/cais--mmlu/_valid_indices_<split>.pkl，
换 tokenizer 后记得删除缓存。
"""

import os

from tasks.common_zh import Task, load_hub_dataset, render_mc, filter_conversations_zh
from nanochat.common import get_base_dir


class MMLU(Task):

    letters = ('A', 'B', 'C', 'D')
    groups = ('abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions')

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all"], f"subset {subset} must be all"
        assert split in ["auxiliary_train", "validation", "dev", "test"], f"split {split} must be auxiliary_train|validation|dev|test"
        self.subset = subset
        self.split = split
        self.ds = load_hub_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)
        cache_root = os.path.join(get_base_dir(), "task_data", "cais--mmlu")

        def builder(i):
            row = self.ds[i]
            question = row["question"]
            choices = row["choices"]
            answer = row["answer"]
            user_message = render_mc(question, self.letters, choices)
            assistant_message = self.letters[answer]
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
            return {"messages": messages}

        self._valid = filter_conversations_zh(
            self.ds,
            max_tokens=512,
            cache_path=os.path.join(cache_root, f"_valid_indices_{split}.pkl"),
            conversation_builder=builder,
        )
        self.length = len(self._valid)
        assert self.length > 0, "cais/mmlu 过滤后为空，检查 tokenizer 或 max_tokens"

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return self.length

    def get_example(self, index):
        physical_index = self._valid[index]
        row = self.ds[physical_index]
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]
        subject = row["subject"]
        assert len(choices) == 4, "MMLU should have 4 choices"
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        conversation = {
            "messages": messages,
            "subject": subject,
            "letters": self.letters,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters, f"MMLU answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation['messages'][-1]['content']
        return assistant_response == assistant_message