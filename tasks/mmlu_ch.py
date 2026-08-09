"""
中文版 MMLU 任务（mmlu_ch.py）→ C-Eval。

原版 tasks/mmlu.py 加载 cais/mmlu（英文 57 学科多选题）；
中文版用 C-Eval（opencompass/ceval-exam），52 学科，结构几乎与 MMLU 一致：
  id / question / A / B / C / D / answer（字母 A~D）/ explanation（仅 dev）

C-Eval 的 split（详见 中文数据集选型.md §4.3）：
- dev：每科 5 题带解析，共 260 题
- val：答案公开，共 ~1346 题
- test：答案不公开（需官网提交评测）

因此这里把 train/test 都映射到 dev + val（合计 ~1606 题），
chat_sft 里 MMLU(subset="all", split="auxiliary_train") 与 split="test" 都可直接用。
"""

import os
import csv
import random

from tasks.common_ch import Task, render_mc_zh
from nanochat.common import get_base_dir


class MMLUCh(Task):
    """C-Eval 中文多选题。"""

    letters = ("A", "B", "C", "D")

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all"], f"subset {subset} must be all"
        assert split in ["auxiliary_train", "validation", "dev", "test"], \
            f"split {split} must be auxiliary_train|validation|dev|test"
        self.subset = subset
        self.split = split
        self.problems = self._load_ceval()
        # 与原版 MMLU 一样，seed=42 确定性打乱，保证多科交错
        rng = random.Random(42)
        rng.shuffle(self.problems)
        self.length = len(self.problems)
        # chat_sft 的 val 用 MMLU(subset="all", split="test", stop=5200)，但 C-Eval
        # dev+val 总共只有 ~1606 题。Task.__len__ 在有 stop 时直接返回 stop，
        # 会超过实际题数导致 get_example 越界。把 stop 夹紧到实际题数。
        if self.stop is not None:
            self.stop = min(self.stop, self.length)

    @property
    def eval_type(self):
        return 'categorical'

    @staticmethod
    def _load_ceval():
        base_dir = get_base_dir()
        ceval_dir = os.path.join(base_dir, "task_data", "ceval")
        assert os.path.isdir(ceval_dir), \
            f"C-Eval 数据目录不存在：{ceval_dir}。请先运行 step1_prepare_data.py 下载。"
        problems = []
        for split in ["dev", "val"]:
            split_dir = os.path.join(ceval_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for filename in sorted(os.listdir(split_dir)):
                if not filename.endswith(".csv"):
                    continue
                subject = os.path.splitext(filename)[0]
                for suffix in ("_dev", "_val"):
                    if subject.endswith(suffix):
                        subject = subject[: -len(suffix)]
                        break
                with open(os.path.join(split_dir, filename), "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        question = row.get("question", "").strip()
                        choices = [row.get(c, "").strip() for c in ("A", "B", "C", "D")]
                        answer = row.get("answer", "").strip().upper()
                        if not question or answer not in ("A", "B", "C", "D"):
                            continue
                        problems.append({
                            "question": question,
                            "choices": choices,
                            "answer": answer,
                            "subject": subject,
                        })
        assert problems, f"no C-Eval problems loaded from {ceval_dir}"
        return problems

    def num_examples(self):
        return self.length

    def get_example(self, index):
        p = self.problems[index]
        question = p["question"]
        choices = p["choices"]
        answer_idx = self.letters.index(p["answer"])
        user_message = render_mc_zh(question, self.letters, choices)
        assistant_message = self.letters[answer_idx]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        return {
            "messages": messages,
            "subject": p["subject"],  # 便于按学科分组统计
            "letters": self.letters,  # 分类式评测时用于夹紧预测
        }

    def evaluate(self, conversation, assistant_response):
        # 与原版 MMLU 相同的评估语义：预测字母 == 真实字母
        assert assistant_response in self.letters, \
            f"MMLUCh answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation["messages"][-1]["content"]
        return assistant_response == assistant_message


# 兼容别名
MMLU = MMLUCh
