"""
中文版 GSM8K 任务（gsm8k_ch.py）→ GSM8K_zh。

原版 tasks/gsm8k.py 加载 openai/gsm8k（英文数学应用题，含 <<表达式=结果>> 工具调用）；
中文版用 testUser/GSM8K_zh（GPT-3.5 直译 GSM8K），字段：
  question / answer_only / answer / question_zh / answer_zh / split
  （train=7473 / test=1319，test 也带 answer_only，可直接当验证集）

注意两个格式坑（详见 中文数据集选型.md §4.3）：
1. 题目/答案是 question_zh / answer_zh，不是 question / answer，必须换字段名；
2. answer_zh 是纯中文解题过程，**没有**保留 "<< >>" 工具调用，也**没有**
   "#### 数字" 结尾标记 → 构造时补上 "\n#### {answer_only}"，
   让原版 GSM_RE / extract_answer / evaluate 的格式假设继续成立。

助手回复渲染成 list of parts（与原版一致），因此 evaluate/reward 逻辑可原样复用。
"""

import re
import random

from tasks.common_ch import Task, load_data_dir_json


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion):
    """
    与原版 tasks/gsm8k.py 相同：提取 "#### 数字" 后面的数值。
    官方归一化代码参考：
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8KCh(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8KCh subset must be main|socratic"
        assert split in ["train", "test"], "GSM8KCh split must be train|test"
        self.subset = subset
        self.split = split
        rows = load_data_dir_json("gsm8k_zh", "GSM8K_zh.json")
        self.rows = [r for r in rows if r.get("split") == split]
        assert self.rows, f"no GSM8K_zh rows found for split={split}"
        # seed=42 确定性打乱（与原版一致），保证多科交错
        rng = random.Random(42)
        rng.shuffle(self.rows)
        self.length = len(self.rows)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.rows[index]
        question = row["question_zh"]
        answer_zh = str(row["answer_zh"]).strip()
        answer_only = str(row["answer_only"]).strip()
        # 原版 answer 的格式是 "...推理...\n#### 答案"，中文版补上结尾标记
        assistant_text = f"{answer_zh}\n#### {answer_only}"
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        """给定 (conversation, completion)，返回 0/1。与原版语义相同。"""
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        assistant_message = conversation["messages"][-1]
        assert assistant_message["role"] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message["content"], list), "This is expected to be a list of parts"
        last_text_part = assistant_message["content"][-1]["text"]
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        return int(pred_num == ref_num)

    def reward(self, conversation, assistant_response):
        """RL 阶段复用 evaluate。"""
        return float(self.evaluate(conversation, assistant_response))


# 兼容别名
GSM8K = GSM8KCh
