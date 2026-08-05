"""
实验3 扩展：本地模拟 SFT 微调全流程
- 生成 3 个数据集的样本 JSON（格式与 HuggingFace 一致，但本地生成）
- 加载到 TaskMixture 中混合
- 跑完整 DataLoader + loss 计算
- 加载 checkpoint 做一次真实 forward
路径来自 config.yaml（相对本目录），产物输出到 runs-with-win/sample_data/
"""
import sys, os, io, time, json, random, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
torch._dynamo.config.disable = True
from tasks.common import Task, TaskMixture
from nanochat.tokenizer import RustBPETokenizer
from nanochat.gpt import GPT, GPTConfig

random.seed(42)
t_start = time.time()

# ========== 读取 config.yaml ==========
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.yaml"), "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
def rel(p):
    return os.path.normpath(os.path.join(HERE, p))

TOKENIZER_DIR = rel(cfg["tokenizer"])
sd_cfg = cfg.get("sft_data", {})
SAMPLE_DIR = rel(sd_cfg.get("sample_dir", "sample_data"))
SEQ_LEN = sd_cfg.get("seq_len", 128)
BATCH_SIZE = sd_cfg.get("batch_size", 4)
SAMPLE_SIZE = sd_cfg.get("sample_size", 30)

pt_cfg = cfg.get("pretrain", {})
CKPT_DIR = rel(pt_cfg.get("checkpoint_dir", "checkpoint"))

parser = argparse.ArgumentParser(description="实验3：生成 SFT 样本 + 管线测试")
parser.add_argument("--tokenizer", default=None)
parser.add_argument("--sample-dir", default=None)
parser.add_argument("--sample-size", type=int, default=None)
parser.add_argument("--ckpt-dir", default=None)
args = parser.parse_args()
if args.tokenizer: TOKENIZER_DIR = rel(args.tokenizer)
if args.sample_dir: SAMPLE_DIR = rel(args.sample_dir)
if args.sample_size: SAMPLE_SIZE = args.sample_size
if args.ckpt_dir: CKPT_DIR = rel(args.ckpt_dir)

if not os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.pkl")):
    sys.exit(f"❌ 未找到分词模型: {TOKENIZER_DIR}/tokenizer.pkl（请先运行 exp1）")

tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
bos_token = tokenizer.get_bos_token_id()

print(f"Tokenizer: 词表={tokenizer.get_vocab_size()}")
print(f"配置: sample输出={SAMPLE_DIR}, sample_size={SAMPLE_SIZE}")
print()

# ========== 1. 生成 3 个数据集的样本 ==========
def make_smoltalk_samples(n):
    dialogs = [
        "你能解释一下什么是量子计算吗？",
        "推荐几本好看的科幻小说。",
        "用Python写一个二分查找算法。",
        "简述一下二战爆发的原因。",
        "如何提高编程效率？",
    ]
    replies = [
        "量子计算是一种利用量子力学原理进行计算的技术。",
        "推荐《三体》、《银河帝国》、《沙丘》。",
        "以下是二分查找的Python实现：\ndef binary_search(arr, x):\n    l, r = 0, len(arr)-1\n    while l <= r:\n        mid = (l+r)//2\n        if arr[mid] == x: return mid\n        elif arr[mid] < x: l = mid+1\n        else: r = mid-1\n    return -1",
        "二战爆发的主要原因包括《凡尔赛条约》的惩罚性条款、全球经济大萧条、纳粹德国的扩张政策等。",
        "提高编程效率的方法：多读优秀源码、善用工具、注重代码设计、持续重构。",
    ]
    samples = []
    for i in range(n):
        q = dialogs[i % len(dialogs)]
        a = replies[i % len(replies)]
        samples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })
    return samples

def make_mmlu_samples(n):
    samples = []
    questions = [
        ("法国的首都是哪里？", ["伦敦", "巴黎", "柏林", "马德里"], 1),
        ("哪颗行星被称为红色星球？", ["金星", "木星", "火星", "土星"], 2),
        ("2加2等于几？", ["3", "4", "5", "6"], 1),
    ]
    letters = ('A', 'B', 'C', 'D')
    for i in range(n):
        q, choices, answer = questions[i % len(questions)]
        prompt = f"单选题：{q}\n"
        for letter, choice in zip(letters, choices):
            prompt += f"- {choice}={letter}\n"
        prompt += "\n请只回答正确选项的字母。"
        samples.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": letters[answer]}
            ]
        })
    return samples

def make_gsm8k_samples(n):
    samples = []
    problems = [
        ("小明有 5 个苹果，他又买了 3 个。现在他总共有多少个苹果？",
         "小明原来有 5 个苹果，又买了 3 个。\n<<5+3=8>>\n所以小明现在有 8 个苹果。\n#### 8"),
        ("一列火车 2 小时行驶了 120 公里。它的速度是多少公里/小时？",
         "速度等于距离除以时间。\n<<120/2=60>>\n速度是 60 公里/小时。\n#### 60"),
    ]
    for i in range(n):
        q, a = problems[i % len(problems)]
        assistant_parts = []
        parts = a.replace("<<", "\x00<<").replace(">>", ">>\x00").split("\x00")
        for part in parts:
            if part.startswith("<<"):
                inner = part[2:-2]
                if "=" in inner:
                    expr, result = inner.rsplit("=", 1)
                    assistant_parts.append({"type": "python", "text": expr.strip()})
                    assistant_parts.append({"type": "python_output", "text": result.strip()})
            elif part.strip():
                assistant_parts.append({"type": "text", "text": part})
        samples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": assistant_parts}
            ]
        })
    return samples

print("生成样本数据...")
samples_smoltalk = make_smoltalk_samples(SAMPLE_SIZE)
samples_mmlu = make_mmlu_samples(SAMPLE_SIZE)
samples_gsm8k = make_gsm8k_samples(SAMPLE_SIZE)

save_dir = SAMPLE_DIR
os.makedirs(save_dir, exist_ok=True)
for name, data in [("smoltalk", samples_smoltalk), ("mmlu", samples_mmlu), ("gsm8k", samples_gsm8k)]:
    path = os.path.join(save_dir, f"{name}_sample.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  已保存 {len(data)} 条 → {path}")

# ========== 2. 用 TaskMixture 混合 ==========
class DictTask(Task):
    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.data = data
    def num_examples(self):
        return len(self.data)
    def get_example(self, index):
        return self.data[index]

train_tasks = [
    DictTask(samples_smoltalk),
    DictTask(samples_mmlu),
    DictTask(samples_gsm8k),
]
mixture = TaskMixture(train_tasks)
print(f"\nTaskMixture 混合后: {len(mixture)} 条")
for i in range(min(3, len(mixture))):
    conv = mixture[i]
    role = conv["messages"][0]["role"]
    content = conv["messages"][0].get("content", "")
    if isinstance(content, list):
        content = str(content[:2])
    print(f"  [{i}] [{role}] {content[:50]}...")
print()

# ========== 3. 渲染 & 统计 ==========
print("渲染 ChatML & 统计:")
all_lengths = []
for i in range(len(mixture)):
    ids, mask = tokenizer.render_conversation(mixture[i], max_tokens=SEQ_LEN)
    all_lengths.append(len(ids))

import numpy as np
lens = np.array(all_lengths)
print(f"  最短: {lens.min()} | 最长: {lens.max()} | 平均: {lens.mean():.1f}")

conv0 = mixture[0]
ids0, mask0 = tokenizer.render_conversation(conv0, max_tokens=SEQ_LEN)
vis = tokenizer.visualize_tokenization(ids0, mask0)
print(f"\n  第 1 条可视化:")
print(f"  {vis}")
train_pct = sum(mask0)/len(mask0)*100
print(f"  训练比例: {sum(mask0)}/{len(mask0)} ({train_pct:.1f}%)")
print()

# ========== 4. 模拟 DataLoader ==========
print("DataLoader best-fit packing:")
def data_generator(mixture, batch_size, seq_len):
    bos = bos_token
    row_cap = seq_len + 1
    buffer = []
    idx = 0
    while len(buffer) < batch_size * 3:
        conv = mixture[idx % len(mixture)]
        ids, mask = tokenizer.render_conversation(conv, max_tokens=seq_len)
        buffer.append((ids, mask))
        idx += 1
    for _ in range(2):
        rows, mask_rows, clens = [], [], []
        for _ in range(batch_size):
            row, mrow = [], []
            padded = False
            while len(row) < row_cap:
                remaining = row_cap - len(row)
                best_i, best_len = -1, 0
                for i, (c, _) in enumerate(buffer):
                    if len(c) <= remaining and len(c) > best_len:
                        best_i, best_len = i, len(c)
                if best_i >= 0:
                    c, cm = buffer.pop(best_i)
                    row.extend(c); mrow.extend(cm)
                else:
                    cl = len(row)
                    row.extend([bos]*remaining); mrow.extend([0]*remaining)
                    padded = True; break
            content_len = cl if padded else row_cap
            rows.append(row[:row_cap]); mask_rows.append(mrow[:row_cap]); clens.append(content_len)
        yield rows, mask_rows, clens

gen = data_generator(mixture, BATCH_SIZE, SEQ_LEN)
for step, (rows, mrows, clens) in enumerate(gen):
    batch = torch.tensor(rows, dtype=torch.long)
    inputs = batch[:, :-1]
    targets = batch[:, 1:].clone()
    mask_t = torch.tensor(mrows, dtype=torch.int8)
    targets[mask_t[:, 1:] == 0] = -1
    for i, cl in enumerate(clens):
        if cl < SEQ_LEN + 1:
            targets[i, cl-1:] = -1
    train_tok = (targets != -1).sum().item()
    total_tok = targets.numel()
    lens_str = ", ".join([f"{c}({c-SEQ_LEN-1:+d})" for c in clens])
    print(f"  Step {step}: content_len={lens_str} | train={train_tok}/{total_tok} ({100*train_tok/total_tok:.1f}%)")
print()

# ========== 5. 加载 checkpoint 做一次 forward ==========
print("加载 exp2 checkpoint 做一次 forward:")
if os.path.exists(os.path.join(CKPT_DIR, "model_001000.pt")):
    from nanochat.checkpoint_manager import load_checkpoint
    CKPT_STEP = pt_cfg.get("steps", 1000)
    model_data, _, meta = load_checkpoint(CKPT_DIR, CKPT_STEP, device="cpu")

    config = GPTConfig(
        sequence_len=SEQ_LEN,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=pt_cfg.get("n_layer", 5),
        n_head=pt_cfg.get("n_head", 4),
        n_kv_head=pt_cfg.get("n_kv_head", 4),
        n_embd=pt_cfg.get("n_embd", 256),
        window_pattern=pt_cfg.get("window_pattern", "L"),
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    model = model_meta.to_empty(device="cpu")
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    gen2 = data_generator(mixture, BATCH_SIZE, SEQ_LEN)
    rows, _, _ = next(gen2)
    batch = torch.tensor(rows, dtype=torch.long)
    x = batch[:, :-1]
    y = batch[:, 1:]

    with torch.no_grad():
        loss = model(x.contiguous(), targets=y.contiguous())
    print(f"  ✅ Forward 成功 | loss={loss.item():.4f}")
else:
    print(f"  ⚠️ 未找到 exp2 checkpoint ({CKPT_DIR})，跳过模型 forward（先运行 exp2）")
print()

total = time.time() - t_start
print(f"总耗时: {total:.2f}s")
print("="*50)
print("本地 SFT 管线模拟完成 ✅")
print("="*50)
