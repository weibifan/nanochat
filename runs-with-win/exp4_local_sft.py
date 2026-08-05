"""
实验4：本地 SFT 微调
使用抽样数据和 exp2 检查点，在 CPU 上跑完整的 SFT 训练流程
路径与参数全部来自 config.yaml（相对本目录）
"""
import sys, os, io, time, json, gc, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
torch._dynamo.config.disable = True
from nanochat.tokenizer import RustBPETokenizer
from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from tasks.common import TaskMixture, Task

# ========== 读取 config.yaml ==========
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.yaml"), "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
def rel(p):
    return os.path.normpath(os.path.join(HERE, p))

TOKENIZER_DIR = rel(cfg["tokenizer"])
pt_cfg = cfg.get("pretrain", {})
CKPT_DIR = rel(pt_cfg.get("checkpoint_dir", "checkpoint"))
sd_cfg = cfg.get("sft_data", {})
SAMPLE_DIR = rel(sd_cfg.get("sample_dir", "sample_data"))
sf_cfg = cfg.get("sft", {})
SAVE_DIR = rel(sf_cfg.get("checkpoint_dir", "sft_checkpoint"))
SEQ_LEN = sf_cfg.get("seq_len", 128)
BATCH_SIZE = sf_cfg.get("batch_size", 4)
TRAIN_STEPS = sf_cfg.get("steps", 50)
LR = sf_cfg.get("lr", 0.01)

parser = argparse.ArgumentParser(description="实验4：本地 SFT 微调")
parser.add_argument("--tokenizer", default=None)
parser.add_argument("--ckpt-dir", default=None)
parser.add_argument("--sample-dir", default=None)
parser.add_argument("--save-dir", default=None)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--lr", type=float, default=None)
args = parser.parse_args()
if args.tokenizer: TOKENIZER_DIR = rel(args.tokenizer)
if args.ckpt_dir: CKPT_DIR = rel(args.ckpt_dir)
if args.sample_dir: SAMPLE_DIR = rel(args.sample_dir)
if args.save_dir: SAVE_DIR = rel(args.save_dir)
if args.steps: TRAIN_STEPS = args.steps
if args.lr: LR = args.lr

if not os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.pkl")):
    sys.exit(f"❌ 未找到分词模型: {TOKENIZER_DIR}/tokenizer.pkl（请先运行 exp1）")
if not os.path.isdir(SAMPLE_DIR):
    sys.exit(f"❌ 未找到样本目录: {SAMPLE_DIR}（请先运行 exp3）")

t_start = time.time()

# ========== 配置（来自 config.yaml） ==========
EVAL_EVERY = max(1, TRAIN_STEPS // 10)
CKPT_STEP = pt_cfg.get("steps", 1000)

# ========== 1. 加载 tokenizer ==========
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
bos_token = tokenizer.get_bos_token_id()
print(f"Tokenizer: 词表={tokenizer.get_vocab_size()}")

# ========== 2. 加载抽样数据 ==========
class JsonTask(Task):
    def __init__(self, json_path, **kwargs):
        super().__init__(**kwargs)
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    def num_examples(self):
        return len(self.data)
    def get_example(self, index):
        return self.data[index]

dataset = TaskMixture([
    JsonTask(os.path.join(SAMPLE_DIR, "smoltalk_sample.json")),
    JsonTask(os.path.join(SAMPLE_DIR, "mmlu_sample.json")),
    JsonTask(os.path.join(SAMPLE_DIR, "gsm8k_sample.json")),
])
print(f"训练数据: {len(dataset)} 条")

# ========== 3. 加载 exp2 模型 ==========
print(f"\n加载 checkpoint: {CKPT_DIR}")
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

print(f"构建模型: {config.n_layer}层/{config.n_embd}维/{config.n_head}头")
with torch.device("meta"):
    model = GPT(config)
model = model.to_empty(device="cpu")
model.init_weights()
model.load_state_dict(model_data, strict=True, assign=True)
model.train()

total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# ========== 4. 优化器 ==========
optimizer = model.setup_optimizer(matrix_lr=LR, weight_decay=0.0)

# ========== 5. SFT DataLoader ==========
def sft_loader(dataset, batch_size, seq_len, tokenizer):
    row_cap = seq_len + 1
    bos = tokenizer.get_bos_token_id()
    buffer = []
    idx = 0
    while True:
        while len(buffer) < batch_size * 3:
            conv = dataset[idx % len(dataset)]
            ids, mask = tokenizer.render_conversation(conv, max_tokens=seq_len)
            buffer.append((ids, mask))
            idx += 1

        rows, mask_rows, content_lens = [], [], []
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
                    content_len = len(row)
                    row.extend([bos] * remaining); mrow.extend([0] * remaining)
                    padded = True; break
            content_len = content_len if padded else row_cap
            rows.append(row[:row_cap]); mask_rows.append(mrow[:row_cap])
            content_lens.append(content_len)

        batch = torch.tensor(rows, dtype=torch.long)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].clone()
        mask_t = torch.tensor(mask_rows, dtype=torch.int8)
        targets[mask_t[:, 1:] == 0] = -1
        for i, cl in enumerate(content_lens):
            if cl < row_cap:
                targets[i, cl - 1:] = -1

        yield inputs, targets, content_lens

train_loader = sft_loader(dataset, BATCH_SIZE, SEQ_LEN, tokenizer)

# ========== 6. 训练循环 ==========
print(f"\n开始 SFT 训练: {TRAIN_STEPS} 步, batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}")
print(f"{'step':>5} | {'loss':>8} | {'train%':>6} | {'dt(ms)':>7}")
print("-" * 35)

losses = []
t0 = time.time()
for step in range(1, TRAIN_STEPS + 1):
    x, y, clens = next(train_loader)
    ts = time.time()
    loss = model(x, targets=y)
    train_tok = (y != -1).sum().item()
    total_tok = y.numel()
    train_pct = 100 * train_tok / total_tok

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    dt = (time.time() - ts) * 1000

    if step == 1 or step % EVAL_EVERY == 0 or step == TRAIN_STEPS:
        print(f"{step:>5} | {loss.item():>8.4f} | {train_pct:>5.1f}% | {dt:>7.1f}")

train_time = time.time() - t0
print(f"\n训练完成: {train_time:.1f}s ({TRAIN_STEPS/train_time:.1f} step/s)")
print(f"首步 loss: {losses[0]:.4f}  末步 loss: {losses[-1]:.4f}")
trend = "✅ 下降" if losses[-1] < losses[0] else "⚠️ 未下降"
print(f"趋势: {trend}")

# ========== 7. 推理测试 ==========
print(f"\n推理测试:")
model.eval()
test_questions = ["什么是注意力机制？", "推荐一本好书。"]
for q in test_questions:
    ids = tokenizer.encode(q, prepend="<|bos|>", append="<|assistant_start|>")
    prompt = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(30):
            logits = model(prompt)
            next_logit = logits[0, -1, :]
            next_id = torch.argmax(next_logit).item()
            prompt = torch.cat([prompt, torch.tensor([[next_id]])], dim=1)
            if next_id == tokenizer.encode_special("<|assistant_end|>"):
                break
    response = tokenizer.decode(prompt[0].tolist())
    print(f"  Q: {q}")
    print(f"  A: {response}")
    print()

# ========== 8. 保存检查点 ==========
os.makedirs(SAVE_DIR, exist_ok=True)
save_checkpoint(SAVE_DIR, TRAIN_STEPS,
    model.state_dict(), optimizer.state_dict(),
    {"step": TRAIN_STEPS, "loss": losses[-1], "train_time": train_time,
     "model_config": {
         "sequence_len": SEQ_LEN, "vocab_size": tokenizer.get_vocab_size(),
         "n_layer": pt_cfg.get("n_layer", 5), "n_head": pt_cfg.get("n_head", 4),
         "n_kv_head": pt_cfg.get("n_kv_head", 4), "n_embd": pt_cfg.get("n_embd", 256),
         "window_pattern": pt_cfg.get("window_pattern", "L"),
     }})
print(f"总耗时: {time.time()-t_start:.2f}s")
print("=" * 50)
print("本地 SFT 微调完成 ✅")
print("=" * 50)
