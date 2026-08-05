"""
实验2：Toy Forward Pass
用 noveltxt 语料训练极小模型(2层/128维)，在 CPU 上跑完整训练流程
路径与参数全部来自 config.yaml（相对本目录），产物输出到 runs-with-win/checkpoint/
"""
import sys, os, io, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
torch._dynamo.config.disable = True  # 系统无 MSVC，禁用 torch.compile
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, get_base_dir, COMPUTE_DTYPE
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS

# ========== 读取 config.yaml ==========
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.yaml"), "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
def rel(p):
    return os.path.normpath(os.path.join(HERE, p))

NOVEL_DIR = rel(cfg["noveltxt"])
TOKENIZER_DIR = rel(cfg["tokenizer"])

pt_cfg = cfg.get("pretrain", {})
CKPT_DIR = rel(pt_cfg.get("checkpoint_dir", "checkpoint"))
MAX_CHARS = pt_cfg.get("max_chars", 2_000_000)
SEQ_LEN = pt_cfg.get("seq_len", 128)
B = pt_cfg.get("batch_size", 8)
TRAIN_STEPS = pt_cfg.get("steps", 1000)
N_LAYER = pt_cfg.get("n_layer", 5)
N_EMBD = pt_cfg.get("n_embd", 256)
N_HEAD = pt_cfg.get("n_head", 4)
N_KV_HEAD = pt_cfg.get("n_kv_head", 4)
WINDOW_PATTERN = pt_cfg.get("window_pattern", "L")
MATRIX_LR = pt_cfg.get("matrix_lr", 0.1)

parser = argparse.ArgumentParser(description="实验2：预训练小 GPT")
parser.add_argument("--novel-dir", default=None)
parser.add_argument("--tokenizer", default=None)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--seq-len", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=None)
args = parser.parse_args()
if args.novel_dir: NOVEL_DIR = rel(args.novel_dir)
if args.tokenizer: TOKENIZER_DIR = rel(args.tokenizer)
if args.steps: TRAIN_STEPS = args.steps
if args.seq_len: SEQ_LEN = args.seq_len
if args.batch_size: B = args.batch_size

if not os.path.isdir(NOVEL_DIR):
    sys.exit(f"❌ 未找到小说语料目录: {NOVEL_DIR}")
if not os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.pkl")):
    sys.exit(f"❌ 未找到分词模型: {TOKENIZER_DIR}/tokenizer.pkl（请先运行 exp1）")

# ========== 0. 加载 tokenizer ==========
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
print(f"加载 tokenizer: 词表={tokenizer.get_vocab_size()}")

# ========== 1. 初始化 ==========
t_start = time.time()
_, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init("cpu")
print(f"设备: {device}")
print(f"计算精度: {COMPUTE_DTYPE}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"配置: novel={NOVEL_DIR}")
print(f"配置: tokenizer={TOKENIZER_DIR}")
print(f"配置: checkpoint输出={CKPT_DIR}, 步数={TRAIN_STEPS}")
print()

# ========== 2. 准备语料 ==========
def load_novels(max_chars=2_000_000):
    texts = []
    total = 0
    for root, dirs, files in os.walk(NOVEL_DIR):
        for fname in files:
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='gbk', errors='ignore') as f:
                    text = f.read()
                if len(text) < 200:
                    continue
                texts.append(text)
                total += len(text)
                if total >= max_chars:
                    break
            except:
                continue
        if total >= max_chars:
            break
    print(f"读取 {len(texts)} 篇小说，共 {total:,} 字符")
    return texts

corpus = load_novels(MAX_CHARS)

# 编码示例
sample = corpus[0][:200]
ids = tokenizer.encode(sample)
print(f"编码示例: \"{sample[:60]}...\" → {len(ids)} 个 token")
print()

# ========== 4. 构建模型 ==========
config = GPTConfig(
    sequence_len=SEQ_LEN,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_kv_head=N_KV_HEAD,
    n_embd=N_EMBD,
    window_pattern=WINDOW_PATTERN,
)
print(f"模型配置:")
print(f"  n_layer={config.n_layer}, n_embd={config.n_embd}, n_head={config.n_head}")
print(f"  vocab_size={config.vocab_size}, sequence_len={config.sequence_len}")
print()

t0 = time.time()
print("构建模型...")
with torch.device("meta"):
    model_meta = GPT(config)
model = model_meta.to_empty(device=device)
model.init_weights()
total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"建模耗时: {time.time()-t0:.2f}s")
print()

# ========== 5. 准备训练数据 ==========
def make_batch(tokenizer, corpus, B, T):
    """从语料中取一批真实文本，编码成 (idx, targets)"""
    ids = []
    target_len = B * (T + 1)
    while len(ids) < target_len:
        text = corpus[torch.randint(0, len(corpus), (1,)).item()]
        text = text[:5000]
        token_ids = tokenizer.encode(text)
        ids.extend(token_ids)
    ids = ids[:target_len]
    x = torch.tensor(ids[:B*T], dtype=torch.long).view(B, T)
    y = torch.tensor(ids[1:B*T+1], dtype=torch.long).view(B, T)
    return x.to(device), y.to(device)

T = SEQ_LEN

x, y = make_batch(tokenizer, corpus, B, T)
print(f"训练数据形状: x={x.shape}, y={y.shape}")

# 展示真实数据
txt_sample = tokenizer.decode(x[0, :10].tolist())
print(f"输入文本: \"{txt_sample}\"")
print()

# ========== 6. 前向 + 反向 ==========
print("前向传播...")
loss = model(x, targets=y)
print(f"  Loss: {loss.item():.4f}")

print("反向传播...")
loss.backward()

grad_norms = []
for name, param in model.named_parameters():
    if param.grad is not None:
        gn = param.grad.norm().item()
        grad_norms.append((name, gn, param.numel()))

grad_norms.sort(key=lambda x: x[1], reverse=True)
print(f"  有梯度的参数: {len(grad_norms)}")
print(f"  梯度总范数: {sum(g[1]**2 for g in grad_norms)**0.5:.4f}")
print(f"  梯度 Top 3:")
for name, gn, numel in grad_norms[:3]:
    print(f"    {name:<30} norm={gn:.6f}")
print()

# ========== 7. 训练 ==========
optimizer = model.setup_optimizer(matrix_lr=MATRIX_LR)
print(f"训练 {TRAIN_STEPS} 步（观察 loss 趋势）:")
losses = []
t0 = time.time()
for step in range(TRAIN_STEPS):
    x, y = make_batch(tokenizer, corpus, B, T)
    loss = model(x, targets=y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0 or step == TRAIN_STEPS - 1:
        print(f"  step {step:>3}: loss={loss.item():.4f}")
train_time = time.time() - t0

print(f"  首步 loss: {losses[0]:.4f}, 末步 loss: {losses[-1]:.4f}")
trend = "✅ 下降" if losses[-1] < losses[0] else "❌ 未下降"
print(f"  趋势: {trend}")
print(f"  训练耗时: {train_time:.2f}s ({30/train_time:.1f} step/s)")
print()

# ========== 8. 生成文本（推理验证）==========
print("推理生成（验证模型能产出有效 token）:")
prompt_ids = tokenizer.encode(corpus[0][:30])
prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
with torch.no_grad():
    logits = model(prompt)
    next_logits = logits[0, -1, :]
    top5 = torch.topk(next_logits, 5)
    print(f"  提示: \"{tokenizer.decode(prompt_ids)}\"")
    print(f"  预测下个 token Top 5:")
    for i in range(5):
        tid = top5.indices[i].item()
        t = tokenizer.decode([tid])
        p = torch.softmax(next_logits, dim=0)[tid].item()
        print(f"    [{tid:>3}] \"{t}\"  prob={p:.2%}")
print()

# ========== 9. 参数统计 ==========
param_counts = model.num_scaling_params()
print(f"参数统计:")
for k, v in param_counts.items():
    print(f"  {k:<25}: {v:>8,}")

num_matmul = model.num_matmul_params()
flops = model.estimate_flops()
print(f"  matmul 参数           : {num_matmul:>8,}")
print(f"  预估 FLOPs/token      : {flops:,}")
print()

# ========== 10. 保存检查点 ==========
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"保存检查点到 {CKPT_DIR}...")
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
save_checkpoint(
    CKPT_DIR, step=TRAIN_STEPS,
    model_data=model.state_dict(),
    optimizer_data=optimizer.state_dict(),
    meta_data={"loss": losses[-1]},
)
print("  完成")

print("\n加载检查点验证...")
model2 = GPT(config)
model2.to_empty(device=device)
model_data_loaded, _, meta_loaded = load_checkpoint(CKPT_DIR, step=TRAIN_STEPS, device=device)
model2.load_state_dict(model_data_loaded)
print(f"  加载 loss={meta_loaded['loss']:.4f}")
with torch.no_grad():
    loss2 = model2(x, targets=y)
print(f"  加载后 forward loss={loss2.item():.4f}")
print()

total_time = time.time() - t_start
print(f"总耗时: {total_time:.2f}s")
print("=" * 50)
print("实验2 全部通过 ✅")
print("=" * 50)
