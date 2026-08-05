"""
实验1：Tokenizer 实验
用本地小说语料训练 5K 词表 BPE，观察 merge 过程
路径全部来自 config.yaml（相对本目录的相对路径），默认数据都在 runs-with-win/ 下。
"""
import sys, os, io, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
t_start = time.time()

# ========== 读取 config.yaml ==========
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 相对路径均以 config.yaml 所在目录（runs-with-win/）为基准
def rel(p):
    return os.path.normpath(os.path.join(HERE, p))

NOVEL_DIR = rel(cfg["noveltxt"])                 # 本地小说语料
TOKENIZER_SAVE_DIR = rel(cfg["tokenizer"])       # 分词模型输出目录

parser = argparse.ArgumentParser(description="实验1：训练 BPE 分词器")
parser.add_argument("--novel-dir", default=None, help="覆盖 config.yaml 的 noveltxt 路径")
parser.add_argument("--tokenizer-out", default=None, help="覆盖 config.yaml 的 tokenizer 输出路径")
parser.add_argument("--max-chars-512", type=int, default=None, help="阶段1 最大字符数")
parser.add_argument("--max-chars-5k", type=int, default=None, help="阶段2 最大字符数")
parser.add_argument("--vocab-512", type=int, default=None, help="阶段1 词表大小")
parser.add_argument("--vocab-5k", type=int, default=None, help="阶段2 词表大小")
args = parser.parse_args()

if args.novel_dir:
    NOVEL_DIR = rel(args.novel_dir)
if args.tokenizer_out:
    TOKENIZER_SAVE_DIR = rel(args.tokenizer_out)

tt_cfg = cfg.get("tokenizer_train", {})
MAX_CHARS_512 = args.max_chars_512 or tt_cfg.get("max_chars_512", 10_000_000)
MAX_CHARS_5K = args.max_chars_5k or tt_cfg.get("max_chars_5k", 100_000_000)
VOCAB_512 = args.vocab_512 or tt_cfg.get("vocab_512", 512)
VOCAB_5K = args.vocab_5k or tt_cfg.get("vocab_5k", 5120)

if not os.path.isdir(NOVEL_DIR):
    sys.exit(f"❌ 未找到小说语料目录: {NOVEL_DIR}（请确认 config.yaml 的 noveltxt 路径，或调整目录结构）")

print(f"配置: noveltxt={NOVEL_DIR}")
print(f"配置: tokenizer输出={TOKENIZER_SAVE_DIR}")
print(f"配置: 阶段1={MAX_CHARS_512}字符/词表{VOCAB_512}, 阶段2={MAX_CHARS_5K}字符/词表{VOCAB_5K}")

def text_iterator(max_chars=100_000_000):
    """逐文件产生文本，控制最大字符数"""
    total = 0
    for root, dirs, files in os.walk(NOVEL_DIR):
        for fname in files:
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='gbk', errors='ignore') as f:
                    text = f.read()
                if len(text) < 100:  # 跳过太短的文件
                    continue
                total += len(text)
                yield text
                if total >= max_chars:
                    return
            except:
                continue
    print(f"共读取 {total} 字符")

def show_encoding(tokenizer, text, label=""):
    ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([i]) for i in ids]
    print(f"\n{label}")
    print(f"原始: {text}")
    print(f"Token IDs ({len(ids)}个): {ids[:30]}{'...' if len(ids)>30 else ''}")
    print(f"Tokens: {tokens[:30]}{'...' if len(tokens)>30 else ''}")
    print(f"解码还原: {tokenizer.decode(ids)}")

# ========== 1. 训练小词表（512）快速验证 ==========
print("=" * 60)
print("阶段1：训练 512 词表（快速验证）")
print("=" * 60)
t0 = time.time()
it512 = text_iterator(max_chars=MAX_CHARS_512)
tokenizer = RustBPETokenizer.train_from_iterator(it512, VOCAB_512 + len(SPECIAL_TOKENS))
print(f"512 词表训练耗时: {time.time()-t0:.2f}s")
real_vocab = tokenizer.get_vocab_size() - len(SPECIAL_TOKENS)
print(f"目标词表大小: {VOCAB_512}，实际训练词表: {real_vocab}")

mergeable = tokenizer.enc._mergeable_ranks
sorted_merges = sorted(mergeable.items(), key=lambda x: x[1])

print(f"\n最早合并的前 30 个 pair（rank 越低越早合并）:")
for token_bytes, rank in sorted_merges[:30]:
    if rank >= 256:
        try:
            decoded = token_bytes.decode('utf-8', errors='replace')
        except:
            decoded = repr(token_bytes)
        print(f"  Rank {rank:>3}: {str(token_bytes):<30} → {decoded}")

# 统计中文 token
cn_tokens = []
for token_bytes, rank in sorted_merges:
    if rank >= 256:
        try:
            decoded = token_bytes.decode('utf-8')
            if any('\u4e00' <= c <= '\u9fff' for c in decoded):
                cn_tokens.append((rank, decoded, token_bytes))
        except:
            pass
print(f"\n中文相关 token 数量: {len(cn_tokens)}")
print(f"前 30 个:")
for rank, decoded, tb in cn_tokens[:30]:
    print(f"  Rank {rank:>3}: {decoded}")

show_encoding(tokenizer, "他缓缓站起身，望向远处的群山", "--- 512词表编码 ---")
show_encoding(tokenizer, "这是一个阳光明媚的下午", "--- 512词表编码 ---")

# ========== 2. 训练 5K 词表 ==========
print("\n" + "=" * 60)
print("阶段2：训练 5K 词表（正式实验）")
print("=" * 60)
t0 = time.time()
it5k = text_iterator(max_chars=MAX_CHARS_5K)
tokenizer_5k = RustBPETokenizer.train_from_iterator(it5k, VOCAB_5K + len(SPECIAL_TOKENS))
print(f"5K 词表训练耗时: {time.time()-t0:.2f}s")
real_vocab_5k = tokenizer_5k.get_vocab_size() - len(SPECIAL_TOKENS)
print(f"实际训练词表大小: {real_vocab_5k}")

mergeable_5k = tokenizer_5k.enc._mergeable_ranks
sorted_5k = sorted(mergeable_5k.items(), key=lambda x: x[1])

print(f"\n5K 词表 - 最早合并的前 20 个 pair:")
for token_bytes, rank in sorted_5k[:20]:
    if rank >= 256:
        try:
            decoded = token_bytes.decode('utf-8', errors='replace')
        except:
            decoded = repr(token_bytes)
        print(f"  Rank {rank:>4}: {str(token_bytes):<25} → {decoded}")

print(f"\n5K 词表 - 最高 rank 的前 10 个 token（最晚合并，最长词组）:")
for token_bytes, rank in sorted_5k[-10:]:
    try:
        decoded = token_bytes.decode('utf-8', errors='replace')
    except:
        decoded = repr(token_bytes)
    print(f"  Rank {rank:>4}: {decoded}")

cn_tokens_5k = []
for token_bytes, rank in sorted_5k:
    if rank >= 256:
        try:
            decoded = token_bytes.decode('utf-8')
            if any('\u4e00' <= c <= '\u9fff' for c in decoded):
                cn_tokens_5k.append((rank, decoded, token_bytes))
        except:
            pass
print(f"\n5K 词表 - 中文 tokens 数量: {len(cn_tokens_5k)}")
print(f"前 30 个:")
for rank, decoded, tb in cn_tokens_5k[:30]:
    print(f"  Rank {rank:>3}: {decoded}")

# 采样不同场景的编码
show_encoding(tokenizer_5k, "他缓缓站起身，望向远处的群山", "\n--- 5K词表编码 1（叙事）---")
show_encoding(tokenizer_5k, "这是一个阳光明媚的下午", "--- 5K词表编码 2（描写）---")
show_encoding(tokenizer_5k, "BPE是一种子词分词算法", "--- 5K词表编码 3（术语）---")
show_encoding(tokenizer_5k, "他说:\"你为什么要这样做？\"", "--- 5K词表编码 4（对话）---")

# 词表大小对比
print(f"\n词表大小对比:")
for text in ["他缓缓站起身", "这是一个", "注意力机制"]:
    ids512 = tokenizer.encode(text)
    ids5k = tokenizer_5k.encode(text)
    print(f"  '{text}': 512→{len(ids512)}个token, 5K→{len(ids5k)}个token")

# 保存 5K 词表供实验2使用
t0 = time.time()
tokenizer_5k.save(TOKENIZER_SAVE_DIR)
print(f"保存到 {TOKENIZER_SAVE_DIR} 耗时: {time.time()-t0:.2f}s")

print(f"\n总耗时: {time.time()-t_start:.2f}s")
