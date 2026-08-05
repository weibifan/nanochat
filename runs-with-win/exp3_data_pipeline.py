"""
实验3：数据管线测试
本地验证 SFT 数据格式化是否正确输出 ChatML token 序列
路径来自 config.yaml（相对本目录）
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
from nanochat.tokenizer import RustBPETokenizer

t_start = time.time()

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.yaml"), "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
TOKENIZER_DIR = os.path.normpath(os.path.join(HERE, cfg["tokenizer"]))
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
V = tokenizer.get_vocab_size()
print(f"加载 tokenizer，词表大小: {V}")
print(f"特殊 tokens: {tokenizer.get_special_tokens()}")
print()

# ========== 辅助函数 ==========
def describe_conversation(conv, label=""):
    messages = conv["messages"]
    print(f"\n{'='*60}")
    print(f"{label} ({len(messages)} 条消息)")
    print(f"{'='*60}")
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            preview = content[:60].replace('\n', '\\n')
            print(f"  [{role}] \"{preview}...\"")
        elif isinstance(content, list):
            print(f"  [{role}] (多部分消息, {len(content)} 部分)")
            for part in content:
                pt = part["type"]
                ptxt = part["text"][:40].replace('\n', '\\n')
                print(f"    - type={pt}: \"{ptxt}...\"")
    print()

def analyze_render(conv, max_tokens=2048, show_detail=True):
    ids, mask = tokenizer.render_conversation(conv, max_tokens=max_tokens)

    total_tokens = len(ids)
    train_tokens = sum(mask)
    non_train_tokens = total_tokens - train_tokens

    print(f"  Token 总数: {total_tokens}")
    print(f"  参与训练 (assistant): {train_tokens} ({100*train_tokens/max(total_tokens,1):.1f}%)")
    print(f"  不参与训练: {non_train_tokens} ({100*non_train_tokens/max(total_tokens,1):.1f}%)")

    special_ids = {
        "<|bos|>": tokenizer.encode_special("<|bos|>"),
        "<|user_start|>": tokenizer.encode_special("<|user_start|>"),
        "<|user_end|>": tokenizer.encode_special("<|user_end|>"),
        "<|assistant_start|>": tokenizer.encode_special("<|assistant_start|>"),
        "<|assistant_end|>": tokenizer.encode_special("<|assistant_end|>"),
        "<|python_start|>": tokenizer.encode_special("<|python_start|>"),
        "<|python_end|>": tokenizer.encode_special("<|python_end|>"),
        "<|output_start|>": tokenizer.encode_special("<|output_start|>"),
        "<|output_end|>": tokenizer.encode_special("<|output_end|>"),
    }
    inv_special = {v: k for k, v in special_ids.items()}

    special_count = {}
    for tid in ids:
        if tid in inv_special:
            name = inv_special[tid]
            special_count[name] = special_count.get(name, 0) + 1

    print(f"\n  特殊 tokens 统计:")
    for name in ["<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
                 "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]:
        cnt = special_count.get(name, 0)
        print(f"    {name:<20}: {cnt}")

    if show_detail:
        print(f"\n  逐 token 序列 (前 40 个):")
        print(f"  {'位置':>4} | {'Token ID':>7} | {'mask':>4} | {'内容':<40}")
        print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*4}-+-{'-'*40}")
        for i in range(min(40, total_tokens)):
            tid = ids[i]
            m = mask[i]
            if tid in inv_special:
                display = inv_special[tid]
            else:
                display = tokenizer.decode([tid])
                display = display.replace('\n', '\\n')
                if len(display) > 40:
                    display = display[:37] + "..."
            print(f"  {i:>4} | {tid:>7} | {m:>4} | {display:<40}")

        if total_tokens > 40:
            print(f"  ... (省略 {total_tokens - 40} 个 token)")

    print(f"\n  可视化 (绿色=train, 红色=no_train):")
    vis = tokenizer.visualize_tokenization(ids, mask, with_token_id=False)
    print(f"  {vis}")
    print()

    return ids, mask

# ========== 测试用例 ==========

# Test 1: 简单对话（无系统消息）
conv1 = {
    "messages": [
        {"role": "user", "content": "什么是注意力机制？"},
        {"role": "assistant", "content": "注意力机制（Attention）是一种让模型关注输入中重要部分的技术。"},
    ]
}
describe_conversation(conv1, "Test 1: 简单对话（无系统消息）")
ids1, mask1 = analyze_render(conv1)
assert ids1[0] == tokenizer.get_bos_token_id(), "第一个 token 必须是 BOS"
assert mask1[0] == 0, "BOS 的 mask 必须为 0"
print("  ✅ BOS 位置和 mask 正确")
print("  ✅ 格式验证通过")
print()

# Test 2: 带系统消息的对话
conv2 = {
    "messages": [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "今天天气怎么样？"},
        {"role": "assistant", "content": "抱歉，我无法获取实时天气信息。"},
    ]
}
describe_conversation(conv2, "Test 2: 带系统消息（应合并到 user 消息）")
ids2, mask2 = analyze_render(conv2)
bos_id = tokenizer.get_bos_token_id()
user_start_id = tokenizer.encode_special("<|user_start|>")
user_end_id = tokenizer.encode_special("<|user_end|>")
ast_start_id = tokenizer.encode_special("<|assistant_start|>")
ast_end_id = tokenizer.encode_special("<|assistant_end|>")

assert ids2[0] == bos_id, "第一个 token 必须是 BOS"
assert user_start_id in ids2, "必须包含 user_start"
assert ast_start_id in ids2, "必须包含 assistant_start"
assert ast_end_id in ids2, "必须包含 assistant_end"
has_system_text = "有帮助的助手" in tokenizer.decode(ids2)
assert has_system_text, "系统消息内容应被合并到 user 消息中"
print("  ✅ 系统消息正确合并到 user 消息")
print()

# Test 3: 多轮对话
conv3 = {
    "messages": [
        {"role": "user", "content": "1+1等于几？"},
        {"role": "assistant", "content": "1+1=2"},
        {"role": "user", "content": "那2+2呢？"},
        {"role": "assistant", "content": "2+2=4"},
    ]
}
describe_conversation(conv3, "Test 3: 多轮对话")
ids3, mask3 = analyze_render(conv3, show_detail=False)

num_ast_start = sum(1 for tid in ids3 if tid == ast_start_id)
num_ast_end = sum(1 for tid in ids3 if tid == ast_end_id)
num_user_start = sum(1 for tid in ids3 if tid == user_start_id)
num_user_end = sum(1 for tid in ids3 if tid == user_end_id)
assert num_ast_start == 2, f"期望 2 个 assistant_start, 实际 {num_ast_start}"
assert num_ast_end == 2, f"期望 2 个 assistant_end, 实际 {num_ast_end}"
assert num_user_start == 2, f"期望 2 个 user_start, 实际 {num_user_start}"
assert num_user_end == 2, f"期望 2 个 user_end, 实际 {num_user_end}"
print("  ✅ 多轮对话格式正确（2 user + 2 assistant 轮次）")
print()

# Test 4: Assistant 内容含代码（多部分消息）
conv4 = {
    "messages": [
        {"role": "user", "content": "写一个 Python 函数计算斐波那契数列"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "以下是斐波那契数列的 Python 实现："},
            {"type": "python", "text": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"},
        ]},
    ]
}
describe_conversation(conv4, "Test 4: Assistant 含 python 工具调用")
ids4, mask4 = analyze_render(conv4)
ps_id = tokenizer.encode_special("<|python_start|>")
pe_id = tokenizer.encode_special("<|python_end|>")

assert ps_id in ids4, "必须包含 python_start"
assert pe_id in ids4, "必须包含 python_end"
# python_start 到 python_end 之间的 mask 应为 1（参与训练）
ps_idx = ids4.index(ps_id)
pe_idx = ids4.index(pe_id)
for i in range(ps_idx, pe_idx + 1):
    assert mask4[i] == 1, f"python 代码区域 mask 必须为 1, 位置 {i} 的 mask={mask4[i]}"
print("  ✅ python 工具调用格式和 mask 正确")
print()

# Test 5: python 输出（不应参与训练）
conv5 = {
    "messages": [
        {"role": "user", "content": "计算 12345 * 6789"},
        {"role": "assistant", "content": [
            {"type": "python", "text": "print(12345 * 6789)"},
            {"type": "python_output", "text": "83810205"},
            {"type": "text", "text": "计算结果为 83,810,205。"},
        ]},
    ]
}
describe_conversation(conv5, "Test 5: 含 python 输出（mask 应为 0）")
ids5, mask5 = analyze_render(conv5)
os_id = tokenizer.encode_special("<|output_start|>")
oe_id = tokenizer.encode_special("<|output_end|>")

assert os_id in ids5, "必须包含 output_start"
assert oe_id in ids5, "必须包含 output_end"
os_idx = ids5.index(os_id)
oe_idx = ids5.index(oe_id)
for i in range(os_idx, oe_idx + 1):
    assert mask5[i] == 0, f"python_output 区域 mask 必须为 0, 位置 {i} 的 mask={mask5[i]}"
print("  ✅ python_output mask=0 正确（输出不参与训练）")
print()

# Test 6: truncation 测试（max_tokens=50）
conv6 = {
    "messages": [
        {"role": "user", "content": "请详细介绍一下 Transformer 架构的原理和历史，包括 Attention 机制、Positional Encoding、Layer Normalization 等关键技术。"},
        {"role": "assistant", "content": "Transformer 是一种基于自注意力机制的神经网络架构，由 Vaswani 等人在 2017 年提出。它抛弃了传统的循环和卷积结构，完全依赖于注意力机制来捕捉序列中的依赖关系。"},
    ]
}
describe_conversation(conv6, "Test 6: truncation 测试（max_tokens=50）")
ids6_full, mask6_full = tokenizer.render_conversation(conv6, max_tokens=2048)
ids6_trunc, mask6_trunc = tokenizer.render_conversation(conv6, max_tokens=50)
print(f"  完整长度: {len(ids6_full)}")
print(f"  截断后: {len(ids6_trunc)}")
assert len(ids6_trunc) == 50, f"截断后应为 50 个 token, 实际 {len(ids6_trunc)}"
assert ids6_trunc == ids6_full[:50], "截断内容应与完整序列的前 50 个 token 一致"
print("  ✅ 截断功能正确")
print()

# ========== 模拟完整训练数据管线 ==========
print("="*60)
print("模拟 SFT 训练数据管线（DataLoader 核心逻辑）")
print("="*60)

test_convs = [conv1, conv2, conv3]
SEQ_LEN = 64
BATCH_SIZE = 2
bos_token = tokenizer.get_bos_token_id()

print(f"\n模拟参数: seq_len={SEQ_LEN}, batch_size={BATCH_SIZE}")

conv_buffer = []
for conv in test_convs:
    ids, mask = tokenizer.render_conversation(conv, max_tokens=SEQ_LEN)
    conv_buffer.append((ids, mask))
print(f"  加载 {len(conv_buffer)} 条对话到 buffer")

row_capacity = SEQ_LEN + 1
batches = []
for _ in range(BATCH_SIZE):
    row = []
    mask_row = []
    padded = False
    while len(row) < row_capacity:
        remaining = row_capacity - len(row)
        best_idx = -1
        best_len = 0
        for i, (conv, _) in enumerate(conv_buffer):
            conv_len = len(conv)
            if conv_len <= remaining and conv_len > best_len:
                best_idx = i
                best_len = conv_len
        if best_idx >= 0:
            conv, conv_mask = conv_buffer.pop(best_idx)
            row.extend(conv)
            mask_row.extend(conv_mask)
        else:
            content_len = len(row)
            row.extend([bos_token] * remaining)
            mask_row.extend([0] * remaining)
            padded = True
            break
    if padded:
        print(f"  Row {len(batches)}: {content_len} 实际内容 + {remaining} 填充 = {row_capacity}")
    else:
        print(f"  Row {len(batches)}: {row_capacity} 全部内容 (无填充)")
        content_len = row_capacity
    batches.append((row[:row_capacity], mask_row[:row_capacity], content_len))

print()
print("构建 inputs/targets 张量 (模拟训练时的处理):")
import torch
for i, (row, mrow, content_len) in enumerate(batches):
    batch_tensor = torch.tensor(row, dtype=torch.long)
    inputs = batch_tensor[:-1]
    targets = batch_tensor[1:]
    mask_t = torch.tensor(mrow[1:], dtype=torch.int8)
    targets[mask_t == 0] = -1
    if content_len < row_capacity:
        targets[content_len - 1:] = -1

    train_tokens = (targets != -1).sum().item()
    total_tokens = targets.numel()
    print(f"  Batch {i}: inputs={inputs.shape}, targets={targets.shape}")
    print(f"    训练 token: {train_tokens}/{total_tokens} ({100*train_tokens/total_tokens:.1f}%)")
    print(f"    填充/忽略 token: {total_tokens - train_tokens}/{total_tokens}")

# ========== 验证汇总 ==========
print()
print("="*60)
print("实验3 全部通过 ✅")
print("="*60)
print(f"总耗时: {time.time()-t_start:.2f}s")
