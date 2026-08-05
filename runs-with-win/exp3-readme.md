# 实验3：数据管线测试（SFT ChatML 格式化验证）

## 目的

在 SFT（Supervised Fine-Tuning）中，原始对话数据不能直接喂给模型。需要先转换成模型能理解的 **token ID 序列**，同时通过 **loss mask** 告诉模型："你只需要预测 assistant 的回答，user 的提问和特殊标记不需要学习"。

这个实验在不涉及 GPU 的前提下，逐行验证 nanochat 的数据格式化逻辑是否正确。它是上 GPU 跑 SFT 前的"必过安检"——数据管线的 bug 不会报错，只会无声地让模型学错东西。

## 什么是 ChatML

ChatML（Chat Markup Language）是一种**用特殊 token 标记对话角色**的格式。nanochat 定义了 9 个特殊 token：

| 特殊 token | Token ID | 用途 |
|------------|----------|------|
| `<\|bos\|>` | 5120 | 文档起始标记（Beginning of Sequence），每个对话以此开头 |
| `<\|user_start\|>` | 5121 | 用户消息开始 |
| `<\|user_end\|>` | 5122 | 用户消息结束 |
| `<\|assistant_start\|>` | 5123 | 助手消息开始 |
| `<\|assistant_end\|>` | 5124 | 助手消息结束 |
| `<\|python_start\|>` | 5125 | 助手调用 Python 工具开始 |
| `<\|python_end\|>` | 5126 | Python 工具调用结束 |
| `<\|output_start\|>` | 5127 | Python 执行结果开始 |
| `<\|output_end\|>` | 5128 | Python 执行结果结束 |

这些 token 在训练 BPE 时被预留（不参与 merge），拥有独立的 token ID（排在普通 token 之后）。一条对话经过 `render_conversation()` 处理后变成：

```
<|bos|> <|user_start|> 用户说了什么 <|user_end|> <|assistant_start|> 助手回答了 <|assistant_end|>
```

## 什么是 dict 格式对话

nanochat 用 **Python dict** 表示一条对话（conversation），这是所有 Task 数据集的统一接口（例如 `SmolTalk`、`MMLU`、`GSM8K` 都返回这种格式）：

```python
{
    "messages": [
        {"role": "system", "content": "系统提示词（可选）"},
        {"role": "user", "content": "用户的提问"},
        {"role": "assistant", "content": "助手的回答"},
        # 还可以继续多轮...
    ]
}
```

**三种 role**：

- **system**（可选）— 系统提示词，会被合并到第一条 user 消息中
- **user** — 用户输入，**不参与 loss 计算**（mask=0）
- **assistant** — 助手回复，**参与 loss 计算**（mask=1）

**两种 content 格式**：

- **简单字符串**：纯文本回答
- **多部分列表**（含工具调用时）：
  ```python
  "content": [
      {"type": "text", "text": "我来算一下："},
      {"type": "python", "text": "print(123 * 456)"},
      {"type": "python_output", "text": "56088"},
      {"type": "text", "text": "结果是 56088。"},
  ]
  ```
  - `type="text"` — 普通文本，mask=1
  - `type="python"` — Python 代码，mask=1（模型需要学会调用工具）
  - `type="python_output"` — 工具执行结果，mask=0（测试时来自外部，不训练）

## 什么是 loss mask

nanochat 的 SFT 只对 **assistant 的回答** 计算损失。user 的提问、特殊 token、工具执行结果都不应该被学习。实现方式：

```
token 序列: [BOS] [U_start] 你几岁 [U_end] [A_start] 我3岁 [A_end]
mask:        0      0       0     0       1        1     1
```

- mask=1 的 token 参与交叉熵损失计算
- mask=0 的 token 在 `CrossEntropyLoss(ignore_index=-1)` 中被忽略

这样模型只学习"如何像 assistant 一样回答"，而不会学习"如何提问"或"特殊 token 的含义"。

## 宏观流程

```
手写 6 组 dict 格式对话（覆盖所有 case）
    → tokenizer.render_conversation(conversation)
        ├─ 合并系统消息 → 校验 role 交替顺序
        ├─ 插入 ChatML 特殊 token
        ├─ 编码文本内容 → 按 role 设置 mask
        └─ 截断到 max_tokens → 返回 (ids, mask)
    → 逐 token 断言验证格式 & mask
    → visualize_tokenization() 输出彩色可视化
    → 模拟 DataLoader best-fit padding 打包
```

## 输入

| 输入 | 说明 |
|------|------|
| `tokenizer_5k/` | exp1 保存的 5K BPE 词表（含 9 个 ChatML 特殊 token） |
| 6 组手写 dict 对话 | 覆盖全部 case（见下文） |

### 测试用例详解

**Test 1: 简单对话** — 最基本的 user → assistant 格式，验证 BOS、mask 正确

```
输入: user "什么是注意力机制？" → assistant "注意力机制（Attention）..."
输出: BOS | U_start | 什么是注意力机制？ | U_end | A_start | 注意力机制... | A_end
mask: 0     0         0 0 0 0 0       0       0         1 1 1...         1
```

**Test 2: 系统消息合并** — system + user + assistant 三消息，验证 system 合并

```
输入: system "你是一个有帮助的助手。" → user "今天天气怎么样？" → assistant "..."
处理: system 内容拼接到 user 内容前 → "你是一个有帮助的助手。\n\n今天天气怎么样？"
输出: BOS | U_start | 你是一个有帮助的助手。\n\n今天天气怎么样？ | U_end | A_start | ... | A_end
```

**Test 3: 多轮对话** — 多组交替，验证每轮格式独立

```
输入: user Q1 → assistant A1 → user Q2 → assistant A2
输出: BOS | U_start Q1 U_end | A_start A1 A_end | U_start Q2 U_end | A_start A2 A_end
```

**Test 4: Python 工具调用** — assistant 内容含 `type=python`，验证 `<|python_start/end|>` 包裹

```
输出: ... A_start | 我来... | python_start | def fib... | python_end | A_end
mask: ... 0         1 1...   1             1 1...       1            1
```

**Test 5: Python 输出** — assistant 内容含 `type=python_output`，验证 mask=0

```
输出: ... A_start | python_start | print... | python_end | output_start | 56088 | output_end | 结果是... | A_end
mask: ... 0         1             1 1...     1            0             0 0...  0            1 1...      1
```

**Test 6: 长文本截断** — 超过 max_tokens=50 时正确截断

```
处理: ids[:50], mask[:50]
验证: 截断结果 == 完整序列的前 50 个 token
```

## 输出

| 输出 | 说明 |
|------|------|
| 每个测试的 token 序列表 | 位置、Token ID、mask、解码内容的逐行对照表 |
| 特殊 token 统计 | 各 ChatML token 出现的次数（验证轮次数正确） |
| 彩色可视化 | 绿色 = 参与训练，红色 = 不参与训练 |
| 模拟 DataLoader 结果 | best-fit 打包 + BOS 填充后的 inputs/targets 张量 |

### 核心调用关系

```
exp3_data_pipeline.py
    └─ RustBPETokenizer.from_directory(tokenizer_5k)    ← 加载 exp1 词表
    └─ tokenizer.render_conversation(conv_dict)          ← ChatML 格式化
        ├─ tokenizer.get_bos_token_id()                  ← BOS token
        ├─ tokenizer.encode_special("<|user_start|>")    ← 特殊 token ID
        ├─ tokenizer.encode(text)                        ← 文本编码
        └─ add_tokens(ids, mask_val)                     ← 构建 ids + mask
    └─ tokenizer.visualize_tokenization(ids, mask)       ← 彩色输出
    └─ 手动实现 best-fit padding                         ← 模拟 DataLoader
```

## 与 chat_sft.py 的关系

`chat_sft.py` 生产的 SFT 训练脚本中，数据管线核心是 `sft_data_generator_bos_bestfit()` 生成器。本实验完整模拟了它的两个关键步骤：

### 步骤 1: render_conversation（本实验 Tests 1-6）

```python
# chat_sft.py:208
ids, mask = tokenizer.render_conversation(conversation)
```

把 dict 格式对话 → (token_ids, loss_mask)。本实验验证了这一步在 6 种 case 下的正确性。

### 步骤 2: best-fit packing（本实验最终模拟）

```python
# chat_sft.py:220-253 — 核心逻辑如下：
# - 每行从 BOS 开始
# - 从 buffer 中选"最长且能完整放入"的对话放入
# - 放不下时用 BOS 填充（不裁剪，防止丢 token）
# - mask 填充位置为 0（不参与 loss）
# - targets 中填充位置设为 -1（CrossEntropy ignore_index）
```

模型输入 `inputs` 张量形状为 `(batch_size, seq_len)`，`targets` 为 inputs 右移 1 位。填充位置在 `targets` 中设为 `-1` 被 loss 函数忽略。

### 生产代码 vs 本实验

| 对比项 | `chat_sft.py`（生产） | `exp3_data_pipeline.py`（教学） |
|--------|----------------------|-------------------------------|
| **数据源** | `TaskMixture([SmolTalk, MMLU, GSM8K])` ~800K 行 | 手写 6 组 dict 对话 |
| **Token 上限** | 2048 | 50/2048 可选 |
| **执行平台** | GPU 分布式 | CPU 单机 |
| **验证方式** | 训练中观察 loss 下降 | 逐 token 断言 + 可视化 |
| **打包策略** | 多行最佳适配（best-fit）+ BOS 填充 | 同上，小规模模拟 |
| **目的** | 微调模型 | 理解格式原理 |

## 与 exp1/exp2 的关系

```
exp1: 训练 BPE tokenizer（512 + 5K 词表）
  ↓ 产物: tokenizer_5k/（含 ChatML 特殊 token）
exp2: 加载 tokenizer + 小说语料，训练小 GPT
  ↓ 只用 tokenizer，不用模型
exp3: 加载 tokenizer，验证对话 → ChatML 格式
```

三个实验按"数据流"递进：**词表 → 模型训练 → 数据格式**。ex3 不需要 ex2 的产物。

## 前置依赖

```powershell
cd nanochat
.\.venv\Scripts\python.exe runs-with-win\exp1_tokenizer_bpe.py
```

## 如何运行

本目录版本已改为读取 `config.yaml`（相对 `runs-with-win/` 的相对路径）。两个脚本：

```powershell
cd nanochat
# 1) 验证 ChatML 数据管线（6 个用例 + best-fit 打包）
.\.venv\Scripts\python.exe runs-with-win\exp3_data_pipeline.py

# 2) 生成本地 SFT 样本数据 + TaskMixture + DataLoader 模拟（产物 → runs-with-win/sample_data/）
.\.venv\Scripts\python.exe runs-with-win\exp3_sample_sft_data.py
```

可选参数（`exp3_sample_sft_data.py`）：`--sample-size N`、`--sample-dir <路径>`、`--ckpt-dir <路径>` 等。

## 预期结果

```
所有 6 项测试 ✅ 通过

Test 1: BOS 开头, user mask=0, assistant mask=1
Test 2: 系统消息成功合并到第一条 user
Test 3: 2轮对话, 各出现 2 次 user_start/end + assistant_start/end
Test 4: python_start~python_end 内 mask=1
Test 5: output_start~output_end 内 mask=0
Test 6: 截断到 50 token 正确

模拟 DataLoader: best-fit 打包 + BOS 填充正确
总耗时: ~5 秒
```

## 如果数据管线出 bug 会怎样？

数据管线的 bug 不会引起 Python 报错或训练崩溃，而是**无声地让模型学错东西**。常见问题：

| Bug 现象 | 后果 |
|----------|------|
| assistant 的 mask 误设为 0 | 模型学不到如何回答 |
| user 的 mask 误设为 1 | 模型学会"提问"而非"回答" |
| 系统消息未被合并 | 多出意外轮次，角色交替校验失败 |
| 填充位置的 BOS 未被 mask 掉 | 模型学习预测无意义的 BOS |
| 工具输出 mask=1 | 模型以为输出是自己生成的，推理时产生幻觉 |
