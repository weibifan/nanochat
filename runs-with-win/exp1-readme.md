# 实验1：BPE Tokenizer 训练实验

> 本文件是从 `local_ex1/README.md` 拷贝并适配到 `runs-with-win/` 的版本：路径改为读取 `config.yaml`，运行方式见文末「如何运行」。

## 目的

用本地小说语料训练一个 BPE 分词器，直观理解 BPE 算法"如何从零学到一套编码规则"——哪些字节对先被合并、中文在 byte-level 下如何被处理、词表大小对压缩率的影响。

## 宏观流程

```
noveltxt/ 目录（几千本小说）
    → 遍历所有 .txt，逐文件 yield 文本内容（text_iterator）
    → RustBPETokenizer.train_from_iterator() 执行 BPE 训练
        → rustbpe 引擎统计相邻字节/字符频率
        → 反复合并最频繁的 pair，直到词表满
        → 包装成 tiktoken.Encoding 返回
    → 分析 mergeable_ranks 观察合并结果
        → 哪些 byte pair 合并最早（rank 最低）
        → 中文相关 token 有哪些
    → 用训练好的 tokenizer 编码示例文本，看压缩效果
```

## 输入

| 输入 | 说明 |
|------|------|
| `noveltxt/` 目录 | 内含 `.txt` 小说文件，遍历所有文本作为训练语料 |
| `max_chars`（代码中硬编码） | 控制读取量：512 词表用 1000 万字，5K 词表用 1 亿字 |

## 输出

打印到控制台，分两个阶段。**5K 词表训练完成后自动保存到 `tokenizer_5k/` 目录**，供实验2使用。

**阶段1（512 词表快速验证）**
- 最早合并的前 30 个 byte pair
- 中文相关 token 列表
- 示例文本编码结果

**阶段2（5K 词表正式实验）**
- 最早合并的前 20 个 pair
- 最晚合并（rank 最高）的前 10 个 token（最长词组）
- 中文相关 token 列表
- 多场景编码示例（叙事、描写、术语、对话）
- 词表大小对比（同一文本两种词表的 token 数量）

## 核心调用关系

```
exp1_tokenizer_bpe.py
    └─ RustBPETokenizer.train_from_iterator(text_iter, vocab_size)
        ├─ rustbpe.Tokenizer.train_from_iterator()    ← BPE 训练（Rust）
        └─ tiktoken.Encoding(...)                     ← 包装成推理格式
    └─ tokenizer.enc._mergeable_ranks                 ← 读取合并结果
    └─ tokenizer.encode(text)                         ← 编码测试
    └─ tokenizer.decode(ids)                          ← 解码验证
```

## 与 tok_train.py 的关系

| 对比项 | `tok_train.py`（生产） | `exp1_tokenizer_bpe.py`（toy） |
|--------|-----------------------|-----------------------------|
| **数据源** | `parquets_iter_batched()` 读 ClimbMix-400B 数据集 | 本地 `noveltxt/` 目录的 txt 小说 |
| **训练量** | 默认 20 亿字符 | 512 词表用 1000 万，5K 用 1 亿 |
| **词表大小** | 默认 32768 | 512 和 5120 |
| **输出** | 存 `tokenizer.pkl` + `token_bytes.pt` 供后续训练用 | 直接打印分析结果到控制台 |
| **完整性检查** | encode→decode 无损验证 + 生成 token_bytes | 各种场景的编码示例 + 词表对比 |
| **目的** | **生产**：训练出可用的 tokenizer，供模型训练使用 | **教学**：观察 BPE 的合并过程和中文字符的处理方式 |

**一句话**：`tok_train.py` 是真正干活的——训练出来的 tokenizer 要喂给模型训练；`exp1.py` 是观察学习的——训练结束后将 5K 词表保存到 `tokenizer_5k/`，供实验2加载使用。

## 如何运行

本目录版本已改为**读取 `config.yaml` 配置**（所有路径为相对 `runs-with-win/` 的相对路径），并支持命令行参数覆盖：

```powershell
# 在 nanochat 仓库根目录下
.\.venv\Scripts\python.exe runs-with-win\exp1_tokenizer_bpe.py

# 或用参数覆盖配置（例如小规模快速验证）
.\.venv\Scripts\python.exe runs-with-win\exp1_tokenizer_bpe.py `
    --max-chars-512 30000 --max-chars-5k 30000 --vocab-512 300 --vocab-5k 400
```

可用参数：

| 参数 | 作用 |
|------|------|
| `--novel-dir` | 覆盖 `config.yaml` 的 `noveltxt` 路径 |
| `--tokenizer-out` | 覆盖 `config.yaml` 的 `tokenizer` 输出路径 |
| `--max-chars-512` / `--max-chars-5k` | 两阶段最大字符数 |
| `--vocab-512` / `--vocab-5k` | 两阶段词表大小 |

## 预期能回答的问题

1. BPE 第一步合并了什么？（中英文分别）
2. 中文在 byte-level BPE 下是什么形态？
3. 词表从 512 扩大到 5K，对中文文本的压缩效果提升了多少？
4. 哪些中文词组被合并成了独立 token？
