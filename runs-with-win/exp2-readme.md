# 实验2：Toy Forward Pass

> 本文件是从 `local_ex2/README.md` 拷贝并适配到 `runs-with-win/` 的版本：路径改为读取 `config.yaml`，产物输出到 `runs-with-win/checkpoint/`，运行方式见文末「如何运行」。

## 目的

用 noveltxt 中文小说语料训练一个约 500 万参数的小 GPT 模型，在 CPU 上完整跑一遍训练流程，验证 NanoChat 的模型代码能用**真实数据**正常执行 forward、backward、optimizer step、文本生成和 checkpoint 保存/加载。这是上 GPU 前的"冒烟测试"。

## 宏观流程

```
exp1 保存的 5K 词表（tokenizer_5k/）
noveltxt/ 小说（GBK 编码）
    → 读取约 100 万字文本
    → 加载 exp1 训练好的 5K BPE tokenizer
    → 编码小说 → 得到真实 token ID 序列
    → 裁切成 (B=8, T=128) 的批次
    → 送入 GPT(5层, 256维, 4头, 词表~5K) 训练 100 步
    → 观察 loss 在真实数据上的下降趋势
    → 保存/加载检查点
```

## 模型参数

| 参数 | exp2 | nanochat 默认（depth=12） |
|------|------|--------------------------|
| n_layer | 5 | 12 |
| n_embd | 256 | 768 |
| n_head | 4 | 6 |
| n_kv_head | 4 | 6 |
| head_dim | 64 | 128 |
| 参数量 | ~500 万 | ~1.5 亿 |
| 训练平台 | CPU | 8×H100 GPU |
| 训练数据 | noveltxt（~6000 万 token） | ClimbMix-400B（4000 亿 token） |
| 序列长度 | 128 | 2048 |
| 批次 token/步 | 1,024 | 524,288 |

## 输入

| 输入 | 说明 |
|------|------|
| `noveltxt/` 目录 | GBK 编码的中文科幻小说，约取 100 万字 |
| `tokenizer_5k/` | exp1 保存的 5K BPE 词表 |
| `model` | GPT(5层, 256维, 4头, 词表~5K)，CPU |

## 输出

| 输出 | 说明 |
|------|------|
| 编码示例 | 小说原文 → token ID 序列 |
| Loss 趋势 | 100 步训练中 loss 是否在真实文本上下降 |
| 梯度范数 | 各参数梯度的 L2 范数，确认反向传播正常 |
| 参数统计 | 各组件（wte/lm_head/transformer/scalar）的参数量 |
| 检查点 | 保存后加载，验证 forward 正常 |

## 与 exp1 的关系

| exp1 | exp2 |
|------|------|
| 训练 tokenizer，**观察 BPE 合并过程** | 用 tokenizer 产生的 ID 训练模型，**观察 loss 下降** |
| 输出 mergeable_ranks 和编码统计 | 输出 loss/梯度/参数统计 |
| 保存 5K 词表到 `tokenizer_5k/` | 加载 exp1 保存的词表 |

## 前置依赖

```powershell
# 先跑 exp1 训练并保存 5K 词表（本目录版本，产物在 runs-with-win/tokenizer_5k/）
cd nanochat
.\.venv\Scripts\python.exe runs-with-win\exp1_tokenizer_bpe.py
```

## 如何运行

本目录版本已改为读取 `config.yaml`（路径/参数均为相对 `runs-with-win/` 的相对路径），产物输出到 `runs-with-win/checkpoint/`：

```powershell
cd nanochat
.\.venv\Scripts\python.exe runs-with-win\exp2_toy_forward.py
# 可选参数覆盖：--steps 30 --batch-size 4 --novel-dir <路径>
```

## 预期结果

```
加载 tokenizer，词表: ~5129
参数量: ~5,000,000 (~5M)
100 步训练: loss 从 ~8.5 缓慢下降
✅ 前向/反向/优化器/检查点全部正常
总耗时: ~1 分钟内
```
