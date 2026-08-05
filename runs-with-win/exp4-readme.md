# 实验4：本地 SFT 微调 + 交互式对话

## 目的

在 CPU 上跑完整的 SFT（Supervised Fine-Tuning）微调流程：加载 exp2 预训练的小 GPT、用 exp3 生成的本地样本数据（格式与 HuggingFace 一致但本地生成）、训练成能"对话"的模型，最后通过命令行与它聊天。

## 两个脚本

| 脚本 | 作用 |
|------|------|
| `exp4_local_sft.py` | 本地 SFT 微调，产物输出到 `runs-with-win/sft_checkpoint/` |
| `chat_cli.py` | 加载 SFT 检查点，交互式对话（空行退出） |

## 宏观流程

```
exp2 预训练 checkpoint（runs-with-win/checkpoint/）
exp3 样本数据（runs-with-win/sample_data/，smoltalk/mmlu/gsm8k JSON）
     ↓
exp4_local_sft.py
    ├─ 加载 tokenizer（runs-with-win/tokenizer_5k/）
    ├─ 加载 exp2 模型权重
    ├─ TaskMixture 混合 3 个数据集的样本
    ├─ render_conversation → ChatML token 序列 + loss mask
    ├─ best-fit 打包成 (B, T) batch
    ├─ SFT 训练（默认 50 步，CPU）
    ├─ 推理测试（两个示例问题）
    └─ 保存到 runs-with-win/sft_checkpoint/
     ↓
chat_cli.py → 终端交互式对话
```

## 配置（config.yaml）

```yaml
sft:
  checkpoint_dir: sft_checkpoint   # SFT 检查点输出目录
  seq_len: 128
  batch_size: 4
  steps: 50                         # 训练步数
  lr: 0.01
```

SFT 模型复用 exp2 的模型结构（n_layer/n_embd 等来自 `pretrain` 段），仅替换最后的优化与训练数据。

## 与 exp1~exp3 的关系

```
exp1: 训练 BPE tokenizer → tokenizer_5k/
exp2: 用小说语料预训练小 GPT → checkpoint/        （本实验的初始权重）
exp3: 生成 SFT 样本 JSON → sample_data/            （本实验的训练数据）
exp4: 加载以上二者做 SFT 微调 → sft_checkpoint/    （最终对话模型）
```

## 如何运行

```powershell
# 在 nanochat 仓库根目录下
# 1) 先确保 exp1~exp3 产物就绪：
#    exp1_tokenizer_bpe.py → tokenizer_5k/
#    exp2_toy_forward.py   → checkpoint/
#    exp3_sample_sft_data.py → sample_data/

# 2) SFT 微调
.\.venv\Scripts\python.exe runs-with-win\exp4_local_sft.py

# 3) 交互式对话（空行退出）
.\.venv\Scripts\python.exe runs-with-win\chat_cli.py
```

## 命令行参数

`exp4_local_sft.py` 支持覆盖 config.yaml：

| 参数 | 作用 |
|------|------|
| `--tokenizer` | 覆盖 tokenizer 路径 |
| `--ckpt-dir` | 覆盖 exp2 checkpoint 路径 |
| `--sample-dir` | 覆盖样本目录路径 |
| `--save-dir` | 覆盖 SFT 检查点输出路径 |
| `--steps` | 训练步数 |
| `--lr` | 学习率 |

## 预期结果

```
加载 tokenizer，词表: ~5129
训练数据: 90 条
加载 checkpoint: .../checkpoint
构建模型: 5层/256维/4头
SFT 训练 50 步: loss 缓慢下降
推理测试输出两句回答
检查点已保存: .../sft_checkpoint
```

## 常见问题

- **找不到 exp2 checkpoint**：先运行 `exp2_toy_forward.py` 生成 `checkpoint/`。
- **找不到样本数据**：先运行 `exp3_sample_sft_data.py` 生成 `sample_data/`。
- **模型答非所问**：SFT 步数太少或语料太少属正常现象，这是教学级小模型。
