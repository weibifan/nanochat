# Windows CPU 本地复现 nanochat 实验记录（runs-with-win）

## 导言

在 **Windows 11（中文版）+ 无 GPU + 无法访问 HuggingFace** 的本地机器上，用一套拆成 4 个实验（exp1~exp4）的 Python 脚本，离线跑通 nanochat 的核心流程：**训练 Tokenizer → 预训练小模型 → 验证 SFT 数据管线 → SFT 微调 → 交互对话**。目标是验证本地所学的 nanochat 代码（`RustBPETokenizer` / `GPT` / `TaskMixture` / `render_conversation`）能否在纯 CPU 单机上端到端运行，为云端 GPU 实验（见 `runs-with-autodl/`）做铺垫。

**最终结果：** 5K 词表 BPE tokenizer → 5 层/256 维小 GPT（~500 万参数）预训练 1000 步 → SFT 数据管线 6 项测试全部通过 → SFT 微调 50 步 → `chat_cli` 可交互对话。全部产物在本地，**全程不联网**。

**核心结论：** 生产脚本（`tok_train.py` / `base_train.py` / `chat_sft.py`）的数据入口全部硬编码 HuggingFace（ClimbMix-400B、SmolTalk、MMLU、GSM8K），直接跑会联网失败。exp 系列把**唯一硬性差异——数据来源**换成读本地 `noveltxt/` 中文小说，即可在无网 Windows CPU 上跑通**同一套核心代码**（见第 2 节）。

---

## 1. 本地运行环境基本情况

### 硬件与系统（本机实际）

| 项目 | 值 |
|------|-----|
| 系统 | Windows 11 中文版 |
| 硬件 | 单机 CPU（无 GPU，可调至 GPU） |
| Python | `.venv`（nanochat 仓库根目录下，`uv sync --extra cpu` 创建） |
| 数据 | 本地 `..\local_ex1\noveltxt\`（GBK 编码中文小说） |
| 网络 | 全程不需要（无 HF、无 wandb 在线记录） |

> 路径、产物目录全部由 `config.yaml` 统一管理，且为相对 `runs-with-win/` 的相对路径。修改后直接生效，无需改代码。

### 依赖环境

前置条件（详见 `README.md` 的「前置依赖」节）：

1. **nanochat 仓库**（`runs-with-win/` 所在仓库）。
2. **Python 虚拟环境**：`nanochat\.venv\Scripts\python.exe`（含 torch CPU 版、rustbpe、tiktoken、numpy）。若无，先执行：
   ```powershell
   cd nanochat
   uv sync --extra cpu
   ```
3. **本地小说语料** `..\local_ex1\noveltxt\`（GBK 编码 `.txt`）。
4. 不需要任何网络访问。

---

## 2. 局限性及解决思路

本地复现主要遇到两类问题：生产代码**硬编码 HuggingFace 数据源**、以及 **Windows 平台特有的坑**。

### 问题 A：生产代码数据入口全部硬编码 HuggingFace

**现象：** `tok_train.py` / `base_train.py` / `chat_sft.py` 的数据入口全部从 HF 下载（ClimbMix-400B、SmolTalk、MMLU、GSM8K），在无网 Windows 上直接跑必然失败。

**解决思路：** 唯一硬性差异是**数据来源**，其余代码（tokenizer 训练、GPT 模型、ChatML 管线）可复用。exp 系列把「HF 下载」换成「读本地 `noveltxt/`（GBK 中文小说）」：

| 环节 | 生产代码（HF 数据） | exp 实验（本地数据） |
|------|-------------------|---------------------|
| 数据准备 | `nanochat.dataset` 从 HF 下载 parquet | 直接读本地 `noveltxt/`（GBK txt） |
| 训练 tokenizer | `scripts/tok_train`（2B 字符、词表 32768） | `exp1`（约 1 亿字符、词表 5120） |
| 预训练 | `scripts/base_train`（d12/d24、分布式） | `exp2`（5 层/256 维，CPU） |
| 数据管线 | `chat_sft.py` 内 `render_conversation` + best-fit | `exp3`（教学复刻，逐行一致） |
| SFT | `scripts/chat_sft`（TaskMixture 真实 HF 数据集） | `exp4`（本地中文样本 JSON） |
| 对话 | `scripts/chat_cli` | `chat_cli.py`（交互，逻辑相同） |

### 问题 B：Windows 平台特有的坑

| 问题 | 原因 | 解决 |
|------|------|------|
| 中文输出乱码 | 控制台 GBK 代码页 | 脚本已 `io.TextIOWrapper(..., 'utf-8')` 包装输出；对话乱码时先 `chcp 65001` |
| 管道传中文变 `???` | PowerShell 把 stdin 按 GBK 写，Python 按 UTF-8 读 | 交互式运行（不加管道），或 `Get-Content -Encoding UTF8 prompts.txt \| python` |
| `torch.compile` 报错 | 系统无 MSVC | 脚本统一 `torch._dynamo.config.disable = True` 禁用编译 |
| 语料乱码 | 语料编码假设是 GBK | 若语料是 UTF-8，改脚本 `open(..., encoding='gbk')` |

---

## 3. 实验基本流程、参数设置与最终结果

### 基本流程

```
noveltxt 中文小说 → exp1 训练 Tokenizer(512/5K 词表) → exp2 预训练小GPT(5层/256维/1000步) → exp3 验证 ChatML 数据管线+生成样本 → exp4 SFT微调(50步) → chat_cli 交互对话
```

四个实验层层递进：**exp1 产出词表 → exp2 用词表训练模型权重 → exp3 产出 SFT 训练数据 → exp4 用二者微调出最终对话模型**。每步产物是下一步输入，任何一环的 bug 都会在当步暴露。

### 各阶段参数

**exp1 tokenizer：** 两阶段训练——512 词表用 1000 万字符、5K 词表用 1 亿字符；5K 词表（含 9 个 ChatML 特殊 token）保存到 `tokenizer_5k/`。

**exp2 预训练（小 GPT）：**
```python
# config.yaml: pretrain 段
n_layer=5, n_embd=256, n_head=4, n_kv_head=4, window_pattern="L"
seq_len=128, batch_size=8, steps=1000, matrix_lr=0.1
```
- 参数量约 500 万（nanochat 默认 d12 约 1.5 亿），CPU 运行
- 1000 步，检查点 `checkpoint/model_0001000.pt`（约 40MB）+ `meta_001000.json`

**exp3 数据管线：** 6 组 dict 对话覆盖全部 case（简单/系统消息/多轮/Python 工具/Python 输出/长截断），逐 token 断言 + best-fit 打包模拟；`exp3_sample_sft_data.py` 从 smoltalk/mmlu/gsm8k 各生成 30 条样本到 `sample_data/`。

**exp4 SFT 微调：**
```python
# config.yaml: sft 段
seq_len=128, batch_size=4, steps=50, lr=0.01
```
- 输入 exp2 checkpoint + exp3 样本（90 条），`TaskMixture` 混合 3 个数据集
- 50 步，检查点 `sft_checkpoint/model_000050.pt` + `meta_000050.json`

### 最终结果

| 指标 | 值 | 说明 |
|------|-----|------|
| tokenizer 词表 | 5120（+9 特殊 token = 5129） | 中文小说训练，byte-level BPE |
| 预训练 loss | ~0.045（1000 步后） | 从 ~8.5 快速下降 |
| SFT loss | ~0.003（50 步后） | 在 90 条样本上微调 |
| 数据管线 | 6/6 测试通过 | ChatML 格式 + mask + best-fit 均正确 |
| 参数量 | ~500 万 | 检查点约 40MB |

**启动 chat_cli 对话（本地）：**

```powershell
cd nanochat
.\.venv\Scripts\python.exe runs-with-win\chat_cli.py
```

**预期输出：** 加载 tokenizer（词表 5129）与 SFT 模型后进入交互对话，空行退出。
> 注：这是教学级小模型，SFT 步数少、语料少，答非所问属正常现象。

---

## 4. 各步骤如何执行与预期结果

四个实验按依赖顺序依次执行，均在 nanochat 仓库根目录下用 `.venv` 直接运行（无需任何 shell 封装）。先确依赖就绪（见第 1 节）：

```powershell
cd nanochat
.\.venv\Scripts\python.exe runs-with-win\exp1_tokenizer_bpe.py   # 实验1
.\.venv\Scripts\python.exe runs-with-win\exp2_toy_forward.py     # 实验2
.\.venv\Scripts\python.exe runs-with-win\exp3_data_pipeline.py   # 实验3a
.\.venv\Scripts\python.exe runs-with-win\exp3_sample_sft_data.py # 实验3b
.\.venv\Scripts\python.exe runs-with-win\exp4_local_sft.py       # 实验4
.\.venv\Scripts\python.exe runs-with-win\chat_cli.py             # 对话
```

> 所有脚本通过 `config.yaml` 读路径/参数（相对 `runs-with-win/`），也支持命令行参数覆盖（`--steps`、`--sample-size` 等），详见各 `expN-readme.md`。
>
> **执行注意：** 直接用 `.venv\Scripts\python.exe` 运行即可。若先 `Activate.ps1` 激活环境，可省略路径直接 `python ...`。

### 实验 1：训练 BPE Tokenizer

**目的：** 用本地小说语料训练 BPE 分词器，直观理解 BPE「如何从零学到编码规则」——哪些字节对先被合并、中文在 byte-level 下如何被处理、词表大小对压缩率的影响。对应第 3 节「基本流程」第一步。

**依赖：** 本地语料 `noveltxt/`（`config.yaml` 的 `noveltxt` 字段）+ 已装依赖（`rustbpe` 训 BPE）。`tokenizer_5k/` 为新目录，本步自建。

**如何做：**
```powershell
cd nanochat
.\.venv\Scripts\python.exe runs-with-win\exp1_tokenizer_bpe.py
```

**预期结果：** 两阶段打印——512 词表（最早合并前 30 个 byte pair、中文相关 token）与 5K 词表（最长词组、多场景编码示例、词表压缩对比）；5K 词表自动保存到 `tokenizer_5k/`。

**如何验证：**
```powershell
# 1) 词表文件是否生成且非空
ls runs-with-win\tokenizer_5k\
# 2) tokenizer 能否真正加载、词表是否 5120（含特殊 token 5129）
.\.venv\Scripts\python.exe -c "from nanochat.tokenizer import RustBPETokenizer; t=RustBPETokenizer.from_directory(r'runs-with-win\tokenizer_5k'); print('vocab =', t.get_vocab_size())"
```

**耗时：** 约 1~2 分钟（视语料量）。详细说明见 `exp1-readme.md`。

### 实验 2：预训练小 GPT（Toy Forward Pass）

**目的：** 用 noveltxt 中文小说语料训练约 500 万参数小 GPT，在 CPU 上完整跑一遍训练流程，验证模型代码能用**真实数据**正常执行 forward、backward、optimizer step、检查点保存/加载——上 GPU 前的冒烟测试。对应生产 `base_train.py`。

**依赖：** 依赖实验 1 的产出——`tokenizer_5k/`（5K 词表）；预训练语料 `noveltxt/`。`checkpoint/` 为新目录，本步自建。

**如何做：**
```powershell
cd nanochat
.\.venv\Scripts\python.exe runs-with-win\exp2_toy_forward.py
# 可选参数覆盖：--steps 30 --batch-size 4 --novel-dir <路径>
```

**预期结果：** 加载 tokenizer（词表 ~5129）、参数量 ~500 万、1000 步训练 loss 从 ~8.5 缓慢下降、检查点保存/加载验证通过。产物到 `checkpoint/`。

**如何验证：**
```powershell
# 1) 检查点是否生成且非空
ls runs-with-win\checkpoint\
# 2) 训练指标是否合理（看 meta 里的最终 loss）
Get-Content runs-with-win\checkpoint\meta_001000.json
```

**耗时：** 1000 步约 1 分钟内（CPU）。详细说明见 `exp2-readme.md`。

### 实验 3：SFT 数据管线验证（ChatML 格式化）

**目的：** 在不涉及 GPU 的前提下，逐行验证 nanochat 的 SFT 数据格式化逻辑是否正确——dict 对话 → ChatML token 序列 + loss mask + best-fit 打包。数据管线 bug 不会报错，只会无声让模型学错，因此是上 SFT 前的「必过安检」。对应生产 `chat_sft.py` 的 `render_conversation` + best-fit packing。

**依赖：** 依赖实验 1 的产出——`tokenizer_5k/`（含 9 个 ChatML 特殊 token）。**不需要实验 2 的产物**（只用 tokenizer，不用模型）。

**如何做：**
```powershell
cd nanochat
# 1) 验证 ChatML 数据管线（6 个用例 + best-fit 打包）
.\.venv\Scripts\python.exe runs-with-win\exp3_data_pipeline.py
# 2) 生成本地 SFT 样本数据（产物 → runs-with-win/sample_data/）
.\.venv\Scripts\python.exe runs-with-win\exp3_sample_sft_data.py
```

**预期结果：** 所有 6 项测试通过（BOS 开头、user mask=0 / assistant mask=1、系统消息合并、Python 工具包裹、输出 mask=0、长截断）；`sample_data/` 下生成 smoltalk/mmlu/gsm8k 各 30 条 JSON。

**如何验证：**
```powershell
# 1) 样本是否生成（各 30 条，行数可查）
Get-ChildItem runs-with-win\sample_data\
# 2) 样本格式是否为 dict 对话（含 messages）
Get-Content runs-with-win\sample_data\smoltalk_sample.json -TotalCount 20
```

**耗时：** 约 5 秒。详细说明见 `exp3-readme.md`。

### 实验 4：本地 SFT 微调 + 交互对话

**目的：** 在 CPU 上跑完整 SFT 流程：加载 exp2 预训练小 GPT、用 exp3 生成的本地样本、训练成能「对话」的模型，最后用 `chat_cli` 交互聊天。对应生产 `chat_sft.py`。

**依赖：** 依赖实验 2 的产出——`checkpoint/`（初始权重）；依赖实验 3 的产出——`sample_data/`（训练数据）；隐式复用实验 1 的 `tokenizer_5k/`。必须先完成实验 1~3。

**如何做：**
```powershell
cd nanochat
# 1) SFT 微调（产物 → runs-with-win/sft_checkpoint/）
.\.venv\Scripts\python.exe runs-with-win\exp4_local_sft.py
# 2) 交互式对话（空行退出）
.\.venv\Scripts\python.exe runs-with-win\chat_cli.py
```

**预期结果：** 加载 tokenizer（~5129）、训练数据 90 条、加载 exp2 checkpoint、50 步 SFT loss 下降、推理测试输出两句回答、检查点保存到 `sft_checkpoint/`；`chat_cli` 进入对话。

**如何验证：**
```powershell
# 1) SFT 检查点是否生成且非空
ls runs-with-win\sft_checkpoint\
# 2) 指标是否合理（看 meta 里的最终 loss）
Get-Content runs-with-win\sft_checkpoint\meta_000050.json
```

**耗时：** 50 步约 10 秒（CPU）。详细说明见 `exp4-readme.md`。

### 完整对照表

| 实验 | 脚本 | 目的 | 产出 | 依赖前序 | 耗时 |
|------|------|------|------|---------|------|
| 1 | `exp1_tokenizer_bpe.py` | 训练 BPE tokenizer | `tokenizer_5k/` | 语料 `noveltxt/` | 1~2 min |
| 2 | `exp2_toy_forward.py` | 预训练小 GPT | `checkpoint/` | exp1 词表 | ~1 min |
| 3 | `exp3_data_pipeline.py` + `exp3_sample_sft_data.py` | 验证 ChatML 管线 + 生成样本 | `sample_data/` | exp1 词表 | ~5 s |
| 4 | `exp4_local_sft.py` | SFT 微调 | `sft_checkpoint/` | exp2 权重 + exp3 样本 | ~10 s |
| 对话 | `chat_cli.py` | 交互对话 | 终端 | exp4 检查点 | 即时 |

---

## 5. 易错问题及解决思路

| 问题 | 原因 | 解决 |
|------|------|------|
| 找不到 exp2 checkpoint | 未先运行 `exp2_toy_forward.py` | 按顺序先跑实验 1~3 |
| 找不到样本数据 | 未先运行 `exp3_sample_sft_data.py` | 生成 `sample_data/` 后再跑 exp4 |
| 模型答非所问 | SFT 步数少或语料少 | 属正常现象，教学级小模型 |
| 中文对话显示 `???` | PowerShell 管道把 stdin 按 GBK 写 | 交互式运行（不加管道）；必须管道时 `chcp 65001` 或用 UTF-8 文件 |
| 中文输出乱码 | 控制台 GBK 代码页 | 脚本已 UTF-8 包装输出；仍乱码先 `chcp 65001` |
| 语料读出来乱码 | 语料编码不是 GBK | 改脚本 `open(..., encoding='gbk')` 为实际编码 |
| `torch.compile` 报错（无 MSVC） | 系统无 C 编译器 | 脚本已 `torch._dynamo.config.disable = True` |
| 无 GPU 也能跑？ | 默认 CPU | 训练/推理全程离线 CPU，`config.yaml` 可调至 GPU |

---

## 6. 与 `runs/runcpu.sh` 的区别

runcpu.sh 是 nanochat 官方自带的教学演示脚本，专为单机 CPU/Mac 设计，与 runs-with-win 的动机一致（都是低算力、理解流程、不求模型质量）。两者跑的是**同一条流水线**（数据 → tokenizer → 预训练 → SFT → 对话），核心都是 `python -m scripts.{tok_train,base_train,chat_sft,chat_cli}`，区别只在执行环境与数据入口。

| 项目 | `runs/runcpu.sh`（官方） | `runs-with-win`（本套） |
|------|------------------------|------------------------|
| 执行方式 | bash 脚本（`bash runs/runcpu.sh`），一条龙跑完 | 直接 `python xxx.py`，拆成 exp1~exp4 逐步运行 |
| 目标系统 | Linux / Mac（bash、`uv venv` + `source`） | Windows 11 中文版（`.venv` 直接调 `python`） |
| 数据来源 | `nanochat.dataset` 从 HuggingFace 下载 ClimbMix-400B（联网） | 本地 `noveltxt/` 小说语料 + 本地样本（完全离线） |
| tokenizer 语料 | `tok_train --max-chars=2000000000`（20 亿字符、词表 32768） | `exp1`（约 1 亿字符、词表 5120，教学级） |
| 预训练 | 6 层 / `head-dim=64` / `max-seq-len=512` / 5000 步（`--depth=6`） | `exp2` 5 层 / 256 维 / 1000 步（CPU 教学） |
| SFT | `chat_sft --num-iterations=1500` | `exp4` 50 步（本地中文样本） |
| 对话 | `chat_cli -p` 单问 | `chat_cli.py` 可交互对话 |
| 依赖下载 | 装 uv → `uv sync --extra cpu` | 已有 `.venv`（`uv sync --extra cpu`） |
| 目的 | 官方 CPU 教学示例 | 理解全流程 + **无网 Windows 本地可复现** |

**数据入口是唯一硬性差异：** runcpu.sh 第 ① 步 `python -m nanochat.dataset` 直接把 ClimbMix-400B 从 HuggingFace 下载（联网、国内不可达），因此直接跑会失败；runs-with-win 把这一步换成「读本地 `noveltxt`（GBK 中文小说）」，即可在无网 Windows CPU 上离线跑通从 tokenizer 到对话的**同一套核心代码**。

---

## 附：本目录文件与对应关系

| 本地文件 | 说明 |
|---------|------|
| `config.yaml` | 所有路径/参数配置（相对本目录，改后直接生效） |
| `README.md` | 总说明（前提动机、与 runs 差异、运行方式、常见问题） |
| `exp1_tokenizer_bpe.py` | 实验1：训练 BPE 分词器 |
| `exp1-readme.md` | 实验1 详细说明 |
| `exp2_toy_forward.py` | 实验2：预训练小 GPT |
| `exp2-readme.md` | 实验2 详细说明 |
| `exp3_data_pipeline.py` | 实验3a：ChatML 数据管线验证 |
| `exp3_sample_sft_data.py` | 实验3b：生成 SFT 样本 |
| `exp3-readme.md` | 实验3 详细说明 |
| `exp4_local_sft.py` | 实验4：SFT 微调 |
| `chat_cli.py` | 实验4：交互式对话 |
| `exp4-readme.md` | 实验4 详细说明 |
| `tokenizer_5k/` | exp1 产物：5K 词表 tokenizer |
| `checkpoint/` | exp2 产物：预训练检查点 |
| `sample_data/` | exp3 产物：smoltalk/mmlu/gsm8k 各 30 条样本 |
| `sft_checkpoint/` | exp4 产物：SFT 检查点 |
