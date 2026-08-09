# NanoChat 全流程复现（英文数据 · CPU/GPU 跨平台）· 使用说明

用**英文原版数据**（ClimbMix + SmolTalk / MMLU / GSM8K）在 **Windows CPU 或带 CUDA 的机器**上
跑通 NanoChat 完整流水线：`下载数据 → 训练 tokenizer → 预训练 → 评测 → SFT → 英文对话`。

两套脚本（都**不改任何原文件**，数据/评测全走 ModelScope 镜像 + 本地 eval_data）：
- **cpustepX**：CPU 套件（Windows / Linux 均可），默认练习版 ≤30 min。
- **gpustepX**：GPU 套件（AutoDL 单卡等），`NANOCHAT_CONFIG=fast|full` 两档
  （fast≈30 min / full≈20 h），已在 RTX 4090D 实测，见第 4 节。

CPU 流水线：

```
cpustep1 下载（ModelScope 镜像） → cpustep2 训练 Tokenizer(8192) → cpustep3 tok 评测
→ cpustep4 base_train(190步,d6) → cpustep5 base_eval（core+bpb+sample，CORE 本地）
→ cpustep6 chat_sft(25步，ChatCORE 本地) → cpustep7 chat_cli 英文对话
```

**目标预算：除下载外整链 ≤ 30 分钟**（本机实测约 25 min，见第 2 节逐步耗时）。

所有步骤脚本在 `runs-in-zh-en/` 下（CPU：`cpustep*.py`；GPU：`gpustep*.py`），
**不改任何原文件**（scripts/、nanochat/ 只读），国内网络 / 小词表适配全部通过新增的
`*_zh` 文件实现（`nanochat/dataset_zh.py`、
`nanochat/eval_data_zh.py`、`tasks/common_zh.py`、`tasks/smoltalk_zh.py`、`tasks/mmlu_zh.py`、
`tasks/download_sft_data_zh.py`、`scripts/base_eval_zh.py`、`scripts/chat_eval_zh.py`、
`scripts/chat_sft_zh.py`）。

**数据目录**默认 `<repo>/data/zh-en/`（`NANOCHAT_BASE_DIR`，`_common.py` 自动推导绝对值），
与 runs-in-ch 中文套件（`<repo>/data/`）互相隔离；**GPU 套件**默认独立使用
`<repo>/data/gpu/`（`_gpu_common.py` 设置），与 CPU 套件互不覆盖。

---

## 1. 概述（运行环境 + 产物速查 + 练习版/完整版）

### 1.1 运行环境（本机实测，Windows CPU）

| 项目 | 值 |
|---|---|
| CPU | 16 核（Intel） |
| 系统 | Windows 11（控制台默认 GBK 编码） |
| Python | 3.12.9（本机实测；uv 环境 `.venv\Scripts\python.exe` 亦可） |
| PyTorch | CPU 版 2.12.1+cpu，fp32 eager |
| 编译器 | **无 MSVC**（缺 `cl`，`torch.compile` 无法编译，见 1.2） |
| 磁盘 | `data\` 可用 70.7 GB（cpustep0 实测） |

> **先跑检测脚本摸清机器**：拿到陌生机器（可能连 Python 都没装、系统配置未知），先执行
> `powershell -ExecutionPolicy Bypass -File runs-in-zh-en/win_env_check.ps1`（Windows）
> 或 `bash runs-in-zh-en/linux_env_check.sh`（Linux）——纯系统原生命令、无需 Python，
> 输出 系统/CPU/内存/磁盘/GPU/Python/torch/编译工具链（MSVC 或 GCC，torch.compile 依赖），
> 再对照本表判断能否直接开跑、还缺哪些依赖。

> 换成带单 GPU 的机器也能跑：`autodetect_device_type()` 自动选 CUDA，见第 4 节。

### 1.2 跨平台三个关键坑（已由 `_common.py` 自动处理）

1. **`torch.compile` 不可用（无 MSVC）**：Windows 若没有 MSVC `cl`，Inductor 编译的 C++
   wrapper 报 `Compiler: cl is not found`。`_common.py` 在 `win32`（或
   `NANOCHAT_TORCH_COMPILE=0`）时置 `TORCH_COMPILE_DISABLE=1`，`torch.compile` 退化为 eager。
2. **线程数必须显式设满**：torch 步骤进程内 `torch.set_num_threads(os.cpu_count())`，
   避免子进程拿不到线程设置（Windows 下 `OMP_NUM_THREADS` 无效）。
3. **控制台 GBK 编码**：脚本重配 UTF-8，子进程继承 `PYTHONIOENCODING=utf-8`。

> 所有步骤都从项目根运行 `python runs-in-zh-en/cpustepX.py`；从任意 cwd 运行也可。

### 1.3 产物位置速查（都在 `data\zh-en\` 下）

| 内容 | 路径 | 说明 | 来源 |
|---|---|---|---|
| **预训练语料** | `base_data_climbmix\shard_*.parquet` | 英文 ClimbMix（≥8 片，末片含 val） | cpustep1 |
| **分词模型** | `tokenizer\tokenizer.pkl`（BPE）+ `token_bytes.pt` | **词表 8192** | cpustep2 |
| 任务数据 | `task_data\` | SmolTalk / MMLU / GSM8K + ChatCORE（ARC/HumanEval）本地缓存 | cpustep1 / eval_data_zh |
| CORE 评测包 | `eval_bundle\` | 由 `eval_data/eval_bundle.zip` 解压（core.yaml + 22 任务 jsonl） | eval_data_zh |
| **预训练模型** | `base_checkpoints\d6\model_*.pt` + `meta_*.json` | d6：6 层 / 384 维，**26.3M 参数** | cpustep4 |
| 基座评测结果 | `base_eval\base_model_000190.csv` | bpb / CORE 22 任务 accuracy | cpustep5 |
| **SFT 模型** | `chatsft_checkpoints\d6\model_*.pt` + `meta_*.json` | SFT 后最终对话模型 | cpustep6 |

> `task_data\` 下还有中文套件（runs-in-ch）留下的数据，两者按 slug 分目录互不冲突。

### 1.4 练习版 vs 完整版

当前默认是**练习版**（目标：全链 ≤30 min 跑通；本机实测见第 2 节）。完整版数值请在
对应文档/脚本里把迭代数改回 runcpu.sh 官方值：

| 步骤 | 练习版默认 | 完整版（runcpu.sh 同款超参） |
|---|---|---|
| cpustep2 词表 | 8192 训练，`--max-chars=120M` | 32768 / `max-chars=2B` |
| cpustep4 预训练 | **190 迭代**（`NANOCHAT_PRETRAIN_ITERS`） | 5000 迭代（~15-16 h） |
| cpustep6 SFT | **25 迭代**（`NANOCHAT_SFT_ITERS`） | 1500 迭代（~4-5 h） |

> 练习版词表 8192 会改变最终指标（与 README 其他处 32768 的预期结果不同）；
> 想要跑完整版指标就把上述词表/迭代数调回。

---

## 2. 各步骤：配置 + 实测耗时（本机 16 核 CPU）

> 除 cpustep1（下载）外，**整链实测合计 ≈ 25.3 min**（cpustep2..7，见下表实测耗时）。

| 脚本 | 步骤 | 实测耗时 | 关键产物 |
|---|---|---|---|
| `cpustep0_env_eval.py` | 环境巡检（可选 `--install`） | < 0.1 min | 环境判定 |
| `cpustep1_prepare_data.py` | 下载 ClimbMix + SFT 任务数据（ModelScope） | **3.4 min**（全新下载；已就绪时 0.0 min） | `base\` + `task_data\` |
| `cpustep2_tok_train.py` | 训练 8192 词表 BPE | **0.1 min** | tokenizer |
| `cpustep3_tok_eval.py` | tokenizer 压缩率评测 | **0.1 min** | bytes/token |
| `cpustep4_base_train.py` | base 预训练 d6 190 步 | **17.3 min** | base_checkpoints |
| `cpustep5_base_eval.py` | base core + bpb + 采样（本地 CORE） | **1.8 min** | core / bpb / sample |
| `cpustep6_chat_sft.py` | SFT 25 步 + ChatCORE（本地） | **6.0 min** | chatsft_checkpoints |
| `cpustep7_chat_cli.py` | 对话界面 | **0.0 min** | 交互回复 |

> 实测（本机 16 核 CPU）：cpustep4 的 190 步训练循环 16.0 min ≈ 0.084 min/步 + 固定开销 ~1.3 min；
> cpustep2 的 BPE 训练仅 4.1s（8192 小词表，120M 字符）。整链耗时主要被 cpustep4 占据。

### cpustep0 — 环境巡检 + 依赖补装

```bat
python runs-in-zh-en/cpustep0_env_eval.py            :: 只巡检
python runs-in-zh-en/cpustep0_env_eval.py --install  :: 巡检 + pip 补齐缺失
```

只读汇报：系统 / CPU 核数 / 内存 / Python / torch / CUDA / 关键依赖 / 编译工具链 /
磁盘剩余。**不含 GPU 时 `torch.compile` 已退化为 eager，C++ 工具链缺失不影响运行。**

### cpustep1 — 准备数据（下载）

```bat
python runs-in-zh-en/cpustep1_prepare_data.py
:: NANOCHAT_NUM_SHARDS=8（默认需 ≥8 片）
```

- ClimbMix 英文语料（`nanochat/dataset_zh.py`，ModelScope 镜像）→ `base_data_climbmix/`。
- SFT 任务（SmolTalk / MMLU / GSM8K，`download_sft_data_zh.py`）→ `task_data/`，SFT 全程离线。

**实测 3.4 min**（本机走镜像，幂等：已存在则跳过）。

### cpustep2 — 训练 tokenizer

```bat
python runs-in-zh-en/cpustep2_tok_train.py
:: NANOCHAT_MAX_CHARS=120000000（默认）  NANOCHAT_VOCAB_SIZE=8192（默认）
:: NANOCHAT_MAX_CHARS=2000000000 NANOCHAT_VOCAB_SIZE=32768   # 完整版
```

rustbpe 训练 8192 词表 BPE（多线程，进程内运行）。实测 **0.1 min**（120M 字符、小词表
merge 只剩 7927 次）；32768/2B 约 25-30 min。

### cpustep3 — tokenizer 压缩率评测

```bash
python runs-in-zh-en/cpustep3_tok_eval.py
```

比较 GPT-2/GPT-4 的 bytes/token。**实测 0.1 min**。8192 词表下英文压缩率略逊 GPT-2
（正常：小词表）。

### cpustep4 — base 预训练

```bash
python runs-in-zh-en/cpustep4_base_train.py            :: 练习版 190 步
NANOCHAT_PRETRAIN_ITERS=40 python runs-in-zh-en/cpustep4_base_train.py  :: 快速版
NANOCHAT_PRETRAIN_ITERS=5000 python runs-in-zh-en/cpustep4_base_train.py :: 完整版
```

模型/超参 == runcpu.sh：depth=6 / head-dim=64 / window-pattern=L / max-seq-len=512 /
batch 32 / total-batch 16384。练习版 eval/sample-every=50、eval-tokens=32768、
**CORE 英文 ICL 跳过**（--core-metric-every=-1，由 cpustep5 单独用本地 bundle 评）。
**实测 17.3 min**（190 步训练循环 16.0 min ≈ 0.084 min/步 ≈ 5.3 s/步 + 固定开销 ~1.3 min）。
val bpb 随步数下降：50 步 2.183 → 100 步 2.001 → 150 步 1.931 → 190 步 **1.906**。
检查点 → base_checkpoints（model_000190.pt）。

### cpustep5 — base 评估

```bash
python runs-in-zh-en/cpustep5_base_eval.py
:: NANOCHAT_EVAL_MAX_PER_TASK=16（默认）样本数
```

`--eval=core,bpb,sample`（**CORE 22 任务本地优先**：数据来自 `<repo>/eval_data/eval_bundle.zip`，
解压到 `data/zh-en/eval_bundle/`，不再走海外 S3；`--max-per-task=16` 限制样本数）。
**实测 1.8 min**。190 步模型：train bpb **1.846** / val bpb **1.889**；CORE metric **-0.0105**
（接近随机——190 步小模型，完整版训练后才见分晓）。结果写入 `base_eval/base_model_000190.csv`。

### cpustep6 — SFT

```bash
python runs-in-zh-en/cpustep6_chat_sft.py
NANOCHAT_SFT_ITERS=1500 python runs-in-zh-en/cpustep6_chat_sft.py  # 完整版
```

在 base 上 SFT（SmolTalk + MMLU×3 + GSM8K×4 混合，默认 25 迭代，检查点落在 step 24）。
**ChatCORE 默认启用**（`--chatcore-every=25`、`--chatcore-max-cat=200`），数据来自本地
`<repo>/eval_data/` 的 parquet（MMLU / ARC×2 / GSM8K / HumanEval，装进 task_data 缓存），
不联网；生成式任务（GSM8K / HumanEval）各 24 样本。**使用 `smoltalk_zh.py` / `mmlu_zh.py`
过滤**（为什么见第 5 节）。
**实测 6.0 min**（25 步训练 2.2 min + ChatCORE 评测 ~3.8 min；含过滤后首次缓存构建）。
val bpb **2.076**（初值 2.45）正常收敛、无 NaN；ChatCORE 分类：ARC-Easy **24.5%** /
ARC-Challenge **28.5%** / MMLU **23.0%**；生成式 GSM8K / HumanEval 0%（小模型算不出算术，
完整版训练后才见效）。检查点 → chatsft_checkpoints（model_000024.pt）。
想跳过 ChatCORE：`NANOCHAT_CHATCORE_EVERY=-1 python runs-in-zh-en/cpustep6_chat_sft.py`。

### cpustep7 — 对话

```bat
python runs-in-zh-en/cpustep7_chat_cli.py "What is the capital of France?"
python runs-in-zh-en/cpustep7_chat_cli.py                                   :: 交互
```

加载最新 SFT 模型。**实测 0.0 min**（2s）。25 步练习模型只能输出极简/幻觉回复
（实测 `What is the capital of France?` → 仅 `A`），完整版训练后才会说清 "Paris"。

---

## 3. 适配文件总览（全部新增，原文件零改动）

| 文件 | 作用 |
|---|---|
| `nanochat/dataset_zh.py` | ClimbMix 下载的 ModelScope 镜像版 |
| `tasks/common_zh.py` | 本地 `task_data/` 缓存加载 + `filter_conversations_zh` 过滤工具 |
| `tasks/smoltalk_zh.py` | SmolTalk 英文版 + 过滤（>512 token / 无监督删除） |
| `tasks/mmlu_zh.py` | MMLU 版 + 过滤（8192 词表下 35% 题超长） |
| `tasks/download_sft_data_zh.py` | ModelScope 预下载 SmolTalk / MMLU / GSM8K |
| `nanochat/eval_data_zh.py` | 本地 `eval_data/` 优先：`ensure_core_eval_bundle()`（解压 CORE 22 任务 bundle）、`install_chatcore_datasets()`（ARC/MMLU/GSM8K/HumanEval parquet 装进 task_data 缓存） |
| `scripts/base_eval_zh.py` | base_eval 的 _zh 拷贝：CORE 用本地 eval_bundle.zip，不再走海外 S3 |
| `scripts/chat_eval_zh.py` | chat_eval 的 _zh 拷贝：启动时安装 ChatCORE 数据；MMLU 固定用**原版**任务类（importlib 绕过 mmlu_zh 过滤，保证评测准确性） |
| `scripts/chat_sft_zh.py` | chat_sft 的 _zh 拷贝：内部 `from scripts.chat_eval_zh import run_chat_eval`，训练前安装数据 |
| `runs-in-zh-en/_common.py` | `NANOCHAT_BASE_DIR` 推导 / sys.path / `TORCH_COMPILE_DISABLE` / UTF-8 / `install_zh_modules`（`sys.modules` 重定向到 `*_zh`）/ `install_windows_signal_patch`（Windows 下把 engine 的 SIGALRM 计算器超时换成线程版）/ `run_in_process` / `timed` / env 变量默认值 |
| `runs-in-zh-en/_gpu_common.py` | GPU 套件共享工具：复用 `_common.py` 纯工具，独立数据目录 `data/gpu/`，`NANOCHAT_CONFIG=fast\|full` 配置 profile，`NANOCHAT_GPU_*` 逐项覆盖 |
| `runs-in-zh-en/cpustep0..7.py` | CPU 各步骤入口 |
| `runs-in-zh-en/gpustep0..8.py` | GPU 各步骤入口（见第 4 节） |
| `runs-in-zh-en/gpu_run_all.sh` | GPU 一键全流程（`NANOCHAT_CONFIG=fast\|full`） |
| `runs-in-zh-en/win_env_check.ps1` | 环境巡检脚本（Windows 版，PowerShell 原生命令，无需 Python）：系统/CPU/内存/磁盘/GPU/**MSVC**/Python/torch |
| `runs-in-zh-en/linux_env_check.sh` | 环境巡检脚本（Linux 版，bash 原生命令，无需 Python）：同上，含 **GCC/g++/make/ninja** 工具链，兼容 AutoDL/无 GPU/无 Python 的机器 |

**环境巡检（无需 Python）**：拿到一台陌生机器先跑 `bash runs-in-zh-en/linux_env_check.sh`
（Linux）或 `powershell -ExecutionPolicy Bypass -File runs-in-zh-en/win_env_check.ps1`
（Windows），即可用系统原生命令摸清 系统/CPU/内存/磁盘/GPU/Python/torch 配置。

**进程内运行技巧**：原脚本（base_train/chat_sft/tok_train/tok_eval/chat_cli）顶部硬编码
`from tasks.common` / `from tasks.smoltalk` 等 import。`install_zh_modules()` 把
`sys.modules` 里的 `tasks.common`、`tasks.smoltalk`、`tasks.mmlu`、`nanochat.dataset`
重定向到 `*_zh`，再在**同一进程** runpy 原脚本；于是那些 import 天然拿到 `*_zh` 版
（本地缓存 + 过滤），且 Windows 下 `torch.set_num_threads` 生效。

**ChatCORE / CORE 本地化**：`scripts/chat_sft_zh.py`、`scripts/chat_eval_zh.py`、
`scripts/base_eval_zh.py` 都是对应原脚本的拷贝，只在启动处多调用
`nanochat/eval_data_zh.py` 的安装函数，把 `<repo>/eval_data/` 下随仓库带的数据装进
`<base_dir>/`；`tasks/*` 因此读到本地 parquet，全程不访问海外。

---

## 4. GPU 套件（gpustepX）· AutoDL 单卡 4090D 实测

`speedrun.sh` 的**单卡 GPU** 改造版，在 AutoDL 租用机（RTX 4090D 单卡、国内网络，
可访问 ModelScope / Gitee、不可访问 HF/GitHub）上完整验证通过。与 cpustepX 同源：
复用同一批 `*_zh` 适配（ModelScope / 本地 CORE/ChatCORE），数据目录独立为
`<repo>/data/gpu/`（`_gpu_common.py` 复用 `_common.py` 纯工具并重设 `NANOCHAT_BASE_DIR`）。

### 4.1 两套配置（`NANOCHAT_CONFIG`）

| 配置 | 数据 | 模型 | 预算 |
|---|---|---|---|
| `fast`（默认） | ClimbMix 8 shard（~0.8 GB）+ SFT | d6 / head-dim 64 / seq 512，base 3000 步 + SFT 1000 步 | **≈30 min** |
| `full` | ClimbMix 170 shard（~17 GB）+ SFT | d24 / head-dim 128 / seq 1024 / total-batch 262144，base 7000 步 + SFT 完整 epoch | **≈20 h** |

所有数值可用 `NANOCHAT_GPU_*` 环境变量逐项覆盖（如 `NANOCHAT_GPU_BASE_ITERS=4000`、
`NANOCHAT_GPU_CHAT_EVAL_MAX_PROBLEMS=50`）。训练中 CORE 与 CORE/ChatCORE 评测全走本地
`eval_data/`（`eval_bundle.zip` + ARC/MMLU/GSM8K/HumanEval parquet），不访问海外。

**fast 模式示例（端到端验证）**：`fast` 训练出的 6 层模型，用
`python runs-in-zh-en/gpustep8_chat_cli.py -p "What is the capital of France?"` 单次提问，
实测输出如下：

> **问**：What is the capital of France?
> **答**：The capital of France is Paris. It is a city renowned for its rich history,
> attracting millions of visitors each year. The capital of France is Paris. It is a
> beautiful city known for its rich history, beautiful architecture, and vibrant cultural
> heritage.

**解释**：这是个"冒烟测试"——模型答对了（Paris），说明下载数据、训 tokenizer、预训练、
SFT、GPU 推理整条链路全部打通。但 6 层模型只见过 ~0.8 GB 数据、训练才 3000+1000 步，
回答明显重复啰嗦，这正常；`full`（24 层 / 170 shard）质量会好得多。`fast` 的定位是
**~30 分钟内验证流水线可复现**，而不是产出可用模型。

### 4.2 步骤与一键运行

| 步骤 | 内容 | 数据源 |
|---|---|---|
| `gpustep0_setup.py` | 环境/GPU 巡检；可选从 Gitee 克隆仓库 | Gitee |
| `gpustep1_prepare_data.py` | 下载 ClimbMix + SFT 任务数据 | ModelScope |
| `gpustep2_tok_train.py` | 训练 BPE tokenizer（纯 CPU） | 本地 |
| `gpustep3_tok_eval.py` | tokenizer 压缩率评估 | 本地 |
| `gpustep4_base_train.py` | 基座预训练（单卡 GPU，bf16，训练中 CORE） | 本地 |
| `gpustep5_base_eval.py` | CORE / bpb / sample 基座评估 | 本地 eval_data |
| `gpustep6_chat_sft.py` | SFT 微调 + 训练中 ChatCORE | 本地 |
| `gpustep7_chat_eval.py` | SFT 模型 ChatCORE 终评 | 本地 eval_data |
| `gpustep8_chat_cli.py` | 对话（交互式 / `-p` 单次问答） | 本地 |

```bash
bash runs-in-zh-en/gpu_run_all.sh                           # fast，一键全流程
NANOCHAT_CONFIG=full bash runs-in-zh-en/gpu_run_all.sh      # full
python runs-in-zh-en/gpustep4_base_train.py             # 单步运行
```

### 4.3 AutoDL 实测记录（2026-08-09 · RTX 4090D 24G · torch 2.8.0+cu128）

| 步骤 | 结果 | 耗时 |
|---|---|---|
| gpustep0 | 通过（SM 8.9 → bf16 自动） | 0.0 min |
| gpustep1 | ModelScope 9 shard + SFT 数据 | 5.4 min |
| gpustep2 | vocab 8192 | 0.2 min |
| gpustep3 | 通过 | 0.1 min |
| gpustep4 | 3000 步 ~410k tok/s、MFU ~24%，CORE 本地 **0.0427** | 5.4 min |
| gpustep5 | CORE 22 任务本地 **0.0346** | 0.4 min |
| gpustep6 | SFT 1000 步，ChatCORE **0.0267** | 8.3 min |
| gpustep7 | ChatCORE 终评 **0.0320**（各任务 100 题） | 4.6 min |
| gpustep8 | GPU 推理：`The capital of France is Paris.` | 0.1 min |

**整链实测 ≈ 24 min**（fast 配置，30 min 预算内；含数据下载）。

### 4.4 关键说明

- **单卡运行**：直接进程内跑 `scripts/base_train.py`，不需要 torchrun。
- **精度**：SM≥8 自动 **bf16**；`--fp8` 需 H100+，本卡不开。
- **注意力**：无 FA3 → 自动回退 PyTorch **SDPA**。
- **评测全本地**：`eval_data/` 缺失时自动降级（gpustep5 只跑 bpb+sample，
  gpustep6/7 关闭 ChatCORE），训练与对话不受影响。
- **fast 的 `chat_eval_max_problems=100`**：生成式 GSM8K/HumanEval 每题需自回归生成，
  样本数不宜过大（实测 1000 → ~17 min，100 → 4.6 min）。
- **训练中 CORE 频率**：`NANOCHAT_GPU_CORE_METRIC_EVERY`（默认 2000）控制 base_train
  中 CORE 评测间隔；`NANOCHAT_GPU_CORE_METRIC_MAX_PER_TASK` 控制每任务样本数
  （fast 200 / full 2000），样本越少训练越快。
- **显存不足（OOM）**：`full` 用 seq 1024 / device-batch 16 已给 24G 显存留余量；
  仍 OOM 时 `NANOCHAT_GPU_BASE_DEVICE_BATCH=8` 或 `NANOCHAT_GPU_BASE_SEQ_LEN=512`。
- **耗时调参**：主要看 `NANOCHAT_GPU_BASE_ITERS` / `NANOCHAT_GPU_SFT_ITERS` /
  `NANOCHAT_GPU_NUM_SHARDS`；tokenizer 训练为纯 CPU（`vocab_size` / `max_chars`）。
- **断点续跑**：各步骤幂等，重复运行跳过已完成部分（数据/检查点已存在）。
- **环境准备（AutoDL 首次）**：PyPI 可访问、wheel 自带 CUDA（pyproject 已锁
  `torch==2.9.1`）：
  ```bash
  git clone <你的 Gitee 地址>/nanochat.git && cd nanochat
  uv sync --extra gpu            # 或直接用 AutoDL 预装的 torch CUDA 环境
  python runs-in-zh-en/gpustep0_setup.py   # 巡检 GPU/依赖（可代做 Gitee 克隆）
  NANOCHAT_CONFIG=fast bash runs-in-zh-en/gpu_run_all.sh
  ```

### 4.5 用 cpustepX 直接在 GPU 跑（备选）

如果只想临时把 cpustepX 切到 GPU：`autodetect_device_type()` 自动选 CUDA，
把 `--device-batch-size` 调大、`--depth` 升 12/24、迭代数改完整版即可
（见 1.4 节）；但推荐直接使用 gpustepX——参数与独立数据目录已按 GPU 调好。

---

## 5. 常见问题 & 排查

- **SFT loss=NaN**：8192 词表下对话普遍超 512 token，bestfit 整行 padding → 0/0。
  本套件已由 `smoltalk_zh/mmlu_zh` 过滤规避；若仍现：
  1) 删 `task_data/HuggingFaceTB--smol-smoltalk/_valid_indices_*.pkl`、`task_data/cais--mmlu/...`
     重跑 cpustep6（换了 tokenizer 必须删）；
  2) 若 index 缓存已过期却报 0 行：确认 `cpustep2` 用的词表与 SFT 加载的 `tokenizer.pkl` 一致。
- **CORE / ChatCORE 下载失败（海外 S3）**：已全部本地化——cpustep5 的 CORE 用
  `<repo>/eval_data/eval_bundle.zip`（解压到 `data/zh-en/eval_bundle/`），cpustep6 的
  ChatCORE 用 `<repo>/eval_data/` 的 parquet（装进 `task_data/` 缓存），国内直跑不访问海外。
- **ChatCORE 生成式评测 `signal` 无 SIGALRM（Windows）**：`nanochat/engine.py` 的计算器
  超时用 Unix 专属 `signal.alarm`，Windows 会崩在 GSM8K 生成评测。已由 `_common.py` 的
  `install_windows_signal_patch()` 换成线程超时版（语义等价），无需改原文件。
- **`Compiler: cl not found`（Windows）**：`_common.py` 默认禁用 compile；想启用需
  VS C++ 工具链并把 `cl` 加入 PATH。
- **`data/task_data/` 里出现中文任务**：是本机 runs-in-ch 遗留，不影响英文套件
  （按 slug 分目录）；彻底隔离可另设 `NANOCHAT_BASE_DIR`。
- **终端中文乱码**：脚本已重配 UTF-8；仍乱码用 `chcp 65001` / VSCode 终端。
