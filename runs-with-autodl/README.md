# GTX 4090 云端训练 nanochat 实验记录（AutoDL）

## 导言

在 AutoDL 平台租用一台 **RTX 4090D（24GB）** GPU 服务器，跑通 nanochat 完整训练流程：**下载数据 → 训练 Tokenizer → 预训练 → SFT 微调 → 推理评估**。目标是验证本地所学的 nanochat 代码能否在真实 GPU 上端到端运行，并绕开国内网络无法直接访问 HuggingFace 的限制。

**最终结果：** d4 玩具模型（4 层 / 256 维 / 2 头，参数量 36.70M）预训练 500 步 + SFT 完整 1 epoch（1868 步，约 25 分钟）训练完成。验证 BPB **0.5576**（对比预训练基线 1.145），MMLU 34.0%（远超随机基线 25%），GSM8K 0.0%。通过 `chat_cli` 可正常交互对话。

**核心结论：** 瓶颈不是算力而是网络。数据下载（HF 被墙）与依赖安装（uv 卡死）才是最大障碍，二者均有对应的解决思路（见第 2 节）。

---

## 1. AutoDL 服务器基本情况

### 租用实例的配置（购买时选定）

以下为在 AutoDL 平台创建实例时选择的配置（这些是购买时就知道的，环境是否如描述一致需用下方脚本验证）：

| 项目 | 值 |
|------|-----|
| GPU | RTX 4090D（24GB），¥1.88/h |
| 镜像 | PyTorch 2.8 + CUDA 12.8（预装 torch） |
| 数据盘 | 50GB（持久化，关机不丢，GPU 关机不计费） |
| SSH | `ssh -p 49302 root@connect.westc.seetacloud.com` |

> 代码目录、缓存目录等路径是实验过程中才确定的，见第 4 节各步骤。

### 查看服务器环境

**目的：** 确认服务器的系统 / Linux / GPU / Python / torch 等是否符合 nanochat 的训练要求，同时暴露版本冲突（torch）与缺失依赖，供第 2 节、第 4 节处理。**注意：这一步只是查看，不做任何安装。**

**做法：** 脚本已放在本目录 `check_server.sh`。分两步——先在本地用 `scp` 上传，再在服务器上运行。

**上传（在本地 PowerShell 执行）：**
```bash
scp -P 49302 ./check_server.sh root@connect.westc.seetacloud.com:/root/
```

**运行（在服务器 SSH 会话中执行）：**
```bash
bash /root/check_server.sh
```

**预期结果（示例，实际以你的机器为准）：**
```
=== 系统 ===
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
5.15.0-94-generic

=== GPU ===
name, memory.total [MiB], driver_version
NVIDIA GeForce RTX 4090 D, 24564 MiB, 595.80

=== CPU / 内存 ===
192
               total        used        free      shared  buff/cache   available
Mem:           1.0Ti        29Gi       169Gi       1.9Gi       807Gi       970Gi
Swap:             0B          0B          0B

=== CUDA 版本 ===
/root/check_server.sh: line 20: nvcc: command not found

=== Python ===
Python 3.12.3
/root/miniconda3/bin/python

=== PyTorch (从 Python 里检测) ===
torch: 2.8.0+cu128
CUDA 可用: True
GPU 名: NVIDIA GeForce RTX 4090 D

=== 磁盘 ===
Filesystem      Size  Used Avail Use% Mounted on
/dev/md0         50G  7.1G   43G  15% /root/autodl-tmp
overlay          30G   15G   16G  48% /

=== 关键 Python 包（本轮实验后续要用，此时可能尚未安装）===
huggingface_hub: 1.25.1
tiktoken: 0.13.0
numpy: 2.3.2
rustbpe: OK
filelock: OK
pyarrow: 25.0.0
```

> 上图是本次实际实例（AutoDL 24GB 4090D 机型）的完整输出。各租户机器 CPU/内存/磁盘等可能不同，属正常；**本实例 192 核 / 1TB 内存 / 30G 系统盘是较大机型，不影响实验。**

**判断：** Ubuntu 22.04 / Python 3.12 / GPU 24GB / torch 2.8+cu128 / `CUDA 可用: True`，满足 nanochat 训练要求（requires-python>=3.10）。注意三点：① torch 是 2.8 而非 pyproject 锁定的 2.9.1，需先处理版本冲突（见第 2 节问题 C）；② `nvcc: command not found` 属正常——torch 自带 cu128 运行时，训练不依赖 nvcc；③ 若上面「关键 Python 包」有任一包报 `ModuleNotFoundError`，说明该包尚未安装，需在第 4 节步骤 1 用 `pip install` 补齐。

---

## 2. 局限性及解决思路

AutoDL 服务器位于中国境内，主要遇到网络与工具链两类问题。

### 问题 A：无法从 HuggingFace 下载数据

**现象：** `huggingface.co` 直连 TCP 超时（网络层阻断）；`hf-mirror.com` 镜像站大文件下载不稳定、且不支持 Datasets Server API（403）。

**两条解决思路：** 按数据集在 Gitee 是否有镜像分流——**Gitee 上有的从 Gitee 下载（直连快、无需代理），Gitee 上没有的从 hf-mirror 镜像下载**。两种方式都需改代码（见下方各自的「配套代码改动」）。

#### 思路 1：Gitee 上有镜像 → 从 Gitee 直连下载

`gitee.com/hf-datasets` 托管了部分热门数据集（SFT 用的 SmolTalk / MMLU / GSM8K 都在）。无需代理，`urllib.request.urlopen()` 直连即可，实测 **14~15 MB/s**：

| SFT 数据集 | 大小 | 耗时 |
|-----------|------|------|
| GSM8K (train+test) | 2.6MB | ~2s |
| MMLU (auxiliary_train+test) | 48.6MB | ~4s |
| SmolTalk (train 4 分片 + test) | ~925MB | ~64s |
| **合计** | **~976MB** | **~70s** |

数据下载到 `/root/.cache/nanochat/task_data/`，`load_hub_dataset()` 检测到缓存即跳过网络直接加载。

**配套代码改动：** ① 新增独立下载脚本 `download_sft_data.py`，用 `urllib.request.urlopen()` 直连 Gitee raw，把 parquet 下载到 `task_data/{slug}/{subset}/{split}/`；② 重写 `tasks/common.py` 的 `load_hub_dataset()`，使训练进程读到 `task_data/` 缓存后直接加载本地 parquet 而不再发起网络请求（避免 SFT 训练时二次联网下载）。

#### 思路 2：Gitee 上没有 → 从 hf-mirror.com 镜像下载

预训练语料 **ClimbMix-400B**（`karpathy/climbmix-400b-shuffle`，~17GB / 170 分片）在 Gitee 上没有镜像，只能走 hf-mirror.com：

```bash
wget https://hf-mirror.com/datasets/karpathy/climbmix-400b-shuffle/resolve/main/shard_00000.parquet
```

下载到 `/root/.cache/nanochat/base_data_climbmix/`（tokenizer 训练与预训练共用，见第 4 节步骤 2）。

**配套代码改动：** 原版 `nanochat/dataset.py` 的 `BASE_URL` 指向 `huggingface.co`（国内不可达），需改为 `https://hf-mirror.com/datasets/karpathy/climbmix-400b-shuffle/resolve/main`；tokenizer 训练用官方 `scripts/tok_train.py`（经 `parquets_iter_batched()` 读本地 `base_data_climbmix/`，不联网）。

#### 选型原则

Gitee 有的走 Gitee（快、稳定）；Gitee 没有的走 hf-mirror（覆盖全，但大文件不稳、不支持 Datasets Server API）。本实验实际对应：SFT 数据 → 思路 1（Gitee）；ClimbMix 预训练数据 + tokenizer → 思路 2（hf-mirror）。

### 问题 B：uv 无法使用（依赖安装卡死）

**现象：** `uv sync --extra gpu` 在 AutoDL 受限网络下并发下载**永久挂起**。

**解决思路：改用 `pip install`**：

```bash
pip install huggingface_hub tiktoken numpy rustbpe filelock kernels psutil pyarrow wandb -q
```

### 问题 C：torch 版本冲突

**现象：** AutoDL 预装 pytorch 2.8，但 `pyproject.toml` 精确锁定 `torch==2.9.1`，pip 会尝试降级或报错。

**解决思路：放宽版本约束**（`sed` 改写）：

```bash
sed -i 's/torch==2\.9\.1/torch>=2.8.0,<2.10.0/g' pyproject.toml
```

### 问题 D：SFT 训练因 ChatCORE 评估崩溃

**现象：** `--chatcore-every` 会让训练过程调用 ChatCORE 评估，其中 ARC 数据集下载失败导致训练崩溃（`URLError`）。

**解决思路：禁用 ChatCORE**，训练命令加 `--chatcore-every=-1`。

### 已排除的失败方案（备查）

huggingface.co 直连（TCP 超时）、hf-mirror Datasets Server API（403）、hf-mirror 大文件下载（超时/Reset）、`huggingface-cli`（已废弃）、`sshpass`（Windows 无）、paramiko `exec_command` 实时输出（缓冲问题）。

---

## 3. 实验基本流程、参数设置与最终结果

### 基本流程

```
hf-mirror 下载 ClimbMix-400B 数据 → 训练 Tokenizer(32768) → d4 预训练(500步) → SFT 微调(完整1 epoch，数据走 Gitee) → chat_cli 推理/评估
```

### 各阶段参数

**Tokenizer：** 词表 32768，训练语料 2B 字符（ClimbMix-400B 前 8 个分片，`doc_cap=10000`）。

**预训练（d4）：**
```bash
python -m scripts.base_train --depth=4 --device-batch-size=8 \
  --num-iterations=500 --run="dummy" --save-every=500 --model-tag="d4"
```
- 模型：4 层 / 256 维 / 2 头 / n_kv_head=2 / window_pattern=SSSL
- 500 步，检查点 `base_checkpoints/d4/model_000500.pt`

**SFT 微调：**
```bash
python -m scripts.chat_sft \
  --run="dummy" --device-batch-size=4 \
  --num-iterations=-1 --model-tag="d4" --chatcore-every=-1
```
- 数据混合：SmolTalk (460K) + MMLU×3 (300K) + GSM8K×4 (32K) = 789,759 行
- 总 batch 262,144 token/步，梯度累积 32；完整 1 epoch = 1868 步，约 25 分钟
- 学习率继承预训练（embedding_lr=0.3, unembedding_lr=0.008, matrix_lr=0.02），warmup 0%，warmdown 50%，final_lr 0
- 优化器 MuonAdamW

### 最终结果

| 指标 | 值 | 说明 |
|------|-----|------|
| 最终验证 BPB | **0.5576** | 预训练基线 1.145 |
| MMLU（50 题样本） | 34.0% | 随机基线 25% |
| GSM8K（50 题样本） | 0.0% | 数学推理超出 4 层模型能力 |
| 参数量 | 36.70M | fp32，检查点 92MB |

**启动 chat_cli 对话（在服务器上）：**

```bash
python -m scripts.chat_cli -g d4 -p "What is the capital of France?"
```

**输入问题得到的输出（示例）：**
```
What is the capital of France?
The capital of France is Paris.
```
> 注：d4 是教学级玩具模型，回答倾向重复/冗长，且仅能生成英文（tokenizer 在英文数据上训练）。模型应能回答法国巴黎、天空是蓝色等常识问题。

---

## 4. 各步骤如何执行与预期结果

六个脚本按依赖顺序依次执行，每步都应在服务器 SSH 会话中运行。先上传脚本（含检查脚本）：

```bash
# 从 runs-with-autodl 目录执行（Windows 用 PowerShell 亦同）
scp -P 49302 check_server.sh 01_setup.sh 02_data_tokenizer.sh \
    03_pretrain.sh 04_sft.sh 05_eval.sh \
    root@connect.westc.seetacloud.com:/root/
```

**关于修改版代码要不要上传：**

新 clone 的 nanochat，各步骤对 `src/` 下修改版代码的需求如下：

| 修改版文件 | 用于步骤 | 是否必须上传 |
|-----------|---------|------------|
| `tasks/common.py` | 步骤 4（SFT 数据下载） | **必须**——原版走 HF API 会 403，新版读本地缓存 |
| `download_sft_data.py` | 步骤 4（可选：Gitee 预下载 SFT 数据） | 用了才要 |
| `dataset.py`（BASE_URL→hf-mirror） | 仅备查 | 复现流程用不到（数据用脚本内 `wget` 下载） |

其中 **`tasks/common.py` 是硬性要求**。`04_sft.sh` 运行时会自动从 `/root/tasks_common.py` 覆盖到 nanochat 仓库（无需手动操作），因此只要先把这个文件上传到服务器即可：

```bash
# 把修改版 common.py 上传为 /root/tasks_common.py（04 脚本会自动复制覆盖）
scp -P 49302 ./src/tasks/common.py root@connect.westc.seetacloud.com:/root/tasks_common.py

# （可选）Gitee 预下载 SFT 数据的脚本
scp -P 49302 ./src/download_sft_data.py root@connect.westc.seetacloud.com:/root/download_sft_data.py
```

> 每个步骤都写了「如何验证」：用 `[ -s 文件 ]` 判定文件存在且非空、`ls -la` 看大小、读 `meta_*.json` 看指标。产物路径统一在 `/root/.cache/nanochat/`（数据盘，持久化）。
>
> **执行注意：** 直接用脚本即可（脚本内已 `source` 并激活 conda）。只有在交互式 shell 里手动敲 `python` 时，才需要先 `source /root/miniconda3/etc/profile.d/conda.sh && conda activate base`，否则会 command not found。

### 步骤 0：检查服务器环境（必须最先做）

**目的：** 复现前先确认服务器系统 / Python / GPU / CUDA / 依赖是否满足训练要求，并暴露版本冲突（torch）与缺失依赖。**本步只查看、不安装任何东西**，确认无误再进入步骤 1。对应第 1 节脚本 `check_server.sh`。

**如何做：**
```bash
bash /root/check_server.sh
```

**预期结果（以实例实际为准，key 点）：**
- 系统 Ubuntu 22.04；GPU 为 RTX 4090 D 24GB
- Python 由 miniconda 提供（`/root/miniconda3/bin/python`）
- `torch` 存在、`CUDA 可用: True`、GPU 名正确
- `huggingface_hub / tiktoken / numpy / rustbpe / filelock / pyarrow` 有一行结果（含已装版本或 `MISSING`）

**如何验证：**
```bash
# 1) 脚本是否各段都输出了信息（没有 command not found 才算完整）
bash /root/check_server.sh 2>&1 | grep -E "ERROR|not found" || echo "check OK"

# 2) 关键项一票否决：torch + CUDA 必须可用
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
python -c "import torch; print(torch.cuda.is_available())"

# 3) 确认缺失的包（若有 MISSING，记下来，步骤 1 的 pip 会补齐）
bash /root/check_server.sh 2>&1 | grep -i "MISSING|ModuleNotFoundError" || echo "no missing deps"

# 4) 镜像 torch 版本是否与 pyproject 目标一致（若不一致，见第 2 节问题 C）
```

> 常见差异举例：`nvcc: command not found` 属正常（torch 自带 cu128 运行时，训练不依赖 nvcc）。若 `python` 也报 not found，说明当前 shell 未激活 conda——`check_server.sh` 及 01~05 脚本内部都已 source 并激活 conda，直接用脚本即可；只有交互式手敲 `python` 前需先 `source /root/miniconda3/etc/profile.d/conda.sh && conda activate base`。

### 步骤 1：环境配置

**目的：** 把 nanochat 代码克隆到服务器，放宽 torch 版本约束（预装 2.8 vs pyproject 锁 2.9.1），并用 pip 装齐数据/训练所需依赖。**本步不下载任何数据。**

**依赖：** 无。仅依赖网络（GitHub 克隆代码）与 AutoDL 预装的 miniconda/torch。是后续所有步骤的前提。

**如何做：**
```bash
ssh -p 49302 root@connect.westc.seetacloud.com
bash /root/01_setup.sh
```

**预期结果：** 代码在 `/root/autodl-tmp/nanochat/`；`pyproject.toml` 中 torch 约束变为 `>=2.8.0,<2.10.0`；依赖 pip 装好；脚本末尾打印 `CUDA available: True`。

**如何验证：**
```bash
# 1) CUDA 是否可用（最关键）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2) 代码是否在
ls /root/autodl-tmp/nanochat/pyproject.toml

# 3) torch 约束是否已放宽
grep -n "torch" /root/autodl-tmp/nanochat/pyproject.toml

# 4) 依赖是否装齐（无 ModuleNotFoundError 即 OK）
python -c "import huggingface_hub, tiktoken, numpy, rustbpe, filelock, pyarrow; print('deps OK')"
```

**耗时：** 3~5 分钟。

### 步骤 2：准备 ClimbMix 数据 + 训练 Tokenizer

**目的：** 两个子任务——① 准备 ClimbMix-400B 预训练分片（`/root/.cache/nanochat/base_data_climbmix/`，新 clone 后为空，需下载 ≥8 个 shard，共 ~700MB）；② 用 ClimbMix 文本训练出词表 32768 的 tokenizer。全程不访问 huggingface.co。

**依赖：**
- **步骤 1 产出**：`/root/autodl-tmp/nanochat/` 代码（跑 `tok_train.py`）+ 已装依赖（`pyarrow` 读 parquet、`rustbpe` 训 BPE）。
- **输入数据源**（本步要拉取的外部数据，非前序产物）：ClimbMix-400B 托管在 HuggingFace 仓库 `karpathy/climbmix-400b-shuffle`（分片名 `shard_00000.parquet`、`shard_00001.parquet`……）。huggingface.co 国内不可达，走 hf-mirror.com 镜像：
  ```bash
  wget https://hf-mirror.com/datasets/karpathy/climbmix-400b-shuffle/resolve/main/shard_00000.parquet
  ```

**如何做：** 上述下载已封装进脚本。`02_data_tokenizer.sh` 会检查 `base_data_climbmix/`，若不足 8 个分片则自动逐片 `wget` 下载 `shard_00000~00007` 到该目录，然后调用官方 `tok_train.py` 训练 tokenizer。产物 `/root/.cache/nanochat/tokenizer/` 供步骤 3、4 复用。

```bash
bash /root/02_data_tokenizer.sh
```

**预期结果：** 脚本跑到底，得到两份产物：① `base_data_climbmix/` 有 ≥8 个 parquet 分片（共 ~700MB，脚本会打印「ClimbMix 分片：N 个，体积」）；② `tok_train.py` 训练日志显示词表 `32768`，`/root/.cache/nanochat/tokenizer/` 生成非空的 `tokenizer.pkl` 与 `token_bytes.pt`。

**如何验证：**
```bash
# 1) ClimbMix 分片是否就绪（≥8 个，共 ~700MB）
ls -la /root/.cache/nanochat/base_data_climbmix/ | head
[ $(ls /root/.cache/nanochat/base_data_climbmix/*.parquet | wc -l) -ge 8 ] && echo "分片 OK" || echo "分片不足"

# 2) tokenizer 是否生成且非空
ls -la /root/.cache/nanochat/tokenizer/          # 应有 tokenizer.pkl、token_bytes.pt，且大小非 0

# 3) tokenizer 能否真正加载、词表是否 32768
python -c "from nanochat.tokenizer import RustBPETokenizer; t=RustBPETokenizer.from_directory('/root/.cache/nanochat/tokenizer'); print('vocab =', t.get_vocab_size())"
```

**耗时：** 数据已就绪时约 2 分钟（tokenizer 训练 ~77s）；需补下载时视网速增加。

### 步骤 3：Toy 预训练（d4）

**目的：** 从零预训练 4 层玩具模型 500 步，产出 base 检查点，作为 SFT 的起点。脚本开头检测到已有 `model_000500.pt` 会自动跳过（幂等）。

**依赖：** 依赖步骤 2 的产出——① 预训练语料 `base_data_climbmix/`（≥8 个 parquet 分片，`base_train.py` 内部经 `parquets_iter_batched` 读取）；② 词表 32768 的 tokenizer（`/root/.cache/nanochat/tokenizer/`，base 结构需它做 tokenize）。

**如何做：**
```bash
bash /root/03_pretrain.sh
```

**预期结果：** 最终检查点写入 `/root/.cache/nanochat/base_checkpoints/d4/`——`model_000500.pt`（数十 MB）+ `meta_000500.json`；base 的 `val_bpb ≈ 1.145`（8 个 ClimbMix shard 的结果）。

**如何验证：**
```bash
# 1) 检查点是否生成且非空（model 应有数十 MB）
ls -la /root/.cache/nanochat/base_checkpoints/d4/
[ -s /root/.cache/nanochat/base_checkpoints/d4/model_000500.pt ] && echo "model OK" || echo "model 缺失"

# 2) 训练指标是否合理（看 meta 里的最终 val_bpb，8-shard 复现约 1.145）
python -c "import json; d=json.load(open('/root/.cache/nanochat/base_checkpoints/d4/meta_000500.json')); print('val_bpb =', d.get('val_bpb'))"
```

**耗时：** 约 4 分钟。

### 步骤 4：SFT 微调

**目的：** 在步骤 3 的 base 模型上，用 SFT 混合数据（SmolTalk + MMLU×3 + GSM8K×4 = 789,759 行，由 `chat_sft.py` 内部经修改后的 `tasks/common.py` 从 hf-mirror/Gitee 下载）做 SFT，完整跑 1 epoch（1868 步），产出最终对话模型。脚本开头检测到已有 `model_001868.pt` 会自动跳过（幂等）。

**依赖：** 依赖步骤 3 的产出——`base_checkpoints/d4/model_000500.pt`（SFT 初始权重）；隐式复用步骤 2 的 `tokenizer/` 与步骤 1 的代码/依赖。SFT 数据（SmolTalk/MMLU/GSM8K）由本步 `tasks/common.py` 下载，**不是**前序步骤的产物。必须先完成步骤 1~3。

**如何做：**
```bash
bash /root/04_sft.sh
```

**预期结果：** 最终检查点写入 `/root/.cache/nanochat/chatsft_checkpoints/d4/`——`model_001868.pt`（96MB）+ `meta_001868.json`，其中 `val_bpb = 0.5576`（远低于 base 的 1.145）。

**如何验证：**
```bash
# 1) 最终检查点是否生成且非空（model 应有 ~92MB）
ls -la /root/.cache/nanochat/chatsft_checkpoints/d4/
[ -s /root/.cache/nanochat/chatsft_checkpoints/d4/model_001868.pt ] && echo "model OK" || echo "model 缺失"

# 2) 指标是否达标（val_bpb 应约 0.5576，远低于 base 的 1.145）
python -c "import json; d=json.load(open('/root/.cache/nanochat/chatsft_checkpoints/d4/meta_001868.json')); print('val_bpb =', d.get('val_bpb'))"
```

**耗时：** 约 25 分钟。
**注意：** 脚本已带 `--chatcore-every=-1`（跳过 ARC 下载崩溃）与 `--num-iterations=-1`（跑完整 epoch）。

### 步骤 5：推理 + 评估

**目的：** 验证最终模型可用——先对话看回答质量，再跑 MMLU / GSM8K（各 50 题样本）量化能力。

**依赖：** 依赖步骤 4 的产出——`chatsft_checkpoints/d4/model_001868.pt`（对话与评估的权重）；评估所用的 MMLU / GSM8K 测试数据由 `chat_eval.py` 从 gated-HF/hf-mirror 下载，**不是**前序步骤的产物。

**如何做：**
```bash
bash /root/05_eval.sh
```

**预期结果：** 对话能回答常识问题（如法国首都巴黎）；评估行输出 `MMLU` ≈ 34.0%、`GSM8K` ≈ 0.0%。

**如何验证：**
```bash
# 1) 交互式对话（仅英文，空行退出）看回答是否合理
python -m scripts.chat_cli -g d4

# 2) 复跑评估，看 stdout 里的 MMLU / GSM8K 分数行
python -m scripts.chat_eval -i sft -g d4 -a "MMLU|GSM8K" -x 50
```

**耗时：** 对话即时，评估 1~2 分钟。

### 完整对照表

| 步骤 | 脚本 | 目的 | 产出 | 验证命令 | 耗时 |
|------|------|------|------|---------|------|
| 0 | `check_server.sh` | 检查环境 | 环境报告 | torch+CUDA 可用；无 MISSING 依赖 | 即时 |
| 1 | `01_setup.sh` | 环境配置 | 代码 + 依赖 | `torch.cuda.is_available()` 打印 True | 3~5 min |
| 2 | `02_data_tokenizer.sh` | ClimbMix 数据 + tokenizer | base_data_climbmix ≥8 分片；tokenizer/ 两文件 | `ls 分片数`；tokenizer 词表 32768 | ~2 min |
| 3 | `03_pretrain.sh` | d4 预训练 | model_000500.pt | `[ -s model ]`；meta 的 val_bpb 下降 | ~4 min |
| 4 | `04_sft.sh` | SFT 微调 | model_001868.pt + meta | `[ -s model ]`；val_bpb≈0.5576 | ~25 min |
| 5 | `05_eval.sh` | 对话 + 评估 | 输出结果 | 对话合理；MMLU 34% / GSM8K 0% | 1~2 min |

---

## 5. 将服务器产物拷贝到本地 + 本地启动模型

### 5.1 拷贝模型、tokenizer 与修改后的代码到本地

在本地（Windows PowerShell）执行：

```powershell
# 模型权重 + 元数据 → 放到 data/
scp -P 49302 root@connect.westc.seetacloud.com:/root/.cache/nanochat/chatsft_checkpoints/d4/model_001868.pt data\
scp -P 49302 root@connect.westc.seetacloud.com:/root/.cache/nanochat/chatsft_checkpoints/d4/meta_001868.json data\

# ⚠️ 分词模型（必须！模型词表 32768 依赖它，本地启动 chat_cli 需要）
# 服务器产物：/root/.cache/nanochat/tokenizer/tokenizer.pkl + token_bytes.pt
scp -P 49302 root@connect.westc.seetacloud.com:/root/.cache/nanochat/tokenizer/tokenizer.pkl data\
scp -P 49302 root@connect.westc.seetacloud.com:/root/.cache/nanochat/tokenizer/token_bytes.pt data\

# 修改后的代码 → 放到 src/（tasks/common.py + pyproject.toml + uv.lock）
scp -P 49302 root@connect.westc.seetacloud.com:/root/autodl-tmp/nanochat/tasks/common.py src\tasks\
scp -P 49302 root@connect.westc.seetacloud.com:/root/autodl-tmp/nanochat/pyproject.toml src\
scp -P 49302 root@connect.westc.seetacloud.com:/root/autodl-tmp/nanochat/uv.lock src\
```

> Windows 下 `scp` 目标路径要用反斜杠（`data\`），源路径用冒号分隔主机与远程路径。若提示 SSH key，加 `-i "$env:USERPROFILE\.ssh\id_rsa_autodl"`。

### 5.2 本地启动该模型（CPU）

需要 nanochat 仓库 + **tokenizer + 模型权重** 都已放到本地。本地已有 `nanochat/.venv`（CPU 版依赖）：

```powershell
# 配置：把 tokenizer 与模型放好，运行 chat_cli 指向 d4
cd nanochat
# 若模型文件在默认 base_dir 下，会自动加载；否则用 -g 指定路径
.\.venv\Scripts\python.exe -m scripts.chat_cli -g d4 -p "What color is the sky?"
```

**预期输出：** 加载 tokenizer（词表 32768）与 d4 模型后回答常识问题（示例：`The sky is blue.`）。`data/` 下 `tokenizer.pkl`、`token_bytes.pt`、`model_001868.pt` 均已就位（见「附」文件表），可直接运行。

> 局限：CPU 推理较慢；模型仅支持英文；PowerShell 终端显示 UTF-8 输出乱码时，先执行 `chcp 65001` + `[Console]::OutputEncoding=[Text.Encoding]::UTF8`，或改用 VSCode 集成终端。

---

## 6. 易错问题及解决思路

| 问题 | 原因 | 解决 |
|------|------|------|
| huggingface.co 连不上 | AutoDL 网络阻断 HF | 用 Gitee 镜像下载数据 |
| 下载数据集 403 | 镜像站不支持 Datasets Server API | 用 `huggingface_hub` 的 `list_repo_files()` 或 Gitee 镜像 |
| `uv sync --extra gpu` 卡死 | uv 并发下载在受限网络挂起 | 改用 `pip install` |
| torch 版本冲突 | 预装 torch 2.8 > pyproject 上限 | `sed` 放宽版本约束 |
| ChatCORE 评估崩溃 | ARC 数据集无法下载 | SFT 加 `--chatcore-every=-1` |
| SFT 只跑几步就停 | `--num-iterations` 数的是 micro-batch 而非优化步 | 用 `--num-iterations=-1` 跑完整 epoch，或设 `N×grad_accum` |
| SSH 连上后卡死 | 网络专线不稳定 | 重连，或改用网页 Terminal |
| 找不到已下载的数据 | 缓存路径不对 | 检查 `/root/.cache/nanochat/`（数据盘持久化） |
| Windows 无 sshpass | 平台限制 | 用 paramiko 或 SSH key 免密登录 |
| paramiko 远程脚本无输出 | channel 缓冲问题 | 改用 `ssh.exe` + SSH key |
| PowerShell 中文/输出乱码 | 控制台 GBK 编码 | `chcp 65001` + 设置 UTF-8 输出，或用 VSCode 终端 |
| 模型答非所问/只会英文 | 4 层玩具模型 + 英文语料 | 属正常现象，仅作教学验证 |

---

## 7. 与 `runs/speedrun.sh` 的区别

本实验的六个脚本（`check_server.sh` + `01~05_*.sh`）是 nanochat 官方一键脚本 [`runs/speedrun.sh`](../../runs/speedrun.sh) 针对 **4090 单卡 + AutoDL 国内网络** 的适配版。两者跑的是同一条流水线（数据 → tokenizer → 预训练 → SFT → 评估/对话，核心命令都是 `python -m scripts.{tok_train,base_train,chat_sft,chat_eval}`），但环境与规模不同，脚本做了如下差异调整：

| 项目 | `runs/speedrun.sh`（官方） | 本实验（runs-with-autodl） |
|------|---------------------------|---------------------------|
| 硬件 | 8×H100，`torchrun --nproc_per_node=8` 分布式 | 单张 RTX 4090D，单进程 |
| 模型 | d24（24 层）、`--fp8`、`--target-param-data-ratio=8` | d4（4 层 / 256 维 / 2 头）、500 步 |
| 数据量 | `nanochat.dataset -n 170`（GPT-2 级需 170+ shards） | 8 个 ClimbMix shard（~700MB，tokenizer + 玩具预训练够用） |
| 数据下载 | 官方 `nanochat.dataset` 直连 HuggingFace | `wget` 走 hf-mirror.com / Gitee（HF 国内不可达） |
| 环境安装 | 装 uv → `uv sync --extra gpu` | `pip install`（AutoDL 下 uv 卡死）+ `sed` 放宽 torch 约束（预装 2.8 vs pyproject 锁 2.9.1） |
| 评估 | `base_eval`（CORE 指标）+ `chat_eval` 全量 | `chat_eval -x 50`（各 50 题样本）+ 对话 |
| 目的 | 训练 GPT-2 级模型上排行榜 | 验证全流程 + 教学复现（约 30 分钟跑完） |

本实验额外增加的工程处理（speedrun.sh 没有）：

- **幂等跳过**：`03_pretrain.sh` / `04_sft.sh` 开头检测产物检查点已存在则直接跳过，脚本可安全重复执行。
- **跳过 ARC 下载崩溃**：SFT 加 `--chatcore-every=-1`，否则 ChatCORE 评估要下载 ARC 数据集而崩溃。
- **`--num-iterations=-1`**：SFT 跑完整 epoch（1868 步），默认值数的是 micro-batch 而非优化步，会提前停下。
- **`tasks/common.py` 覆盖**：原版走 HF Datasets API（`huggingface.co/api/...`）会 403，修改版读本地缓存；`04_sft.sh` 运行时自动从 `/root/tasks_common.py` 覆盖。
- **逐步验证**：speedrun.sh 是一条龙跑完；本实验拆成 6 个脚本、每步带「如何验证」，产物路径统一在 `/root/.cache/nanochat/`。

---

## 附：本目录文件与对应关系

| 本地文件 | 说明 |
|---------|------|
| `data/model_001868.pt` | SFT 最终模型权重（92MB） |
| `data/meta_001868.json` | 训练元数据 + 模型配置 |
| `data/tokenizer.pkl` | **分词模型（词表 32768）— 已从服务器拷贝** |
| `data/token_bytes.pt` | token 字节映射（BPB 评估用）— 已从服务器拷贝 |
| `src/pyproject.toml` | 放宽 torch 后的版本 |
| `src/tasks/common.py` | 改写后的 `load_hub_dataset`（huggingface_hub 方案，读本地缓存不再联网） |
| `src/download_sft_data.py` | Gitee 直连下载脚本（思路 1 配套） |
| `src/dataset.py` | `BASE_URL` 改为 hf-mirror 的修改版（思路 2 配套） |
| `src/uv.lock` | 更新后的依赖锁 |
| `01~05_*.sh` | 复现脚本 |