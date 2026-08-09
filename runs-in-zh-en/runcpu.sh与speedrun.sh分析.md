# runcpu.sh 与 speedrun.sh 分析

> **目的**：nanochat 官方提供两条开箱即用的训练流水线脚本——`runs/runcpu.sh`（单机 CPU/Mac 教学演示）与 `runs/speedrun.sh`（8×H100 GPU 全量实战）。本文逐项对比两者的**目标硬件、训练规模、参数设置与执行步骤**，为选择「该用哪条线复现/改造」提供依据。

---

## 1. 一句话对比（结论先行）

| 项目 | `runcpu.sh` | `speedrun.sh` |
|------|-----------|---------------|
| 定位 | 教学 / 演示，跑通代码路径 | 正式练兵，训练 GPT-2 级模型 |
| 目标硬件 | CPU / Mac（M3 Max 实测） | **8×H100** GPU 节点 |
| 运行方式 | 单进程 `python -m ...` | `torchrun --standalone --nproc_per_node=8` 分布式 |
| 依赖安装 | `uv sync --extra cpu` | `uv sync --extra gpu`（+ `--fp8`） |
| 全程耗时 | 预训练 ~30 min + SFT ~10 min | 约 **1.5 小时**（8 卡并行） |

> 共同点：两条流水线的**步骤顺序完全一致**——`下载数据(8 shard) → 训练 tokenizer → base_train → base_eval → chat_sft → chat_cli / chat_eval`，核心命令都是 `python -m scripts.{tok_train, base_train, chat_sft, ...}`,且都用 `NANOCHAT_BASE_DIR` 指定缓存目录。

---

## 2. 目标硬件与安装差异

### 2.1 runcpu.sh（CPU / Mac）

- 面向无 GPU 的场景（Intel CPU / Apple Silicon MacBook，实测 M3 Max）。
- 脚本头部注释明确：*"Training LLMs requires GPU compute and $$$. You will not get far on your Macbook. Think of this run as educational/fun demo."*
- `uv sync --extra cpu`：装 CPU 版 PyTorch，无 CUDA。
- 全部命令单进程 `python -m` 直接跑。

### 2.2 speedrun.sh（8×H100）

- 面向 **8 卡 GPU 节点**（脚本注释：*"designed to run on a blank 8XH100 GPU node"*）。
- `uv sync --extra gpu`：装 CUDA 版 PyTorch。
- 训练/评估用 `torchrun --standalone --nproc_per_node=8` 做数据并行 DDP。

---

## 3. 模型规模与训练参数

### 3.1 分词器（tokenizer）——两者一致

- 都 `python -m nanochat.dataset -n 8` 下载前 8 个 ClimbMix shard（~2B 字符），再 `python -m scripts.tok_train`（runcpu 显式加 `--max-chars=2000000000`）。
- 然后 `python -m scripts.tok_eval` 评估压缩率。

### 3.2 runcpu 的 base_train 参数

```bash
python -m scripts.base_train \
    --depth=6 --head-dim=64 --window-pattern=L --max-seq-len=512 \
    --device-batch-size=32 --total-batch-size=16384 \
    --eval-every=100 --eval-tokens=524288 \
    --core-metric-every=-1 --sample-every=100 \
    --num-iterations=5000 --run=$WANDB_RUN
```

- **6 层 / head-dim 64 / 512 序列**的小模型，普通精度（fp32）。
- `--core-metric-every=-1`：**跳过 CORE 评估**（CORE 是英文 ICL 基准，CPU 上意义不大且慢）。
- `--sample-every=100`：每 100 步抽样看生成内容。
- 目标 ~30 分钟跑完 5000 步（教学级，调大迭代可提质）。

### 3.3 speedrun 的 base_train 参数

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 \
    --run=$WANDB_RUN
```

- **24 层 / fp8 / target-param-data-ratio=8**（稍微欠训练，因为目标就是跑赢 GPT-2）。

- 评估是 `base_eval` 全量（含 CORE 指标 + 采样），未设 `-1` 跳过。

---

## 4. 数据下载策略差异

| 步骤 | runcpu | speedrun |
|------|--------|----------|
| 分词器前 | `-n 8` 同步下载 8 shard | 同样先 `-n 8` |
| 预训练前 | **复用前 8 个 shard**（玩具规模够用） | 再**后台 `-n 170` 补足**（约 150 shard 够 GPT-2，+20 余量），`wait $DATASET_DOWNLOAD_PID` 等下载完成才开始训练 |
| 数据总量 | ~800MB | ~800MB + ~17GB |

> speedrun 在代码注释里说得很清楚：每个 shard ~100MB 压缩 / ~250M 字符，170 shard 合计约 17GB，是 GPT-2 级预训练的**最低配置**；runcpu 是「玩具规模」，8 片即可。

---

## 5. SFT 与收尾

| 步骤 | runcpu.sh | speedrun.sh |
|------|-----------|-------------|
| SFT | `python -m scripts.chat_sft --eval-every=200 --eval-tokens=524288 --num-iterations=1500 --run=$WANDB_RUN`（~10 min） | `torchrun ... -m scripts.chat_sft -- --run=$WANDB_RUN`（用默认迭代数） |
| 评估 | 不跑 chat_eval（注释掉） | `torchrun ... -m scripts.chat_eval -- -i sft` |
| 对话 | `chat_cli -p "What is the capital of France?"`（注释示例） | `chat_cli` 默认参数（注释下也有示例） |
| wandb | 默认 `WANDB_RUN=dummy`，可手动设 | 同样默认 dummy，且支持 `WANDB_RUN` 环境变量覆盖 |

---

## 6. 结论 / 选择建议

| 场景 | 选哪个 |
|------|--------|
| 本机（无 GPU）快速验证、教学跑通链路 | **runcpu.sh** |
| 国内云 GPU 单卡 / 小卡复现 | 参照 `runs-in-zh-en` 改编 rucpu（缩小迭代数、换 ModelScope 数据源） |
| 云上 8×H100 真练兵、刷榜 | **speedrun.sh** |
| 分布式训练上手 | speedrun（内含 torchrun DDP 用法） |