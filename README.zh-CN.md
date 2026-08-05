# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat 是用于训练大语言模型（LLM）最简单的实验工具。它可以在单张 GPU 节点上运行，代码极少且易于修改，并覆盖了 LLM 的所有主要阶段，包括分词、预训练、微调、评估和推理。例如，你可以自行训练一个 GPT-2 水平的 LLM（2019 年训练成本约 4.3 万美元），现在只需 48 美元（约 2 小时的 8 卡 XH100 GPU 节点），并能通过简单的命令行与它对话。如果使用竞价实例，总成本可低至约 15 美元。更普遍地，nanochat 开箱即可通过设置一个单一的复杂度旋钮 `--depth`（GPT 转换器模型的层数，GPT-2 水平恰好大约为 depth 26）训练一整套计算最优的模型系列。其他所有超参数（转换器的宽度、注意力头数、学习率调整、训练时长、权重衰减……）都会自动以最优方式计算。

关于本仓库的问题，我推荐使用 [DeepWiki](https://deepwiki.com/karpathy/nanochat)（来自 Devin/Cognition）提问，或者使用 [Discussions 板块](https://github.com/karpathy/nanochat/discussions)，或到 Discord 上的 [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) 频道交流。

## Time-to-GPT-2 排行榜

目前开发的主要重点是调优预训练阶段，它消耗最多的算力。受 modded-nanogpt 仓库的启发，为激励进展和社区协作，nanochat 维护了一个 "GPT-2 速通" 排行榜，即把 nanochat 模型训练到 GPT-2 水平（以 DCLM CORE 分数衡量）所需的墙上时间。[runs/speedrun.sh](runs/speedrun.sh) 脚本始终反映了训练 GPT-2 水平模型并与之对话的参考方法。当前排行榜如下：

| # | 时间 | val_bpb | CORE | 描述 | 日期 | 提交 | 贡献者 |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 小时 | - | 0.2565 | 原始的 OpenAI GPT-2 检查点 | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 基线，略微过度训练 | 2026年1月29日 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 略微欠训练 **+fp8** | 2026年2月2日 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | 将总批大小提升到 1M 个 token | 2026年2月5日 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | 更换数据集为 NVIDIA ClimbMix | 2026年3月4日 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | 自动研究 [第1轮](https://x.com/karpathy/status/2031135152349524125) | 2026年3月9日 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | 自动研究第 2 轮 | 2026年3月14日 | a825e63 | @karpathy |

我们关心的主要指标是 "time to GPT-2"——在 8 卡 XH100 GPU 节点上超过 GPT-2（1.6B）CORE 指标所需的墙上时间。GPT-2 的 CORE 分数为 0.256525。在 2019 年，GPT-2 的训练成本约 4.3 万美元，因此令人难以置信的是，得益于过去 7 年从底层到上层的诸多进步，我们现在能以更快的速度、不到 100 美元（例如按照当前约 3 美元/GPU/小时的价格，一个 8 卡 XH100 节点约 24 美元/小时，那么 2 小时约 48 美元）做到这一点。

更多关于如何解读和贡献排行榜的文档，请参阅 [dev/LEADERBOARD.md](dev/LEADERBOARD.md)。

## 快速开始

### 环境搭建

nanochat 使用 [uv](https://docs.astral.sh/uv/) 进行依赖管理。安装方式：

```bash
uv sync --extra gpu    # 适用于 CUDA（A100/H100 等）
uv sync --extra cpu    #（或）适用于仅 CPU / MPS
source .venv/bin/activate
```

用于开发（额外安装 pytest、matplotlib、ipykernel、transformers 等）：

```bash
uv sync --extra gpu --group dev
```

### 复现并对话 GPT-2

最有乐趣的事情就是训练你自己的 GPT-2 并与之对话。整个流程都包含在单个文件 [runs/speedrun.sh](runs/speedrun.sh) 中，它专为 8 卡 XH100 GPU 节点设计。从你常用的供应商那里租一台新的 8 卡 XH100 GPU 机器（例如我使用并喜欢 [Lambda](https://lambda.ai/service/gpu-cloud)），然后启动训练脚本：

```bash
bash runs/speedrun.sh
```

你可能希望在 screen 会话中运行它，因为这大约需要 1.5 小时。完成后，你可以通过命令行与你的模型对话。请再次确保你本地的 uv 虚拟环境已激活（运行 `source .venv/bin/activate`），然后对话：

```bash
python -m scripts.chat_cli
```

让它写故事或诗歌。让它告诉你"你是谁"以观察幻觉。问它天空为什么是蓝色的，或者为什么是绿色的。speedrun 是一个 4e19 FLOPs 能力的模型，所以有点像和幼儿园小朋友对话 :)。以下是和一个 speedrun 模型的示例对话：

> **用户：** 你好，怎么了？
>
> **助手：** 你好！今天我能帮你什么吗？
>
> **用户：** 天空为什么是蓝色的？
>
> **助手：** 天空之所以是蓝色的，是因为一种叫做"光的瑞利散射"的光学现象，它发生在阳光穿过空气中微小颗粒的时候。这些颗粒由微小的蓝色和紫色粒子组成，它们导致光线以特定方式发生弯曲。
>
> **用户：** 太酷了。写一首关于它的诗。
>
> **助手：** 在暮光的余晖中，天空是蓝色的，<br>
> 一片闪烁的蓝色画布，天空在这里展开，<br>
> 一个蓝色的世界，蓝与更多蓝，<br>
> 编织成一幅如此精美的挂毯，<br>
> 每一种色调似乎在吟唱一个故事。<br>
> ...

更多说明：

- 该代码在 Ampere 8 卡 A100 GPU 节点上也能正常运行，只是稍微慢一些。
- 所有代码在单张 GPU 上通过省略 `torchrun` 也能正常运行，并会产生几乎相同的结果（代码会自动切换为梯度累积），但你得等待 8 倍的时间。
- 如果你的 GPU 显存低于 80GB，你需要调整部分超参数，否则会 OOM / 耗尽显存。请在脚本中查找 `--device-batch-size`，并将其调低直到能放下。例如从默认的 32 降到 16、8、4、2，甚至 1。如果还更低，你就需要更懂行、更有创造力了。
- 大部分代码都是相当常规的 PyTorch，因此应该能在任何支持它的平台上运行——xpu、mps 等——但我没有亲自测试过所有这些代码路径，所以可能存在一些边缘问题。

## 研究

如果你是希望帮助改进 nanochat 的研究者，有两个值得关注的脚本：[runs/scaling_laws.sh](runs/scaling_laws.sh) 和 [runs/miniseries.sh](runs/miniseries.sh)。相关文档见 [1月7日 miniseries v1](https://github.com/karpathy/nanochat/discussions/420)。对于快速实验（约 5 分钟的预训练运行），我最喜欢的是训练一个 12 层的模型（GPT-1 大小），例如这样：

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

这会使用 wandb（运行名 "d12"），只在最后一步运行 CORE 指标，并且不采样、不保存中间检查点。我喜欢改动一些代码，重新运行一个 d12（或 d16 等），看在迭代循环中是否有帮助。判断一次运行是否有帮助，我喜欢监控 wandb 图表中的：

1. `val_bpb`（以字节比特的词汇量不变单位衡量的验证损失），作为 `step`、`total_training_time` 和 `total_training_flops` 的函数。
2. `core_metric`（DCLM CORE 分数）
3. VRAM 利用率、`train/mfu`（模型 FLOPS 利用率）、`train/tok_per_sec`（训练吞吐量）

示例见 [这里](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044)。

需要说明的重要一点是，nanochat 是围绕单一复杂度旋钮——转换器的深度——编写和配置的。这一个整数会自动确定所有其他超参数（转换器的宽度、注意力头数、学习率调整、训练时长、权重衰减……），从而使训练出的模型达到计算最优。其理念是用户无需考虑或设置任何这些参数，只需要通过 `--depth` 请求一个更小或更大的模型，一切"自动运转"。通过扫掠深度，你就能获得不同规模的 nanochat 计算最优模型系列。GPT-2 水平模型（目前最受关注）在当前代码下恰好落在 d24–d26 区间。不过，任何对仓库的候选改动都必须足够有原则，使其适用于所有 depth 设置。

## 在 CPU / MPS 上运行

脚本 [runs/runcpu.sh](runs/runcpu.sh) 展示了在 CPU 或 Apple Silicon 上运行的非常简单的示例。它会大幅缩小正在训练的 LLM，以便在几十分钟内合理完成训练。这种方式不会得到很强壮的结果。

## 精度 / dtype

nanochat 不使用 `torch.amp.autocast`。相反，精度通过一个全局的 `COMPUTE_DTYPE`（定义在 `nanochat/common.py`）显式管理。默认情况下，它会根据你的硬件自动检测：

| 硬件 | 默认 dtype | 原因 |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | 原生的 bf16 张量核心 |
| CUDA SM < 80 (V100, T4, ...) | `float32` | 无 bf16；可通过 `NANOCHAT_DTYPE=float16` 使用 fp16（使用 GradScaler） |
| CPU / MPS | `float32` | 安全的默认值。在较新的 macOS 上，MPS 也能良好运行 `NANOCHAT_DTYPE=bfloat16`（约节省 25% 内存，速度相近） |

你可以用 `NANOCHAT_DTYPE` 环境变量覆盖默认值：

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # 强制 fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # 强制 bf16
```

原理：模型权重以 fp32 存储（为了优化器精度），而我们自定义的 `Linear` 层在前向传播时将它们转换为 `COMPUTE_DTYPE`。Embedding 直接以 `COMPUTE_DTYPE` 存储以节省内存。这让我们获得与 autocast 相同的混合精度收益，同时完全显式控制每种精度运行的设置。

注意：`float16` 训练会自动在 `base_train.py` 中启用 `GradScaler` 以防止梯度下溢。SFT 也支持这一点，但 RL 目前不支持。fp16 的推理在所有地方都能正常工作。

## 指南

我发布了一些可能含有有用信息的指南，从最新到最早：

- [2026年2月1日：以 <<100 美元击败 GPT-2：nanochat 之旅](https://github.com/karpathy/nanochat/discussions/481)
- [1月7日 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) 记录了第一批 nanochat 模型系列。
- 要为 nanochat 添加新能力，请参阅 [指南：数 strawberry 里的 r（以及如何一般性地添加能力）](https://github.com/karpathy/nanochat/discussions/164)。
- [2025年10月13日：最初的 nanochat 帖子](https://github.com/karpathy/nanochat/discussions/1) 介绍了 nanochat，不过现在它包含一些过时的信息，且模型比当前 master 分支旧得多（结果也更差）。

## 文件结构

```
.
├── LICENSE
├── README.md
├── dev
│   ├── nanochat.png
│   └── repackage_data_reference.py # 预训练数据分片生成
├── nanochat
│   ├── __init__.py                 # 空文件
│   ├── checkpoint_manager.py       # 保存/加载模型检查点
│   ├── common.py                   # 杂项小工具、生活质量优化
│   ├── core_eval.py                # 评估基础模型的 CORE 分数（DCLM 论文）
│   ├── dataloader.py               # 分布式的 token 化数据加载器
│   ├── dataset.py                  # 下载/读取预训练数据的工具
│   ├── engine.py                   # 带 KV Cache 的高效模型推理
│   ├── execution.py                # 让 LLM 以工具形式执行 Python 代码
│   ├── gpt.py                      # GPT nn.Module 转换器
│   ├── loss_eval.py                # 评估字节比特（代替 loss）
│   ├── optim.py                    # AdamW + Muon 优化器，单 GPU 及分布式
│   └── tokenizer.py                # GPT-4 风格的 BPE 分词器封装
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries 训练脚本
│   ├── runcpu.sh                   # 在 CPU/MPS 上运行的小示例
│   ├── scaling_laws.sh             # 缩放法则实验
│   └── speedrun.sh                 # 训练约 100 美元的 nanochat d20
├── scripts
│   ├── base_eval.py                # 基础模型：CORE 分数、字节比特、采样
│   ├── base_train.py               # 基础模型：训练
│   ├── chat_cli.py                 # 对话模型：通过命令行对话
│   ├── chat_eval.py                # 对话模型：评估任务
│   ├── chat_rl.py                  # 对话模型：强化学习
│   ├── chat_sft.py                 # 对话模型：训练 SFT
│   ├── infer_bench.py              # 推理：延迟/吞吐量/VRAM 基准
│   ├── tok_eval.py                 # 分词器：评估压缩率
│   └── tok_train.py                # 分词器：训练它
├── tasks
│   ├── arc.py                      # 科学多选题
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── gsm8k.py                    # 8K 道小学数学题
│   ├── humaneval.py                # 名不副实；简单的 Python 编码任务
│   ├── mmlu.py                     # 多选题，主题广泛
│   └── smoltalk.py                 # 来自 HF 的 SmolTalk 数据聚合集
├── tests
│   ├── test_attention_fallback.py  # FA3/SDPA 注意力回退
│   ├── test_engine.py              # 推理引擎、KV cache
│   ├── test_execution.py           # 沙盒化代码执行
│   ├── test_optim.py               # MuonAdamW 优化器（需要 GPU）
│   ├── test_tasks.py               # 任务切片、混合、HubDataset
│   └── test_tokenizer.py           # BPE 往返、对话渲染
└── uv.lock
```

## 贡献

nanochat 的目标是改进微模型的最新水平，让它们在 1000 美元的预算内可以端到端地研究使用。可及性既关乎总成本，也关乎认知复杂度——nanochat 不是一个配置极其详尽的 LLM "框架"；代码库中没有巨大的配置对象、模型工厂或一堆 if-then-else。它是一个单一、内聚、极简、可读、可修改、可最大程度分支的"强基线"代码库，旨在端到端运行并生成一个可以与之对话的 ChatGPT 模型。目前，我个人最感兴趣的是缩短达到 GPT-2 水平的耗时（即让 CORE 分数超过 0.256525）。目前这大约需要 1.5 小时（从前稳定 3 小时降下来），但通过改进预训练阶段，我们可以进一步缩短。

当前的 AI 政策：披露。在提交 PR 时，请声明任何有重大 LLM 贡献、并非你自己编写或你并不完全理解的部分。

## 致谢

- 名称（nanochat）源自我之前只涵盖预训练的项目 [nanoGPT](https://github.com/karpathy/nanoGPT)。
- nanochat 也受到 [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) 的启发，它用清晰的指标和排行榜把 nanoGPT 仓库游戏化，并借鉴了大量思路以及部分预训练实现。
- 感谢 [HuggingFace](https://huggingface.co/) 提供 fineweb 和 smoltalk。
- 感谢 [Lambda](https://lambda.ai/service/gpu-cloud) 为开发本项目提供的算力。
- 感谢首席 LLM 通灵师 🧙‍♂️ Alec Radford 的建议和指导。
- 感谢仓库管理者 Sofie [@svlandeg](https://github.com/svlandeg) 帮助管理 nanochat 的 issue、PR 和讨论。

## 引用

如果你在研究中觉得 nanochat 有用，请如下引用：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## 许可证

MIT