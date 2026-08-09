# nanochat 中大模型的评估方法

> 本文档面向 nanochat 项目，梳理大模型评估的通用方法论、指标体系/benchmark/工具，并重点剖析 nanochat 中实际使用的评估方法与代码实现。

---

## 第 1 节 大模型评估方法概述

### 1.1 为什么需要评估

大模型（LLM）本质上是"多面手"，其能力横跨知识、推理、语言理解、代码、数学等众多维度。训练过程中我们只能观测到 loss / perplexity 等内部信号，它们无法回答"这个模型到底能不能用、好不好用"。评估（Evaluation）是回答这一问题的关键手段，也是：

- **训练过程监控**：判断一次改动是否有帮助、是否需要回滚；
- **模型选型与对比**：在相似参数量 / 算力下公平比较不同模型；
- **能力定位**：回答"这个模型达到了什么水平"（如"GPT-2 能力"）。

### 1.2 从不同维度评估大模型

可以根据**评估对象**、**评估方式**、**测试时条件**等不同维度对大模型进行评估。

**按评估对象（阶段）分：**

| 对象 | 说明 | 典型指标 |
|------|------|----------|
| Tokenizer | 词元化器的压缩能力 | 压缩率（bytes/token）、往返一致性 |
| 基础模型（Base Model，预训练产出） | 语言建模能力、通用能力 | bpb/BPC、perplexity、CORE、MMLU |
| 对话模型（SFT/RL 产出） | 指令跟随、对话、代码、数学 | ChatCORE、HumanEval、GSM8K |
| 推理引擎 | 延迟 / 吞吐 / 显存 / 带宽利用率 | TTFT、TPOT、tok/s、MBU、MFU |

**按评估方式分：**

- **内在评估（Intrinsic）**：不依赖具体任务，直接度量模型在文本上的建模能力，如 perplexity / bits per byte（bpb）。其优点是平滑、低噪声、几乎不增加额外数据；缺点是"损失低≠能力强"，对知识、推理等能力的刻画很弱。
- **外在评估（Extrinsic）**：把模型放到具体 benchmark 任务上测准确率 / 正确率，如 CORE、MMLU、HumanEval。优点是直接反映"能不能干某类事"；缺点是噪声较大、成本高、存在数据污染风险。

**按测试时条件（Few-shot 设置）分：**

- **0-shot**：直接给问题，不给示例；
- **Few-shot（K-shot）**：给 K 个示例后再答题，考察 in-context learning 能力；
- **CoT（Chain-of-Thought）**：要求模型先输出推理过程再给答案（如 GSM8K、AQuA）。

### 1.3 主流评估思路

1. **单一 benchmark**：如只测 MMLU。简单但信号单一、方差大。
2. **复合 / 聚合指标**：把几十个 benchmark 聚合为一个综合分数（如 CORE、MosaicML Eval Gauntlet 的 aggregate）。**核心思想**：单指标噪声大，用多样化任务求平均来压低方差、提高信噪比；同时按任务类型分大类汇报，便于针对具体能力做决策。
3. **中心化（Centering）校准**：对选择题这类"乱猜也能拿分"的任务，把原始准确率减去随机基线再缩放，使分数落在 0（随机）~1（完美）之间，保证多个任务可公平平均。公式为
   `centered = (acc - random_baseline) / (1 - random_baseline)`。
4. **排行榜 / 竞赛机制**：如 DCLM 的 CORE、nanochat 的 "Time-to-GPT-2" 排行榜，用统一脚本 + 统一基线让社区在同一个标准下比拼。
5. **效率评估**：LLM 评估还应包含"跑起来贵不贵"——延迟、吞吐、显存占用、MFU/MBU 等工程维度（nanochat 的 `infer_bench.py` 即此类）。

### 1.4 主流项目及论文

| 项目 / 论文 | 说明 |
|-------------|------|
| **lm-evaluation-harness**（EleutherAI） | 最流行的开源评测框架，支持数百个 benchmark。https://github.com/EleutherAI/lm-evaluation-harness |
| **OpenCompass / 司南**（上海 AI Lab） | 中文友好的全面评测体系。https://github.com/open-compass/opencompass |
| **HELM**（Stanford CRFM） | Holistic Evaluation of Language Models，多维度综合评测框架。https://crfm.stanford.edu/helm/ |
| **LLM-Foundry / Eval Gauntlet**（MosaicML/Databricks） | 评测 53 个下游任务，CORE 指标体系即源自此。https://github.com/mosaicml/llm-foundry |
| **DCLM / DataComp-LM**（ML Foundations） | 提出 CORE 中心化指标与 22 任务子集。论文：*DataComp-LM: In search of the next generation of training sets for language models*（arXiv:2406.11794）。https://github.com/mlfoundations/dclm |
| **modded-nanogpt**（KellerJordan） | 把 nanoGPT 游戏化，引入"指标 + 排行榜"，nanochat 的 GPT-2 speedrun 受其启发。https://github.com/KellerJordan/modded-nanogpt |

---

## 第 2 节 大模型评估的指标体系、benchmark 及工具

### 2.1 核心指标

| 指标 | 类型 | 含义 |
|------|------|------|
| **Perplexity** | 内在 | 语言建模困惑度，`exp(loss)`，越小越好 |
| **bpb / bpc / bpel** | 内在 | Bits per byte / character / token，把 loss 按字节归一，**与词表大小无关**，可比性更强 |
| **CORE（centered accuracy）** | 外在·聚合 | DCLM 提出的复合指标，22 个任务中心化准确率的均值，0=随机、1=完美 |
| **Extended** | 外在·聚合 | DCLM 全部 53 个任务中心化准确率均值 |
| **ChatCORE** | 外在·聚合 | nanochat 自定义：对话模型在 5 个任务上的中心化准确率均值 |
| **Exact Match / Accuracy** | 外在·单任务 | 生成答案与参考答案是否完全一致 / 选择题正确率 |
| **Pass@k** | 外在·代码 | 生成 k 个代码，至少一个通过全部单测的比例 |
| **MFU / MBU** | 效率 | Model FLOPs Utilization（算力利用率）/ Model Bandwidth Utilization（带宽利用率） |
| **TTFT / TPOT / tok/s** | 效率 | 首 token 延迟 / 每 token 延迟 / 吞吐 |

### 2.2 常用 benchmark

**常识 / 语言理解类**

- **MMLU**：14042 道 4 选 1 学术题，57 个科目（jurisprudence→math→morality），知识面极广。
- **HellaSwag**：10042 个场景完型选择，测试常识推理，区分度好。
- **ARC-Easy / ARC-Challenge**：小学~初中科学知识题（Easy 2376 道，Challenge 2376 道）。
- **PIQA / COPA / OpenBookQA / CommonsenseQA**：物理直觉、因果、日常常识推理。
- **LAMBADA**：书籍段落预测最后一个词，考验长程依赖与语言理解。
- **Winograd / Winogrande**：代词指代消解（schema 类）。
- **BoolQ / SQuAD / CoQA**：阅读理解（BoolQ 是 yes/no，SQuAD 是抽取式问答）。

**符号推理 / 数学类**

- **GSM8K**：1319 道小学应用题，需要 CoT。
- **AQuA / SVAMP**：数学词问题。
- **BIG-bench** 系列（dyck 语言、CS 算法、运算符、repeat_copy 等）。
- **AGI-Eval LSAT / SAT**：逻辑推理题。

**代码类**

- **HumanEval**：OpenAI 发布，164 道 Python 函数补全，Pass@1 评估。

### 2.3 工具链

- **评测框架**：lm-evaluation-harness、OpenCompass、LLM-Foundry、Helm。
- **数据来源**：HuggingFace `datasets` / `huggingface_hub`（支持 `HF_ENDPOINT` 镜像，如 hf-mirror.com）。
- **nanochat 自带评测**：`scripts/base_eval.py`、`scripts/chat_eval.py`、`scripts/tok_eval.py`、`scripts/infer_bench.py`，以及评测数据包 `eval_bundle.zip`（托管于 karpathy-public S3，首次运行时自动下载）。

### 2.4 CORE 指标详解

CORE 是 nanochat 预训练阶段最重要、被反复优化的指标。它来自 DCLM 论文（arXiv:2406.11794），构成如下：

- **22 个 ICL（In-Context Learning）任务**，分属 multiple_choice（11 个）、language_modeling（9 个）、schema（2 个）三类；
- 每个任务在特定 few-shot 设置下算原始准确率；
- 用每个任务的**随机基线**（如 4 选 1 是 25%）做中心化：`centered = (acc - base) / (1 - base)`；
- **CORE = 22 个 centered accuracy 的均值**。

nanochat 的目标是"Time to GPT-2"——在 8 卡 XH100 上训练一个 CORE 分数**超过 0.256525**（GPT-2 的分数）的模型，目前排行榜最快已到 **1.65 小时 / CORE 0.2626**。

---

## 第 3 节 nanochat 中的大模型评估方法

nanochat 的评估按**阶段**分布在 4 个脚本 + 1 个评测包中，下面逐一说明"哪个阶段、哪个任务、哪个数据集、哪个指标、测了什么、代码在哪、怎么实现"。

### 3.1 总体地图

| 阶段 | 脚本 | 评估对象 | 主要指标 |
|------|------|----------|----------|
| 词元化 | `scripts/tok_eval.py` | Tokenizer | 压缩率（bytes/token） |
| 预训练 | `scripts/base_train.py` | Base 模型（训练中） | val_bpb、CORE（周期性） |
| 预训练后 | `scripts/base_eval.py` | Base 模型 | CORE、bpb、采样文本 |
| SFT / RL | `scripts/chat_sft.py`、`scripts/chat_eval.py` | Chat 模型 | ChatCORE、单任务准确率 |
| 推理 | `scripts/infer_bench.py` | 推理引擎 | TTFT、TPOT、tok/s、MBU、MFU |

### 3.2 词元化评估（Tokenizer）

- **脚本**：`scripts/tok_eval.py`
- **测了什么**：nanochat 自训 BPE tokenizer 与 GPT-2 / GPT-4（cl100k_base）tokenizer 的压缩能力对比。
- **数据**：内置的新闻、韩文、代码、LaTeX、科学文本各一段 + ClimbMix 训练/验证语料。
- **指标**：`ratio = utf-8 bytes / token 数`（越高压缩越好），并校验"编码→解码"往返一致。
- **怎么实现**：对每段文本分别用 3 个 tokenizer 编码，统计 token 数与 bytes 数，输出对比表。

### 3.3 预训练评估（Base 模型，训练中 + 训练后）

**训练中（`scripts/base_train.py`）** 每 N 步做两类评估（N 通过参数控制）：

1. **验证集 bpb**：`--eval-every`（默认 250 步），用 `nanochat/loss_eval.py::evaluate_bpb` 计算 **bits per byte**——与词表大小无关的 loss 归一化指标，tokenizer 换词表后仍可比（`nanochat/loss_eval.py:8`）。bpb 平滑、噪声小，适合逐次迭代判断改动是否有效。
2. **CORE**：`--core-metric-every`（默认 2000 步，`-1` 禁用；`--core-metric-max-per-task` 默认 500，`-1` 表示全量），训练末尾必跑一次。全部 rank 参与，结果写入 wandb（`base_train.py:442-452`）。

**训练后（`scripts/base_eval.py`）** 支持 `--eval core,bpb,sample` 三合一：

- `core`：跑 CORE 22 任务；
- `bpb`：在 train / val 两个 split 上分别算 bits per byte（`--split-tokens` 控制 token 量）；
- `sample`：用给定 7 个 prompt 做条件采样，再无条件采样 8 段（temperature=1.0），人工查看模型输出质量。

结果保存为 CSV：`<base_dir>/base_eval/base_model_<step>.csv`（每任务 accuracy + centered + 最终 CORE）。

#### CORE 评测的数据从哪来？需要自己下载吗？

**nanochat 会自动下载，不需要手动准备。** 逻辑在 `scripts/base_eval.py:42,45-66`：

- 评测包 URL 为 `EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"`（约 26MB，来自 MosaicML Eval Gauntlet）；
- 首次运行时 `download_file_with_lock()` 下载并解压到 `get_base_dir()`（`nanochat/common.py:71-91`：可用环境变量 `NANOCHAT_BASE_DIR` 指定，否则是仓库根 `.nanochat/` 或 `~/.cache/nanochat/`），目录名为 `eval_bundle/`；
- 包内含：
  - `core.yaml` —— **CORE 22 任务配置**（任务名、数据集路径、few-shot 数、任务类型）；
  - `eval_meta_data.csv` —— 每任务的**随机基线**（如 4 选 1 是 25）与描述；
  - `eval_data/**` —— **22 个任务的 jsonl 数据**（如 `hellaswag.jsonl`、`arc_easy.jsonl`、`boolq.jsonl` 等）；
- 用 `FileLock` 加锁，`torchrun` 多进程时只有一个 rank 下载，其余等待；
- 网络受限时可用 `HF_ENDPOINT`/镜像方案（仓库 `runs-with-*` 有离线/镜像经验）。

#### CORE 22 任务明细（`core.yaml`）

| 类别（task_type） | 任务（label） | few-shot | 随机基线 | 测什么 |
|---|---|---|---|---|
| multiple_choice | hellaswag_zeroshot | 0 | 25% | 场景完型（0-shot） |
| multiple_choice | hellaswag | 10 | 25% | 场景完型（10-shot） |
| multiple_choice | arc_easy | 10 | 25% | 基础科学知识 |
| multiple_choice | arc_challenge | 10 | 25% | 科学推理（难） |
| multiple_choice | copa | 0 | 50% | 因果推理 |
| multiple_choice | commonsense_qa | 10 | 20% | 日常常识 |
| multiple_choice | piqa | 10 | 50% | 物理直觉 |
| multiple_choice | openbook_qa | 0 | 25% | 开放书籍常识 |
| multiple_choice | boolq | 10 | 62% | 篇章 yes/no 判断 |
| multiple_choice | agi_eval_lsat_ar | 3 | 20% | LSAT 分析推理 |
| multiple_choice | bigbench_language_identification | 10 | 9.1% | 语种识别 |
| language_modeling | jeopardy | 10 | 0 | 知识问答（完形填空式） |
| language_modeling | bigbench_qa_wikidata | 10 | 0 | 维基百科事实补全 |
| language_modeling | lambada_openai | 0 | 0 | 书籍末词预测 |
| language_modeling | bigbench_dyck_languages | 10 | 0 | 括号平衡补全 |
| language_modeling | bigbench_cs_algorithms | 10 | 0 | 算法（LCS/平衡性） |
| language_modeling | bigbench_operators | 10 | 0 | 自定义运算符计算 |
| language_modeling | bigbench_repeat_copy_logic | 10 | 0 | 重复/复制逻辑 |
| language_modeling | squad | 10 | 0 | 抽取式阅读理解 |
| language_modeling | coqa | 0 | 0 | 对话式阅读理解 |
| schema | winograd | 0 | 50% | 指代消解 |
| schema | winogrande | 0 | 50% | 指代消解（规模化） |

#### CORE 的实现流程（`nanochat/core_eval.py`）

1. **加载配置**：读 `core.yaml` 得到任务列表；读 `eval_meta_data.csv` 得到随机基线（`base_eval.py:72-83`）。
2. **按任务评估**（`evaluate_task`，`core_eval.py:244`）：对每个任务加载 jsonl，多进程按 `rank` 分片并行。
3. **逐条样例**（`evaluate_example`，`core_eval.py:168`）：
   - 按任务类型（multiple_choice / language_modeling / schema）用 jinja2 模板渲染 prompt（`render_prompts_*`）；
   - 随机采样 few-shot 示例（seed 固定 `1234+idx`，排除当前条）；
   - 构造 batch，前向得到各 token 的 loss 与 argmax 预测（`forward_model`，`core_eval.py:145`）。
4. **判定对错**：
   - multiple_choice / schema：**最小平均 loss 选答案**（`core_eval.py:232-237`）；
   - language_modeling：逐 token 预测是否与答案完全一致（`core_eval.py:224-231`）。
5. **聚合**：跨 rank `all_reduce` 求和 → 每任务准确率 → 用随机基线中心化 → 22 项取均值得 CORE（`base_eval.py:109-117`）。
6. **TRUNCATION 细节**：若模型有 `max_seq_len`，超出部分从右侧截断并平移 index（`core_eval.py:198-213`），兼容 GPT-2 等短上下文模型。

### 3.4 对话模型评估（Chat，SFT/RL 阶段）

- **训练中（`scripts/chat_sft.py`）**：`--chatcore-every`（默认 200 步）周期性跑 **ChatCORE**——在 `ARC-Easy、ARC-Challenge、MMLU、GSM8K、HumanEval` 5 个任务上评估，按随机基线中心化求均值（`chat_sft.py:356-389`）。`chat_rl.py` 类似。
- **训练后（`scripts/chat_eval.py`）**：命令行指定任务（`-a`），5 个任务全部跑完则输出 ChatCORE（`chat_eval.py:228-238`）。

**两种评估回路（`chat_eval.py`）：**

- **categorical**（`run_categorical_eval`，`chat_eval.py:87`）：ARC / MMLU 这类选择题，**无需采样**，直接批量前向，取**答案位置在候选字母上的 logits argmax** 作为预测（`chat_eval.py:116-135`），按 batch 并行，效率高。
- **generative**（`run_generative_eval`，`chat_eval.py:28`）：GSM8K / HumanEval 这类开放题，用 `Engine.generate_batch` **实际采样**出补全，再调用任务的 `evaluate()` 判断（GSM8K 对最终数字，HumanEval 跑单测）。

**数据下载**：对话任务通过 `tasks/*.py` 从 HuggingFace 自动拉取，如 `cais/mmlu`（`tasks/mmlu.py:19`，用 `load_hub_dataset`，见 `tasks/common.py:45-92`，支持 `HF_ENDPOINT` 镜像，多进程只下载一次，缓存于 `<base_dir>/task_data/`）。

### 3.5 推理效率评估

- **脚本**：`scripts/infer_bench.py`（单卡）。
- **测了什么**：给定 checkpoint，扫 decode batch size（默认 1,8,32,128），度量：
  - prefill：tok/s、**MFU**（compute-bound 下离计算 roofline 的距离）；
  - decode：**TTFT**（首 token 延迟）、**TPOT**（每 token 延迟）、tok/s、**MBU**（带宽利用率，decode 每步都要重读全部权重 + KV cache，是 memory-bandwidth-bound）、峰值显存。
- **输出**：可读表格 + 最后一行紧凑 JSON 便于脚本解析。
- **背景**：`infer_bench.py:9-17` 指出智能指标（CORE 等）不回答"跑起来多贵"，架构选择（GQA、sliding window 等）的收益要靠此类 bench 在工程轴向上评估。

### 3.6 一个综合的例子：`runs/speedrun.sh` 的评估流程

参考 `README.md` / `dev/LEADERBOARD.md`：speedrun 训练结束时
`--core-metric-every=999999` 令 CORE 只在最后一步跑一次，`--core-metric-max-per-task=-1` 全量跑 22 个任务，配合 val_bpb 一起记录。判断一次 run 是否有效，看 wandb 中：
1. `val_bpb`（vs step / total_training_time / total_training_flops）；
2. `core_metric`（CORE 分数）；
3. 训练吞吐 `train/tok_per_sec`、`train/mfu`、显存占用（`README.md:100-104`）。

### 3.7 nanochat 测评涉及的外部工具及数据清单

nanochat **没有**使用 lm-evaluation-harness、OpenCompass 等现成评测框架，而是**自己实现了评估代码**，只借助少量通用外部库和外部数据源。评估涉及的外部工具与数据分为两大类：

#### 3.7.1 外部数据源（自动下载，非仓库内代码）

**（1）eval_bundle（基础模型 CORE 专用，约 26MB）**

- 来源：`https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip`（源自 MosaicML Eval Gauntlet）；
- 内容：22 个 CORE 任务的 jsonl 数据（`eval_data/**`）+ `core.yaml` 任务配置 + `eval_meta_data.csv` 随机基线；
- 首次运行自动下载并解压到 `<base_dir>/eval_bundle/`（`base_eval.py:42-66`）。
- 覆盖的 22 个数据集（确认存在）：hellaswag（0/10-shot）、arc_easy、arc_challenge、copa、commonsense_qa、piqa、openbook_qa、boolq、agi_eval_lsat_ar、bigbench_language_identification、jeopardy、bigbench_qa_wikidata、lambada_openai、bigbench_dyck_languages、bigbench_cs_algorithms、bigbench_operators、bigbench_repeat_copy_logic、squad、coqa、winograd、winogrande。

**（2）HuggingFace 标准数据集（对话模型 ChatCORE 评估用，拉取 test 分片）** — 确认存在清单：

| 任务 | HF 数据集 | 具体配置 | 代码位置 |
|------|-----------|----------|----------|
| MMLU | `cais/mmlu` | subset=all, split=**test** | `tasks/mmlu.py:19` |
| ARC-Easy | `allenai/ai2_arc` | subset=ARC-Easy, split=**test** | `tasks/arc.py:14` |
| ARC-Challenge | `allenai/ai2_arc` | subset=ARC-Challenge, split=**test** | `tasks/arc.py:14` |
| GSM8K | `openai/gsm8k` | subset=main, split=**test** | `tasks/gsm8k.py:42` |
| HumanEval | `openai/openai_humaneval` | subset=openai_humaneval, split=**test** | `tasks/humaneval.py:51` |

> 注意：ARC-Easy / ARC-Challenge 在**两处**都会出现：基础模型 CORE（eval_bundle 里的 jsonl）和对话模型 ChatCORE（HF 的 `allenai/ai2_arc`）。
>
> 另有一个 `HuggingFaceTB/smol-smoltalk`（SmolTalk）仅用于 SFT **训练**（train 分片，460K 行），**不用于评估**（`tasks/smoltalk.py:15`）。

#### 3.7.2 外部 Python 库

| 库 | 用途 | 代码位置 |
|----|------|----------|
| torch / torch.distributed | 前向计算 + 多卡并行评估聚合 | 所有 eval 脚本 |
| jinja2 | CORE 任务的 prompt 模板渲染 | `core_eval.py:10` |
| pyyaml | 解析 `core.yaml` 任务配置 | `base_eval.py:23` |
| pyarrow + huggingface_hub | 下载/读取 HF 数据集 parquet | `tasks/common.py:14-15,52` |
| filelock | 多进程下载评测包/数据的并发锁 | `common.py:105`、`tasks/common.py:16` |
| tiktoken / rustbpe | `tok_eval.py` 对比 GPT-2/GPT-4 tokenizer 压缩率 | `tok_eval.py:5` |
| numpy | 数据 shuffle 置换（HubDataset） | `tasks/common.py:13,35` |
| wandb | 评测结果记录与可视化 | `base_train.py:447` 等 |

**总结**：nanochat 的**核心评测逻辑（CORE/ChatCORE/bpb/采样）全部是自研代码**（`nanochat/core_eval.py`、`loss_eval.py`、`scripts/*_eval.py`），外部只借用数据包（eval_bundle、HF 数据集）和基础库，不依赖任何第三方评测框架。

---

## 第 4 节 基本概念、支撑技术

### 4.1 核心概念

- **Perplexity 与 bpb**：perplexity 与词表大小有关，换词表后不可比；**bits per byte**（或 per token / per character）把 loss 除以目标 token 对应的**字节数**归一，消除词表影响，是 nanochat 首选的"平滑主力指标"。
- **中心化（Centering）**：对随机基线 >0 的任务，`(acc - base)/(1 - base)`，使分数落在 0~1，多任务可公平平均。CORE、ChatCORE、Eval Gauntlet aggregate 都用了它。
- **Few-shot / In-Context Learning（ICL）**：把示例拼进 prompt 让模型按示例风格答题，无需改权重，是 base 模型评估的主流范式。CORE 的 22 个任务就是在 0/3/10-shot 下评测的。
- **chain-of-thought（CoT）**：让模型先推理再作答，显著提升数学/逻辑题表现（GSM8K、AQuA 等）。
- **Pass@k**：代码生成指标，k 次采样中至少一次通过测试的概率。
- **MFU / MBU**：MFU = 实际计算量 / 理论峰值算力；MBU = 实际搬运字节 / 理论峰值带宽。训练主要看 MFU，decode 受带宽约束主要看 MBU。
- **数据污染（Contamination）**：训练语料若包含评测集文本会高估能力；nanochat 用 ClimbMix/FineWeb 等公开语料，存在该风险，因此榜单同时记录 val_bpb 作交叉验证。

### 4.2 支撑技术 / 实现要点

- **`Engine`（`nanochat/engine.py`）**：带 KV cache 的高效自回归生成引擎，支撑 sampling、generative eval、chat CLI 与 infer_bench。
- **`tokenizing_distributed_data_loader_bos_bestfit`（`nanochat/dataloader.py`）**：分布式数据加载，评估 bpb 时按 best-fit 打包 token 序列。
- **`disable_fp8`**：评估时关闭 FP8、用 BF16 保证结果稳定可比（`base_train.py:444`）。
- **多进程并行评估**：`evaluate_task` 按 rank 分片样例，`dist.all_reduce` 聚合，`torchrun --nproc_per_node=8` 时 8 卡并行评估 22 任务，显著加速。
- **确定性**：few-shot 采样与数据 shuffle 均固定随机种子（1234/1337/42），保证可复现、可横向对比。
- **文件锁（FileLock）**：多 rank 并发下载评测包 / HF 数据时只下载一份。
- **镜像 / 离线方案**：`load_hub_dataset` 支持 `HF_ENDPOINT`（如 hf-mirror.com）；仓库 `runs-with-win/`、`runs-with-autodl/` 记录了在中国大陆网络 / 离线环境下跑通评测的完整过程。

### 4.3 小结

nanochat 的评估体系是"**内外结合、层层递进**"的：

- 预训练用 **bpb（内）** 做低噪声的日常监控，用 **CORE（外）** 做最终能力裁定，二者互补；
- SFT/RL 后用 **ChatCORE** 把关对话能力；
- 最后用 **infer_bench** 度量工程效率。
- 评测数据（eval_bundle、HF 数据集）**全部自动下载**，开箱即用，这是 nanochat "端到端可复现"理念的一部分。
