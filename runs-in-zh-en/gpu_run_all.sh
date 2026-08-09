#!/usr/bin/env bash
# gpu_run_all.sh —— speedrun.sh 的单卡 GPU 改造：把 speedrun.sh 的多卡 torchrun 全流程
# 拆成 gpustep0..8 顺序执行（每步对应一个 gpustepX.py，失败即停）。
#
# 在 AutoDL 租用机器（RTX 4090D 单卡）上，从 nanochat 仓库根目录运行：
#   bash runs-in-zh-en/gpu_run_all.sh                  # 默认 fast（约 30 分钟全流程）
#   NANOCHAT_CONFIG=full bash runs-in-zh-en/gpu_run_all.sh   # full（约 20 小时全流程）
#
# 关键差异 vs speedrun.sh：
#   - 单卡 → 不再需要 torchrun/分布式；base_train/chat_sft 直接进程内运行，
#     nanochat 自动按 SM 8.9 选 bf16（FP8 需 H100+，本卡不支持，故不开 --fp8）。
#   - 数据/评测全部走 ModelScope 与本地 eval_data（服务器无法访问 HF/GitHub）。
#   - 所有数值可用 NANOCHAT_GPU_* 环境变量逐项覆盖（见 README 第 4 节）。
#
# 可选：NANOCHAT_SKIP_CLI=1 跳过最后一步（交互式对话 chat_cli）。

set -euo pipefail

CONFIG="${NANOCHAT_CONFIG:-fast}"
export PYTHONUNBUFFERED=1

echo "[gpu_run_all] NANOCHAT_CONFIG=${CONFIG}  （fast≈30min / full≈20h）"
echo "[gpu_run_all] 数据目录: ${NANOCHAT_BASE_DIR:-<项目根>/data/gpu}"

steps=(
  gpustep0_setup
  gpustep1_prepare_data
  gpustep2_tok_train
  gpustep3_tok_eval
  gpustep4_base_train
  gpustep5_base_eval
  gpustep6_chat_sft
  gpustep7_chat_eval
)

if [ "${NANOCHAT_SKIP_CLI:-0}" != "1" ]; then
  steps+=(gpustep8_chat_cli)
fi

for s in "${steps[@]}"; do
  echo
  echo "########## ${s} ##########"
  python "runs-in-zh-en/${s}.py" "$@"
done

echo
echo "全流程完成！"
