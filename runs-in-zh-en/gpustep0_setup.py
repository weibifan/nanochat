"""
gpustep0：环境巡检 + （可选）从 Gitee 克隆 nanochat 仓库。

AutoDL 租用机器要点：
- 可访问 ModelScope / Gitee，但**不可访问** huggingface.co 与 github.com。
- 本步骤检查 torch + CUDA + 显卡是否就绪；缺依赖会给出安装命令。
- 若当前目录还不是 nanochat 仓库（缺 scripts/chat_sft_zh.py）且设置了
  NANOCHAT_GITEE_REPO，则自动 `git clone` 到 <项目根>/nanochat 旁边。

用法：
  python runs-in-zh-en/gpustep0_setup.py
  NANOCHAT_CONFIG=full python runs-in-zh-en/gpustep0_setup.py          # 选 full 配置
  NANOCHAT_GITEE_REPO=https://gitee.com/xxx/nanochat.git ...           # 需要克隆时
"""

import os
import sys
import platform
import subprocess

import _gpu_common as g
from _common import timed

GITEE_REPO = os.environ.get("NANOCHAT_GITEE_REPO", "").strip()


@timed
def main():
    g.print_profile()
    print("=" * 66)
    print("gpustep0：环境巡检")
    print("=" * 66)

    # 0) 仓库是否就位（本步骤脚本所在仓库根是否完整）
    repo_root = g.PROJECT_ROOT
    need_zh = os.path.join(repo_root, "scripts", "chat_sft_zh.py")
    if not os.path.exists(need_zh):
        if not GITEE_REPO:
            print(f"[env] 当前目录不是 nanochat 仓库（缺少 {need_zh}）。")
            print(f"[env] 请设置 NANOCHAT_GITEE_REPO 后重跑本步骤，例如：")
            print(f'[env]   $ env NANOCHAT_GITEE_REPO=https://gitee.com/<用户名>/nanochat.git \\')
            print(f'[env]       python runs-in-zh-en/gpustep0_setup.py')
            sys.exit(1)
        print(f"[env] 从 Gitee 克隆 nanochat 仓库：{GITEE_REPO}")
        subprocess.run(["git", "clone", "--depth=1", GITEE_REPO,
                        os.path.join(os.path.dirname(repo_root), "nanochat")], check=True)
        # 克隆完成后，脚本所在目录不再是仓库根，需要提示重新进入仓库运行
        print("[env] 克隆完成。请进入克隆出的仓库目录后，重新运行 gpustep0 及其余步骤。")
        sys.exit(0)
    print(f"[env] 仓库就位：{repo_root}")

    # 1) Python
    print(f"[env] Python {platform.python_version()} ({platform.platform()})")

    # 2) torch + CUDA
    try:
        import torch
        print(f"[env] torch {torch.__version__}  CUDA 可用: {torch.cuda.is_available()}")
    except Exception as e:  # noqa: BLE001
        print(f"[env] torch 导入失败：{e}")
        print("[env] 建议安装 GPU 版依赖（PyPI 可访问，wheel 自带 CUDA）：")
        print("[env]   $ uv sync --extra gpu          # pyproject 已锁 torch==2.9.1")
        print("[env]   或（已有 uv 环境时）")
        print("[env]   $ uv pip install torch --index-url https://download.pytorch.org/whl/cu126")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[env] 检测不到 CUDA 设备！请确认：")
        print("[env]   1) 租用的机器确实分配了 GPU（AutoDL 控制台-实例-状态）；")
        print("[env]   2) 当前 venv 里装的是 CUDA 版 torch（见上方安装命令）。")
        sys.exit(1)

    # 3) 显卡信息
    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[env] GPU: {name}  计算能力 {cap[0]}.{cap[1]}  显存 {vram_gb:.1f} GB")
        if cap[0] < 8:
            print("[env] 警告：计算能力 < 8.0，nanochat 会自动退到 fp16 而非 bf16，速度较慢。")
    except Exception as e:  # noqa: BLE001
        print(f"[env] 读取显卡信息失败：{e}")

    print("\ngpustep0 完成！下一步：python runs-in-zh-en/gpustep1_prepare_data.py")


if __name__ == "__main__":
    sys.exit(main())
