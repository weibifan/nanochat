"""
cpustep0：环境巡检 + 依赖补装（合为一步，对应 runcpu.sh 的 setup 段）。

两部分：
1. 环境巡检（只读）：系统 / CPU / 内存 / Python / torch / 关键依赖 /
   编译工具链 / 磁盘剩余，定位「能不能跑」与缺失项。
2. 依赖补装：--install 对缺失的 Python 包 pip install 补齐
   （runcpu.sh 用 `uv sync --extra cpu`，本机系统 Python 3.12 已有 torch
   CPU 版，直接用 pip 即可）。

注意：C++ 工具链（gcc/g++/cl/make/cmake）缺失不影响 CPU 跑——_common.py
已把 torch.compile 退化为 eager；rustbpe 走官方多平台 wheel。

用法：
  python runs-in-zh-en/cpustep0_env_eval.py            # 只巡检
  python runs-in-zh-en/cpustep0_env_eval.py --install  # 巡检 + pip 补装缺失依赖
"""

import os
import sys
import shutil
import subprocess
import argparse

# Windows 控制台默认 GBK，重配为 UTF-8（与其他 step 一致）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

REQUIRED = ["torch", "huggingface_hub", "tiktoken", "numpy", "rustbpe", "filelock", "pyarrow"]
COMPILE_TOOLS = ["cmake", "make", "gcc", "g++", "cl", "cc"]

_py = sys.executable


def _run(code, timeout=40):
    try:
        return subprocess.run(
            [_py, "-c", code], capture_output=True, text=True, timeout=timeout
        ).stdout.strip()
    except Exception as e:
        return f"<err: {e}>"


def _show(title, value, ok=True):
    mark = "ok " if ok else "**"
    print(f"  [{mark}] {title}: {value}" if value else f"  [{mark}] {title}: (空)")


def check_env():
    print("=" * 66)
    print("cpustep0：环境巡检（只读）")
    print(f"  脚本目录 : {HERE}")
    print(f"  项目根   : {PROJECT_ROOT}")
    print("=" * 66)

    print("\n=== 系统 / CPU / 内存 ===")
    _show("平台", sys.platform)
    _show("CPU 逻辑核", os.cpu_count())
    try:
        import psutil

        _show("内存", f"{psutil.virtual_memory().total / 1e9:.1f} GB")
    except Exception:
        _show("内存", "(psutil 未安装，跳过)")

    print("\n=== Python / torch ===")
    _show("Python", _run("import sys; print(sys.version.split()[0])"))
    _show("torch", _run("import torch; print(torch.__version__)"))
    cuda = _run("import torch; print(torch.cuda.is_available())")
    _show("CUDA 可用", cuda)

    print("\n=== 训练 Python 包 ===")
    missing = []
    for pkg in ["nanochat", "scripts"]:
        okp = os.path.isdir(os.path.join(PROJECT_ROOT, pkg))
        _show(pkg, "OK" if okp else "缺失", okp)
        if not okp:
            missing.append(pkg)
    for pkg in REQUIRED:
        if pkg == "torch":
            continue  # 上一节已显示
        ver = _run(f"import {pkg}; print(getattr({pkg}, '__version__', 'OK'))")
        okpy = not ver.startswith("<err")
        _show(pkg, ver if okpy else "缺失", okpy)
        if not okpy:
            missing.append(pkg)

    print("\n=== 编译 / 工具链 ===")
    for t in COMPILE_TOOLS:
        found = shutil.which(t)
        _show(t, found if found else "缺失", bool(found))
    if not any(shutil.which(t) for t in ("cc", "cl", "gcc", "g++")):
        print("  说明：无 C++ 工具链 → torch.compile 已退化为 eager，不影响 CPU 跑。")

    print("\n=== 磁盘（NANOCHAT_BASE_DIR） ===")
    base = os.environ.get("NANOCHAT_BASE_DIR") or os.path.join(PROJECT_ROOT, "data")
    try:
        free = shutil.disk_usage(base).free / 1e9
        _show("data 目录剩余", f"{free:.1f} GB", free > 5)
    except Exception as e:
        _show("磁盘", str(e), False)

    print("\n=== 判断 ===")
    _show("CUDA", cuda if not cuda.startswith("False") else "无 CUDA，走 CPU", True)
    if missing:
        print(f"  * 缺失 Python 包：{' '.join(missing)} → 用 --install 或手动 pip 补齐")
    else:
        print("  * 依赖齐全，可直接开始。")
    print("  巡检完成。")
    return missing


def install(missing):
    print("\n=== 依赖补装（--install）===")
    if not missing:
        print("  无缺失依赖。")
        return
    cmd = [sys.executable, "-m", "pip", "install", "-q", *missing]
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("  pip 安装返回非零，请检查网络后重试。")
        sys.exit(1)
    print("  补装完成，建议重跑一次 cpustep0 复核。")


if __name__ == "__main__":
    import time

    _t0 = time.monotonic()
    parser = argparse.ArgumentParser(description="cpustep0: 环境巡检 + 可选依赖补装")
    parser.add_argument("--install", action="store_true", help="pip 自动补齐缺失的 Python 依赖")
    args = parser.parse_args()

    missing = check_env()
    if args.install:
        install(missing)

    _dt = time.monotonic() - _t0
    _mm, _ss = divmod(int(_dt), 60)
    print(f"\n[time] 本步耗时: {_mm:02d}:{_ss:02d} ({_dt/60:.1f} min)")
    print("\ncpustep0 完成。下一步：python runs-in-zh-en/cpustep1_prepare_data.py")