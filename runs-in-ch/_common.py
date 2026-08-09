"""
runs-in-ch 各 step 脚本共享的初始化与工具函数。

要点（对应 任务88 与 中文数据集选型.md §4.2）：
- NANOCHAT_BASE_DIR 必须是绝对路径 <项目根>/data/zh-ch，由脚本所在目录
  __file__ 推导，不依赖调用时的 cwd。
  原版 nanochat/common.py 的 get_base_dir() 会读取这个环境变量，
  于是 tok_train / tok_eval / base_train / base_eval / chat_sft 全都
  一行不改地消费中文数据（tokenizer/、base_data_climbmix/、task_data/、
  base_checkpoints/、chatsft_checkpoints/ 都落在 data/zh-ch/ 之下）。
- 数据目录与 runs-in-zh-en（../data/zh-en）相互隔离，两套并行不互覆。
- 项目根目录加入 sys.path，保证 `python runs-in-ch/stepX.py` 也能 import
  nanochat / scripts / tasks。
"""

import os
import sys
import subprocess
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "zh-ch")

os.environ["NANOCHAT_BASE_DIR"] = DATA_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Windows CPU：本机没有安装 MSVC `cl` 编译器，torch.compile（Inductor 后端）
# 无法编译生成的 C++ 代码，一调用就抛 "Compiler: cl is not found"。用
# TORCH_COMPILE_DISABLE=1 让 torch.compile 退化为 eager 模式（返回一个
# OptimizedModule 包装，前向照常跑），base_train / chat_sft 里的
# `model = torch.compile(model, dynamic=False)` 因此可以直接通过。
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Windows 控制台默认用 GBK 编码，原版 banner 里的制表符（如 ░）会抛
# UnicodeEncodeError。把当前进程的 stdout/stderr 重配为 UTF-8；
# 子进程则继承 PYTHONIOENCODING。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def guard(condition, message):
    """前置条件检查：不满足则打印原因并退出（不破坏后续步骤）。"""
    if not condition:
        print(f"\n[step] 前置条件不满足：{message}")
        sys.exit(1)


def data_dir(*parts):
    return os.path.join(DATA_DIR, *parts)


def run_module(args, check=True):
    """以 `python -m ...` 方式在项目根目录运行子命令（继承 NANOCHAT_BASE_DIR）。"""
    cmd = [sys.executable, "-m", *args]
    print(f"\n[step] $ python -m {' '.join(args)}\n", flush=True)
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=check)


def run_module_inproc(module_path_rel, argv, note=""):
    """进程内以 __main__ 运行脚本，并设置 torch 线程数。

    Windows 下 PyTorch 的线程数只能靠 torch.set_num_threads() 设置，子进程
    （run_module）拿不到，而默认线程数=物理核（这台 i5-14400 是 10）会让训练
    慢 ~3 倍（实测 28s/step -> 16 线程 9.7s/step）。所以 torch 步骤都在进程内
    跑，先设满逻辑线程（16）再执行脚本。
    """
    import torch

    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    if note:
        print(f"\n[step] {note}", flush=True)

    module_path = os.path.join(PROJECT_ROOT, module_path_rel)
    print(f"\n[step] $ python {module_path_rel} {' '.join(argv)}\n", flush=True)
    sys.argv = [module_path_rel, *argv]
    runpy.run_path(module_path, run_name="__main__")


def has_shards(rel_dir="base_data_climbmix"):
    d = data_dir(rel_dir)
    if not os.path.isdir(d):
        return False
    return any(f.endswith(".parquet") and not f.endswith(".tmp") for f in os.listdir(d))


def has_tokenizer():
    return os.path.exists(data_dir("tokenizer", "tokenizer.pkl"))


def has_checkpoints(rel_dir):
    d = data_dir(rel_dir)
    return os.path.isdir(d) and any(
        os.path.isdir(os.path.join(d, f)) for f in os.listdir(d)
    )
