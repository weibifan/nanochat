"""
runs-in-zh-en 各 step 脚本共享的初始化与工具函数（跨平台版）。

设计目标：Windows 与 Linux 都能直接运行（对应任务要求「sh 改 py」）。

要点：
- NANOCHAT_BASE_DIR 必须是绝对路径 <项目根>/data/zh-en，由脚本所在目录 __file__ 推导，
  与调用时的 cwd/无关。原版 nanochat/common.py 的 get_base_dir() 读取该环境变量，
  于是 tok_train / base_train / base_eval / chat_sft / chat_cli 都消费同一份数据
  （tokenizer/、base_data_climbmix/、task_data/、base_checkpoints/、chatsft_checkpoints/）。
- 数据目录与 runs-in-ch（../data）**相互隔离**，两套并行不互覆。
- 项目根加入 sys.path，保证 `python runs-in-zh-en/stepX.py` 也能 import
  nanochat / scripts / tasks。
- install_zh_modules()：把 tasks.common 与 nanochat.dataset 在 sys.modules 里重定向到
  *_zh 版本（tasks/common_zh.py、nanochat/dataset_zh.py）。这样 chat_sft / chat_eval /
  tok_train 内部 "from tasks.common / nanochat.dataset import ..." 拿到的就是 zh-en
  适配：download_sft_data_zh 预下载后的本地缓存 + ModelScope 下载源。
- 运行 torch 步骤统一用 run_torch_cmd()：在进程内 runpy 执行（可设 CPU 线程数），
  避免 Windows 子进程拿不到 torch.set_num_threads 的坑；GPU 上线程设置无害。
"""

import os
import sys
import time
import subprocess
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))        # runs-in-zh-en/
PROJECT_ROOT = os.path.dirname(HERE)                     # nanochat repo 根
# zh-en 套件独立数据目录：与 runs-in-ch（data/）彻底隔离，避免互相覆盖
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "zh-en")

os.environ["NANOCHAT_BASE_DIR"] = DATA_DIR
# 把项目根放最前（保证 import nanochat / tasks 命中 repo 根的真实包；_zh 变体也在其下）
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Windows CPU：本机如果没装 MSVC `cl`，torch.compile（Inductor）无法编译生成的
# C++ 代码会直接抛错。默认在 Windows 上禁用（退化 eager），Linux 保留编译能力。
# 用环境变量 NANOCHAT_TORCH_COMPILE 显式覆盖（NANOCHAT_TORCH_COMPILE=0 强制禁用）。
if os.environ.get("NANOCHAT_TORCH_COMPILE", "1") == "0":
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
elif sys.platform == "win32":
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Windows 控制台默认 GBK，重配为 UTF-8；子进程继承 PYTHONIOENCODING。
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


def timed(func):
    """装饰器：包装一个 step 的 main()，结束时打印该步总耗时（分:秒）。

    用法：`@timed` 加在 `def main():` 上方，运行结束自动输出
      [time] 本步耗时: 12:34 (12.6 min)
    便于把实测耗时直接填进 README 的“实测耗时”列。
    """

    def wrapper(*args, **kwargs):
        t0 = time.monotonic()
        try:
            return func(*args, **kwargs)
        finally:
            dt = time.monotonic() - t0
            mm, ss = divmod(int(dt), 60)
            print(f"\n[time] 本步耗时: {mm:02d}:{ss:02d} ({dt/60:.1f} min)", flush=True)

    return wrapper


def data_dir(*parts):
    return os.path.join(DATA_DIR, *parts)


def run_module(args, check=True):
    """以 `python -m ...` 在项目根运行子命令（继承 NANOCHAT_BASE_DIR / _zh 重定向）。"""
    cmd = [sys.executable, "-m", *args]
    print(f"\n[step] $ python -m {' '.join(args)}\n", flush=True)
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=check)


def run_in_process(module_path, argv, note=""):
    """
    进程内以 __main__ 运行一个脚本，并（若 import 得到 torch）设置 CPU 线程数。

    为什么进程内：
    - Windows 下 torch 线程数只能 torch.set_num_threads() 设置，子进程拿不到；
    - chat_sft / chat_eval 需要先做完 sys.modules 的 *_zh 重定向，子进程里改不到，
      只能在带重定向的进程内 runpy。
    """
    if note:
        print(f"\n[step] {note}", flush=True)
    try:
        import torch
        torch.set_num_threads(os.cpu_count() or 4)
    except Exception:
        pass
    module_path = os.path.join(PROJECT_ROOT, module_path)
    print(f"\n[step] $ python {module_path} {' '.join(argv)}\n", flush=True)
    sys.argv = [module_path, *argv]
    runpy = __import__("runpy")
    runpy.run_path(module_path, run_name="__main__")


def install_windows_signal_patch():
    """Windows 兼容补丁（不改原文件）：nanochat/engine.py 的 eval_with_timeout 依赖
    Unix 专属的 signal.SIGALRM / signal.alarm，Windows 上不存在，会在 ChatCORE 的
    GSM8K 生成式评测（模型调用计算器）时崩溃。这里替换为线程超时版，语义等价
    （同样禁用 __builtins__），仅 win32 生效。"""
    if sys.platform != "win32":
        return
    import warnings
    import concurrent.futures
    import nanochat.engine as engine

    def _eval_with_timeout(formula, max_time=3):
        def _run():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run)
            try:
                return fut.result(timeout=max_time)
            except Exception:  # noqa: BLE001  # 超时/计算失败一律忽略，同原逻辑
                return None

    engine.eval_with_timeout = _eval_with_timeout


def install_zh_modules():
    """
    把原版模块在 sys.modules 重定向到 *_zh 版本（不改任何原文件）：

    - tasks.common -> tasks/common_zh.py   （读本地 task_data 缓存，不联网）
    - tasks.smoltalk -> tasks/smoltalk_zh.py （过滤 >512 token / 无监督对话，杜绝 SFT NaN）
    - tasks.mmlu -> tasks/mmlu_zh.py        （同上：8192 词表下 35% MMLU 超长，必须过滤）
    - nanochat.dataset -> nanochat/dataset_zh.py （BASE_URL = ModelScope 镜像）

    _zh 变体与 *_ch 一样放在 repo 根对应包下（nanochat/dataset_zh.py、tasks/common_zh.py）。
    之后 chat_sft / chat_eval 里的 "from tasks.common / nanochat.dataset import ..."
    都会拿到 *_zh 版。
    """
    install_windows_signal_patch()
    import tasks.common_zh as common_zh
    import tasks.smoltalk_zh as smoltalk_zh
    import tasks.mmlu_zh as mmlu_zh
    import nanochat.dataset_zh as dataset_zh

    sys.modules["tasks.common"] = common_zh
    sys.modules["tasks.smoltalk"] = smoltalk_zh
    sys.modules["tasks.mmlu"] = mmlu_zh
    sys.modules["nanochat.dataset"] = dataset_zh
    return common_zh


def count_shards(rel_dir="base_data_climbmix"):
    d = data_dir(rel_dir)
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.endswith(".parquet") and not f.endswith(".tmp"))


def has_tokenizer():
    return os.path.exists(data_dir("tokenizer", "tokenizer.pkl"))


def has_checkpoints(rel_dir):
    return os.path.isdir(data_dir(rel_dir)) and any(
        os.path.isdir(os.path.join(data_dir(rel_dir), f)) for f in os.listdir(data_dir(rel_dir))
    )


def has_sft_data():
    return os.path.isdir(data_dir("task_data"))