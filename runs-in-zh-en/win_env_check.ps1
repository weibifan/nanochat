<#
win_env_check.ps1 —— 检查一台 Windows 机器的系统/CPU/内存/磁盘/GPU/Python 环境。
纯 PowerShell 实现，不依赖 Python：即使 Python 没装、GPU 没驱动，也能照常输出系统信息。

用法（右键"用 PowerShell 运行" 或命令行）：
  powershell -ExecutionPolicy Bypass -File win_env_check.ps1
#>

$ErrorActionPreference = "SilentlyContinue"
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }
$env:PYTHONIOENCODING = "utf-8"

function Get-CimSafe {
  param([string]$Query)
  try { Get-CimInstance -Query $Query } catch { $null }
}

Write-Output "=== 系统 ==="
$os = Get-CimSafe "SELECT Caption, Version, OSArchitecture, TotalVisibleMemorySize, FreePhysicalMemory FROM Win32_OperatingSystem"
if ($os) {
  Write-Output "系统: $($os.Caption)  (版本 $($os.Version))  [$($os.OSArchitecture)]"
} else {
  Write-Output "系统: (无法获取操作系统信息)"
}
Write-Output "PowerShell: $($PSVersionTable.PSVersion.ToString())  ($($PSVersionTable.PSEdition))"
Write-Output "主机名: $env:COMPUTERNAME"

Write-Output ""
Write-Output "=== CPU ==="
$cpu = Get-CimSafe "SELECT Name, NumberOfCores, NumberOfLogicalProcessors FROM Win32_Processor"
if ($cpu) {
  Write-Output "型号: $($cpu.Name)"
  Write-Output "物理核心: $($cpu.NumberOfCores)   逻辑核心: $($cpu.NumberOfLogicalProcessors)"
} else {
  Write-Output "CPU: (无法获取)"
}

Write-Output ""
Write-Output "=== 内存 ==="
if ($os) {
  $totalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
  $freeGB  = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
  Write-Output "总内存: ${totalGB} GB    可用: ${freeGB} GB"
} else {
  Write-Output "内存: (无法获取)"
}

Write-Output ""
Write-Output "=== 磁盘 ==="
Get-CimSafe "SELECT DeviceID, Size, FreeSpace FROM Win32_LogicalDisk WHERE DriveType = 3" | ForEach-Object {
  if ($_.Size) {
    $sizeGB = [math]::Round($_.Size / 1GB, 1)
    $freeGB = [math]::Round($_.FreeSpace / 1GB, 1)
    Write-Output ("{0}  共 {1} GB    可用 {2} GB" -f $_.DeviceID, $sizeGB, $freeGB)
  }
}

Write-Output ""
Write-Output "=== GPU ==="
$gpus = Get-CimSafe "SELECT Name, AdapterRAM FROM Win32_VideoController"
if ($gpus) {
  foreach ($g in $gpus) {
    $ram = if ($g.AdapterRAM) { "$([math]::Round($g.AdapterRAM / 1GB, 1)) GB" } else { "未知" }
    Write-Output "型号: $($g.Name)    显存: $ram"
  }
} else {
  Write-Output "GPU: (无法获取)"
}
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  Write-Output "--- nvidia-smi ---"
  nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.gpu,memory.used --format=csv 2>&1
} else {
  Write-Output "nvidia-smi 不在 PATH（可能无 NVIDIA 驱动，或驱动未装）"
}

Write-Output ""
Write-Output "=== MSVC（torch.compile 需要）==="
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($cl) {
  Write-Output "cl.exe: $($cl.Source)"
} else {
  Write-Output "cl.exe: 不在 PATH"
}
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
  $vs = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property displayName,installationPath 2>$null) | Select-Object -First 1
  if ($vs) {
    Write-Output "Visual Studio C++ 工具链: $vs"
  } else {
    Write-Output "Visual Studio C++ 工具链: 未装（缺 VS 的 VC.Tools 组件，torch.compile 无法编译 C++）"
  }
} else {
  Write-Output "vswhere: 未找到（未装 Visual Studio Installer）"
}
if (Get-Command ninja -ErrorAction SilentlyContinue) {
  Write-Output "ninja: $(ninja --version)"
} else {
  Write-Output "ninja: 未安装（torch.compile 通常需要）"
}

Write-Output ""
Write-Output "=== Python ==="
$py = $null
foreach ($cand in @("python", "py", "python3")) {
  $cmd = Get-Command $cand -ErrorAction SilentlyContinue
  if ($cmd) {
    $ver = (& $cand --version 2>&1) -join " "
    Write-Output "$cand : $ver   @ $($cmd.Source)"
    $py = $cmd.Source
    break
  }
}
if (-not $py) { Write-Output "未安装 Python（官网 python.org 下载，或装 Anaconda/Miniconda）" }
if (Get-Command conda -ErrorAction SilentlyContinue) {
  Write-Output "conda: $(conda --version)  @ $((Get-Command conda).Source)"
}

Write-Output ""
Write-Output "=== PyTorch / 关键依赖（需要 Python）==="
if ($py) {
  & $py -c "import torch; print('torch:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('GPU 名:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1 | ForEach-Object { Write-Output $_ }
  & $py -c "import numpy; print('numpy:', numpy.__version__)" 2>&1 | ForEach-Object { Write-Output $_ }
  & $py -c "import rustbpe; print('rustbpe: OK')" 2>&1 | ForEach-Object { Write-Output $_ }
  & $py -c "import filelock; print('filelock: OK')" 2>&1 | ForEach-Object { Write-Output $_ }
  & $py -c "import tiktoken; print('tiktoken:', tiktoken.__version__)" 2>&1 | ForEach-Object { Write-Output $_ }
  & $py -c "import pyarrow; print('pyarrow:', pyarrow.__version__)" 2>&1 | ForEach-Object { Write-Output $_ }
} else {
  Write-Output "未找到 Python，跳过依赖检测。"
}
