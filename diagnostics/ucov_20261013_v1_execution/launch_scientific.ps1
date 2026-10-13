# UCOV-20261013-v1: the one authorized launch, with stdout/stderr, times and exit codes saved under launch/.
# Adapted from the URP-v1 launch script. It refuses to start if launch/ exists, or if --status does not show
# 0 invocations, 0 starts and no run root.
# It is started detached (Start-Process -WindowStyle Hidden), so an interrupted chat or tool call neither stops the
# bounded run nor leads to a second launch. The supervisor keeps ownership of its child processes and its guards.
# The last line of COMMANDS_AND_EXIT_CODES.txt is "LAUNCH_SCRIPT_END ..." whatever the outcome.
$ErrorActionPreference = 'Stop'
$pkg = 'D:\projects-cd\cg-fedllm-reproduction\diagnostics\ucov_20261013_v1'
$ev = 'D:\projects-cd\cg-fedllm-reproduction\diagnostics\ucov_20261013_v1_execution\launch'
$py = 'D:/conda_envs/cgfedllm/python.exe'
$sha = '3bc2ab302f6cd77b63b9bb8b77894041fd97cbdd63d5fa7f380a4dae080fe824'
New-Item -ItemType Directory -Path $ev | Out-Null  # throws if launch/ exists: never a second launch
$utf8 = New-Object Text.UTF8Encoding $false
$log = Join-Path $ev 'COMMANDS_AND_EXIT_CODES.txt'
function Note([string]$s) { [IO.File]::AppendAllText($log, $s + "`n", $utf8) }
function Stamp { (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss.fffzzz') }
function Load {  # reported only; a failure here never blocks or alters the launch
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        "cpu_load_percent=$cpu free_physical_bytes=$([int64]$os.FreePhysicalMemory * 1024) at $(Stamp)"
    } catch { "load unavailable ($($_.Exception.Message)) at $(Stamp)" }
}
function Run([string]$name, [string[]]$argv) {
    $t0 = Stamp
    $p = Start-Process -FilePath $py -ArgumentList $argv -WorkingDirectory $pkg -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $ev "$name.stdout.txt") -RedirectStandardError (Join-Path $ev "$name.stderr.txt")
    $t1 = Stamp
    Note "command ${name}: $py $($argv -join ' ')"
    Note "start: $t0"
    Note "end: $t1"
    Note "exit_code: $($p.ExitCode)"
    Note ''
    return $p.ExitCode
}
$code = 'none'
try {
    Set-Location $pkg
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    $env:CUDA_VISIBLE_DEVICES = '-1'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'
    Note "launch wrapper: powershell pid=$PID started $(Stamp)"
    Note "cwd: $pkg"
    Note "env: CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=$env:HF_HUB_OFFLINE PYTHONDONTWRITEBYTECODE=$env:PYTHONDONTWRITEBYTECODE PYTHONPATH=[$env:PYTHONPATH]"
    Note "before: $(Load)"
    Note ''
    $null = Run 'status_before' @('-B', 'ucov_supervise.py', '--status')
    $s = Get-Content (Join-Path $ev 'status_before.stdout.txt') -Raw | ConvertFrom-Json
    if ($s.invocations -ne 0 -or $s.starts -ne 0 -or $s.run_root_exists) {
        Note 'NOT LAUNCHED: status_before does not show 0 invocations, 0 starts and no run root'
        $code = 'not_launched'
    } else {
        $code = Run 'scientific' @('-B', 'ucov_supervise.py', '--scientific', '--frozen-ucov-manifest-sha256', $sha)
        Note "after: $(Load)"
        Note ''
        $null = Run 'status_after' @('-B', 'ucov_supervise.py', '--status')
    }
} catch {
    Note "WRAPPER ERROR: $($_.Exception.GetType().FullName): $($_.Exception.Message)"
} finally {
    Note "LAUNCH_SCRIPT_END scientific_exit_code=$code at $(Stamp)"
}
