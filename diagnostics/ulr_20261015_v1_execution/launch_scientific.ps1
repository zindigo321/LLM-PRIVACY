# ULR-20261015-v1: the one authorized launch, with stdout/stderr, times and exit codes saved under launch/.
# Usage: powershell -NoProfile -File launch_scientific.ps1 <manifest_sha256>
# Refuses to start if launch/ exists, or if --status does not show 0 invocations, no records and no run root.
# Started detached (Start-Process -WindowStyle Hidden), so an interrupted chat or tool call neither stops the bounded
# run nor leads to a second launch. The last line of COMMANDS_AND_EXIT_CODES.txt is "LAUNCH_SCRIPT_END ...".
param([Parameter(Mandatory = $true)][string]$Sha)
$ErrorActionPreference = 'Stop'
$pkg = 'D:\projects-cd\cg-fedllm-ulr\diagnostics\ulr_20261015_v1'
$ev = 'D:\projects-cd\cg-fedllm-ulr\diagnostics\ulr_20261015_v1_execution\launch'
$py = 'D:/conda_envs/cgfedllm/python.exe'
New-Item -ItemType Directory -Path $ev | Out-Null  # throws if launch/ exists: never a second launch
$utf8 = New-Object Text.UTF8Encoding $false
$log = Join-Path $ev 'COMMANDS_AND_EXIT_CODES.txt'
function Note([string]$s) { [IO.File]::AppendAllText($log, $s + "`n", $utf8) }
function Stamp { (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss.fffzzz') }
function Load {
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
    Note "command ${name}: $py $($argv -join ' ')"
    Note "start: $t0"
    Note "end: $(Stamp)"
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
    Note "manifest: $Sha"
    Note "env: CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=$env:HF_HUB_OFFLINE PYTHONDONTWRITEBYTECODE=$env:PYTHONDONTWRITEBYTECODE PYTHONPATH=[$env:PYTHONPATH]"
    Note "before: $(Load)"
    Note ''
    $null = Run 'status_before' @('-B', 'ulr_supervise.py', '--status')
    $s = Get-Content (Join-Path $ev 'status_before.stdout.txt') -Raw | ConvertFrom-Json
    if ($s.invocations -ne 0 -or $s.run_root_exists -or $s.records.Count -ne 0) {
        Note 'NOT LAUNCHED: status_before does not show 0 invocations, no records and no run root'
        $code = 'not_launched'
    } else {
        $code = Run 'scientific' @('-B', 'ulr_supervise.py', '--scientific', '--frozen-ulr-manifest-sha256', $Sha)
        Note "after: $(Load)"
        Note ''
        $null = Run 'status_after' @('-B', 'ulr_supervise.py', '--status')
    }
} catch {
    Note "WRAPPER ERROR: $($_.Exception.GetType().FullName): $($_.Exception.Message)"
} finally {
    Note "LAUNCH_SCRIPT_END scientific_exit_code=$code at $(Stamp)"
}
