# URC-20261013-v1: the approved launch (the PROPOSED_LAUNCH.md commands, unchanged), with stdout/stderr, times and
# exit codes saved under launch/. Refuses to start if launch/ exists or if --status shows any invocation or start.
$ErrorActionPreference = 'Stop'
$pkg = 'D:\projects-cd\cg-fedllm-reproduction\diagnostics\urc_20261013_v1'
$ev = 'D:\projects-cd\cg-fedllm-reproduction\diagnostics\urc_20261013_v1_execution\launch'
$py = 'D:/conda_envs/cgfedllm/python.exe'
$sha = 'b058adf35bf326516f17cb3d773222840b0d1f6605a0f6e4a535564e94588cb6'
New-Item -ItemType Directory -Path $ev | Out-Null
$utf8 = New-Object Text.UTF8Encoding $false
$log = Join-Path $ev 'COMMANDS_AND_EXIT_CODES.txt'
function Note([string]$s) { [IO.File]::AppendAllText($log, $s + "`n", $utf8) }
function Stamp { (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss.fffzzz') }
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
Set-Location $pkg
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:CUDA_VISIBLE_DEVICES = '-1'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'
Note "cwd: $pkg"
Note "env: CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=$env:HF_HUB_OFFLINE PYTHONDONTWRITEBYTECODE=$env:PYTHONDONTWRITEBYTECODE PYTHONPATH=[$env:PYTHONPATH]"
Note ''
$null = Run 'status_before' @('-B', 'urc_supervise.py', '--status')
$s = Get-Content (Join-Path $ev 'status_before.stdout.txt') -Raw | ConvertFrom-Json
if ($s.invocations -ne 0 -or $s.starts -ne 0 -or $s.run_root_exists) {
    Note 'NOT LAUNCHED: status_before does not show 0 invocations, 0 starts and no run root'
    exit 3
}
$code = Run 'scientific' @('-B', 'urc_supervise.py', '--scientific', '--approved-urc-manifest-sha256', $sha)
$null = Run 'status_after' @('-B', 'urc_supervise.py', '--status')
exit $code
