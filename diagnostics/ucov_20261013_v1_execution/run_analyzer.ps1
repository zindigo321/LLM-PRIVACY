# UCOV-20261013-v1: run the frozen analyzer once and record the command, the hashes of the files it runs, times,
# exit code, stdout and stderr in this directory. Refuses if analysis/ or any of its own output files already exist.
$ErrorActionPreference = 'Stop'
$pkg = 'D:\projects-cd\cg-fedllm-reproduction\diagnostics\ucov_20261013_v1'
$here = 'D:\projects-cd\cg-fedllm-reproduction\diagnostics\ucov_20261013_v1_execution'
$dest = Join-Path $here 'analysis'
$py = 'D:/conda_envs/cgfedllm/python.exe'
$log = Join-Path $here 'ucov_analyze.command.txt'
$out = Join-Path $here 'ucov_analyze.out.txt'
$err = Join-Path $here 'ucov_analyze.err.txt'
foreach ($p in $dest, $log, $out, $err) { if (Test-Path $p) { throw "$p exists; never overwritten" } }
$utf8 = New-Object Text.UTF8Encoding $false
function Note([string]$s) { [IO.File]::AppendAllText($log, $s + "`n", $utf8) }
function Stamp { (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss.fffzzz') }
Set-Location $pkg
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:CUDA_VISIBLE_DEVICES = '-1'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'
foreach ($f in 'manifest_ucov.json', 'ucov_analyze.py', 'ucov_control.py', '..\upp_20261013_v1\upp_classify.py', '..\upp_20261013_v1\upp_policy.json') {
    Note "sha256 before run: $f $((Get-FileHash -Algorithm SHA256 (Join-Path $pkg $f)).Hash.ToLower())"
}
Note "cwd: $pkg"
Note "env: CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=$env:HF_HUB_OFFLINE PYTHONDONTWRITEBYTECODE=$env:PYTHONDONTWRITEBYTECODE PYTHONPATH=[$env:PYTHONPATH]"
$argv = @('-B', 'ucov_analyze.py', '--evidence-dir', $dest)
$t0 = Stamp
$p = Start-Process -FilePath $py -ArgumentList $argv -WorkingDirectory $pkg -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err
$t1 = Stamp
Note "command: $py $($argv -join ' ')"
Note "start: $t0"
Note "end: $t1"
Note "exit_code: $($p.ExitCode)"
