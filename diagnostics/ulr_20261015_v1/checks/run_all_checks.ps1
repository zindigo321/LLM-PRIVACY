# ULR-20261015-v1 pre-freeze checks. Usage (from the package directory, PYTHONPATH unset):
#   powershell -NoProfile -File checks/run_all_checks.ps1 <tag>
# Writes evidence/<tag>/<check>.log and evidence/<tag>/SUMMARY.txt; refuses if evidence/<tag> exists.
# ruff excludes vendor/ (byte copies of frozen sources). The UPP-20261014-v1 policy tests run in place in the UCX
# worktree (read-only: no bytecode, no cache provider).
param([Parameter(Mandatory = $true)][string]$Tag)
$ErrorActionPreference = 'Stop'
$pkg = Split-Path -Parent $PSScriptRoot
$ev = Join-Path $pkg "evidence\$Tag"
New-Item -ItemType Directory -Path $ev | Out-Null
$py = 'D:/conda_envs/cgfedllm/python.exe'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:CUDA_VISIBLE_DEVICES = '-1'; $env:PYTHONDONTWRITEBYTECODE = '1'; $env:HF_HUB_OFFLINE = '1'
$utf8 = New-Object Text.UTF8Encoding $false
$summary = @()
function Check([string]$name, [string[]]$argv) {
    $log = Join-Path $ev "$name.log"
    $out = & $py @argv 2>&1 | Out-String
    $code = $LASTEXITCODE
    [IO.File]::WriteAllText($log, "command: $py $($argv -join ' ')`nexit_code: $code`n`n$out", $utf8)
    $script:summary += "$name exit=$code"
}
Set-Location $pkg
Check 'ruff_check' @('-m', 'ruff', 'check', '.', '--exclude', 'vendor')
Check 'ruff_format_check' @('-m', 'ruff', 'format', '--check', '.', '--exclude', 'vendor')
Check 'test_ulr_units' @('-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'checks/test_ulr_units.py')
Check 'test_ulr_lifecycle' @('-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'checks/test_ulr_lifecycle.py')
Check 'test_upp_20261014_classify_in_place' @('-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'D:/projects-cd/cg-fedllm-ucx/diagnostics/upp_20261014_v1/test_upp_20261014_classify.py')
Check 'status_readonly' @('-B', 'ulr_supervise.py', '--status')
[IO.File]::WriteAllText((Join-Path $ev 'SUMMARY.txt'), (($summary -join "`n") + "`n"), $utf8)
$summary
