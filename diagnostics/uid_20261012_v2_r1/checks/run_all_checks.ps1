# ENGINEERING: run every r1 check once on the current bytes; one log per check with command, timestamps and exit code.
# Usage (from the r1 directory): powershell -NoProfile -File checks/run_all_checks.ps1 <evidence_subdir>
param([Parameter(Mandatory = $true)][string]$Tag)
# Continue (not Stop): in Windows PowerShell 5.1 a native command's stderr line becomes an ErrorRecord, and Stop
# would abort the whole script on a mere warning. Exit codes are recorded per step instead.
$ErrorActionPreference = 'Continue'
$r1 = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ev = Join-Path $r1 "evidence\$Tag"
New-Item -ItemType Directory -Path $ev | Out-Null
Set-Location $r1
$py = 'D:/conda_envs/cgfedllm/python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
if ($env:PYTHONPATH) { throw 'PYTHONPATH must not be set for r1 checks' }
$utf8 = New-Object Text.UTF8Encoding $false
$steps = @(
    @{ n = 'ruff_check';        a = @('-B', '-m', 'ruff', 'check', '--no-cache', '--config', '../../pyproject.toml', '.') },
    @{ n = 'ruff_format_check'; a = @('-B', '-m', 'ruff', 'format', '--check', '--no-cache', '--config', '../../pyproject.toml', '.') },
    @{ n = 'test_r1_control';   a = @('-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--rootdir=.', '-rA', 'checks/test_r1_control.py') },
    @{ n = 'test_r1_measure';   a = @('-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--rootdir=.', '-rA', 'checks/test_r1_measure.py') },
    @{ n = 'test_r1_metrics';   a = @('-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--rootdir=.', '-rA', 'checks/test_r1_metrics.py') },
    @{ n = 'test_r1_lifecycle'; a = @('-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--rootdir=.', '-rA', "--basetemp=$env:TEMP\uid_r1_$Tag", 'checks/test_r1_lifecycle.py') },
    @{ n = 'upstream_correspondence';  a = @('-B', 'checks/check_upstream_correspondence.py') },
    @{ n = 'upstream_negative_control'; a = @('-B', 'checks/check_upstream_negative_control.py') },
    @{ n = 'resnet_sampling';   a = @('-B', 'checks/check_resnet_sampling.py') }
)
$summary = @()
foreach ($s in $steps) {
    $t0 = Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'
    $out = & $py @($s.a) 2>&1 | ForEach-Object { "$_" } | Out-String
    $code = $LASTEXITCODE
    $t1 = Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'
    $text = "command: $py $($s.a -join ' ')`ncwd: $r1`nstart: $t0`nend: $t1`nexit_code: $code`n--- output (stdout+stderr) ---`n$out"
    [IO.File]::WriteAllText((Join-Path $ev "$($s.n).log"), $text, $utf8)
    $summary += "{0,-28} exit={1}  {2} -> {3}" -f $s.n, $code, $t0, $t1
}
[IO.File]::WriteAllText((Join-Path $ev 'SUMMARY.txt'), ($summary -join "`n") + "`n", $utf8)
$summary
