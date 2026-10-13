# ENGINEERING: run every URC-v1 check once on the current bytes; one log per check with command, timestamps, exit code.
# Usage (from the URC directory): powershell -NoProfile -File checks/run_all_checks.ps1 <evidence_subdir>
# No registered model is optimized: lifecycle/analyze tests use the fixture config; check_reference_input builds only
# the registered input and baselines; check_init_comparison builds fresh models without a forward pass or optimizer.
param([Parameter(Mandatory = $true)][string]$Tag)
$ErrorActionPreference = 'Continue'
$urc = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ev = Join-Path $urc "evidence\$Tag"
New-Item -ItemType Directory -Path $ev | Out-Null
Set-Location $urc
$py = 'D:/conda_envs/cgfedllm/python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
if ($env:PYTHONPATH) { throw 'PYTHONPATH must not be set for URC checks' }
$utf8 = New-Object Text.UTF8Encoding $false
$pt = @('-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--rootdir=.', '-rA')
$steps = @(
    @{ n = 'ruff_check';            a = @('-B', '-m', 'ruff', 'check', '--no-cache', '--config', '../../pyproject.toml', '.') },
    @{ n = 'ruff_format_check';     a = @('-B', '-m', 'ruff', 'format', '--check', '--no-cache', '--config', '../../pyproject.toml', '.') },
    @{ n = 'test_urc_model';        a = $pt + @('checks/test_urc_model.py') },
    @{ n = 'test_urc_control';      a = $pt + @("--basetemp=$env:TEMP\urc_v1_ctl_$Tag", 'checks/test_urc_control.py') },
    @{ n = 'test_urc_lifecycle';    a = $pt + @("--basetemp=$env:TEMP\urc_v1_lc_$Tag", 'checks/test_urc_lifecycle.py') },
    @{ n = 'test_urc_analyze';      a = $pt + @("--basetemp=$env:TEMP\urc_v1_an_$Tag", 'checks/test_urc_analyze.py') },
    @{ n = 'check_reference_input'; a = @('-B', 'checks/check_reference_input.py') },
    @{ n = 'check_init_comparison'; a = @('-B', 'checks/check_init_comparison.py') },
    @{ n = 'probe_cpu_cost';        a = @('-B', 'checks/probe_cpu_cost.py', '6') },
    @{ n = 'status_readonly';       a = @('-B', 'urc_supervise.py', '--status') }
)
$summary = @()
foreach ($s in $steps) {
    $t0 = Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'
    $out = & $py @($s.a) 2>&1 | ForEach-Object { "$_" } | Out-String
    $code = $LASTEXITCODE
    $t1 = Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'
    $text = "command: $py $($s.a -join ' ')`ncwd: $urc`nstart: $t0`nend: $t1`nexit_code: $code`n--- output (stdout+stderr) ---`n$out"
    [IO.File]::WriteAllText((Join-Path $ev "$($s.n).log"), $text, $utf8)
    $summary += "{0,-24} exit={1}  {2} -> {3}" -f $s.n, $code, $t0, $t1
}
[IO.File]::WriteAllText((Join-Path $ev 'SUMMARY.txt'), ($summary -join "`n") + "`n", $utf8)
$summary
