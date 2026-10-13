# ENGINEERING: run every UCOV-v1 check once on the current bytes; one log per check with command, timestamps, exit code.
# Usage (from the UCOV directory): powershell -NoProfile -File checks/run_all_checks.ps1 <evidence_subdir>
# No registered model is optimized: lifecycle/analyze tests use the fixture config and the toy tiles;
# check_registered_binding builds the registered model and loads the registered tiles for one forward pass each on a
# disposable copy, with no optimizer and no metric of the output.
param([Parameter(Mandatory = $true)][string]$Tag)
$ErrorActionPreference = 'Continue'
$ucov = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ev = Join-Path $ucov "evidence\$Tag"
New-Item -ItemType Directory -Path $ev | Out-Null
Set-Location $ucov
$py = 'D:/conda_envs/cgfedllm/python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
if ($env:PYTHONPATH) { throw 'PYTHONPATH must not be set for UCOV checks' }
$utf8 = New-Object Text.UTF8Encoding $false
$pt = @('-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--rootdir=.', '-rA')
$cfg = @('--no-cache', '--config', '../../pyproject.toml')
$steps = @(
    @{ n = 'ruff_check';               a = @('-B', '-m', 'ruff', 'check') + $cfg + @('.', '../upp_20261013_v1') },
    @{ n = 'ruff_format_check';        a = @('-B', '-m', 'ruff', 'format', '--check') + $cfg + @('.', '../upp_20261013_v1') },
    @{ n = 'test_upp_classify';        a = $pt + @("--basetemp=$env:TEMP\ucov_upp_$Tag", '../upp_20261013_v1/test_upp_classify.py') },
    @{ n = 'test_ucov_input_rule';     a = $pt + @("--basetemp=$env:TEMP\ucov_in_$Tag", 'checks/test_ucov_input_rule.py') },
    @{ n = 'test_ucov_control';        a = $pt + @("--basetemp=$env:TEMP\ucov_ctl_$Tag", 'checks/test_ucov_control.py') },
    @{ n = 'test_ucov_lifecycle';      a = $pt + @("--basetemp=$env:TEMP\ucov_lc_$Tag", 'checks/test_ucov_lifecycle.py') },
    @{ n = 'test_ucov_analyze';        a = $pt + @("--basetemp=$env:TEMP\ucov_an_$Tag", 'checks/test_ucov_analyze.py') },
    @{ n = 'check_registered_binding'; a = @('-B', 'checks/check_registered_binding.py') },
    @{ n = 'status_readonly';          a = @('-B', 'ucov_supervise.py', '--status') }
)
$summary = @()
foreach ($s in $steps) {
    $t0 = Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'
    $out = & $py @($s.a) 2>&1 | ForEach-Object { "$_" } | Out-String
    $code = $LASTEXITCODE
    $t1 = Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'
    $text = "command: $py $($s.a -join ' ')`ncwd: $ucov`nstart: $t0`nend: $t1`nexit_code: $code`n--- output (stdout+stderr) ---`n$out"
    [IO.File]::WriteAllText((Join-Path $ev "$($s.n).log"), ($text -replace "`r`n", "`n"), $utf8)
    $summary += "{0,-26} exit={1}  {2} -> {3}" -f $s.n, $code, $t0, $t1
}
[IO.File]::WriteAllText((Join-Path $ev 'SUMMARY.txt'), ($summary -join "`n") + "`n", $utf8)
$summary
