# Engineering check (no package file, no model, no run root): is a process started with Start-Process -WindowStyle
# Hidden outside a kill-on-close job, so that it outlives the tool shell that started it? Writes job information for
# this shell and for a detached child to process_detach_check.txt in this directory.
param([string]$Role = 'parent')
$ErrorActionPreference = 'Stop'
$here = 'D:\projects-cd\cg-fedllm-reproduction\diagnostics\urp_20261013_v1_execution'
$out = Join-Path $here 'process_detach_check.txt'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class JobProbe {
  [DllImport("kernel32.dll", SetLastError = true)]
  public static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool result);
  [DllImport("kernel32.dll")]
  public static extern IntPtr GetCurrentProcess();
  [StructLayout(LayoutKind.Sequential)]
  public struct Basic {
    public long PerProcessUserTimeLimit; public long PerJobUserTimeLimit; public uint LimitFlags;
    public UIntPtr MinimumWorkingSetSize; public UIntPtr MaximumWorkingSetSize; public uint ActiveProcessLimit;
    public UIntPtr Affinity; public uint PriorityClass; public uint SchedulingClass;
  }
  [StructLayout(LayoutKind.Sequential)]
  public struct Io { public ulong R, W, O, RB, WB, OB; }
  [StructLayout(LayoutKind.Sequential)]
  public struct Ext {
    public Basic B; public Io I; public UIntPtr ProcessMemoryLimit; public UIntPtr JobMemoryLimit;
    public UIntPtr PeakProcessMemoryUsed; public UIntPtr PeakJobMemoryUsed;
  }
  [DllImport("kernel32.dll", SetLastError = true)]
  public static extern bool QueryInformationJobObject(IntPtr job, int cls, out Ext info, int len, IntPtr ret);
}
'@
function Describe([string]$who) {
    $in = $false
    [void][JobProbe]::IsProcessInJob([JobProbe]::GetCurrentProcess(), [IntPtr]::Zero, [ref]$in)
    $line = "$who pid=$PID ppid=$((Get-CimInstance Win32_Process -Filter "ProcessId=$PID").ParentProcessId) in_job=$in"
    if ($in) {
        $e = New-Object JobProbe+Ext
        $ok = [JobProbe]::QueryInformationJobObject([IntPtr]::Zero, 9, [ref]$e, [Runtime.InteropServices.Marshal]::SizeOf($e), [IntPtr]::Zero)
        $f = $e.B.LimitFlags
        $line += (" query_ok={0} flags=0x{1:X} kill_on_close={2} breakaway_ok={3} silent_breakaway_ok={4}" -f `
            $ok, $f, (($f -band 0x2000) -ne 0), (($f -band 0x800) -ne 0), (($f -band 0x1000) -ne 0))
    }
    $line += " time=$((Get-Date).ToString('yyyy-MM-ddTHH:mm:ss.fffzzz'))"
    [IO.File]::AppendAllText($out, $line + "`n", (New-Object Text.UTF8Encoding $false))
}
if ($Role -eq 'parent') {
    if (Test-Path $out) { throw "$out exists" }
    Describe 'tool-shell'
    $p = Start-Process -FilePath powershell.exe -WindowStyle Hidden -PassThru -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $here 'process_detach_check.ps1'), '-Role', 'child')
    [IO.File]::AppendAllText($out, "detached child started pid=$($p.Id); tool shell exits now`n", (New-Object Text.UTF8Encoding $false))
} else {
    Describe 'detached-child (at start)'
    Start-Sleep -Seconds 25
    Describe 'detached-child (25 s later, after the tool shell exited)'
}
