# 拟议的科学启动命令（未执行；需对下列 manifest 哈希单独书面批准）

批准对象：`manifest_r1.json`，SHA-256 `9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044`（29 个文件）。
本文件不在 manifest 中，因为它要引用 manifest 的哈希。

```powershell
Set-Location D:\projects-cd\cg-fedllm-reproduction\diagnostics\uid_20261012_v2_r1
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue   # 若已设置；supervisor 在设置了 PYTHONPATH 时拒绝运行
$env:CUDA_VISIBLE_DEVICES = "-1"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONDONTWRITEBYTECODE = "1"
& "D:/conda_envs/cgfedllm/python.exe" -B uid_r1_supervise.py --status
& "D:/conda_envs/cgfedllm/python.exe" -B uid_r1_supervise.py --scientific `
    --approved-r1-manifest-sha256 9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044
```

- 一次 invocation 依次运行 U-CHECK，再运行 U-FIT（仅当 U-CHECK 为 COMPLETED_PASS），然后写 end 记录。不会重试，也不会续跑。
- 输出：`D:/cgfed-runs/uid_20261012_v2_r1/`（`ledger/`、`arms/U-CHECK/`、`arms/U-FIT/`）；ledger 镜像在
  `diagnostics/uid_20261012_v2_r1/scientific_ledger_mirror/`。
- 退出码：0 = 各臂均为 COMPLETED_PASS、COMPLETED_FIT_FAIL 或 NOT_RUN；1 = 有臂为 RESOURCE_STOP、EXECUTION_ERROR 或 INCOMPLETE；
  2 = 启动前被拒绝（包身份不符、ledger 状态不允许、设置了 PYTHONPATH 等）。
