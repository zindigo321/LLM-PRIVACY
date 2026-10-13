# 启动前聚焦检查（r1，2026-10-12）

结论：**通过**。未发现阻止合规执行的不一致。本检查由实现者完成，不是独立验收。

| 检查项 | 结果 | 依据（实测） |
|---|---|---|
| manifest 文件哈希 | `9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044`，与批准值一致 | PowerShell `Get-FileHash`，19:46:04 |
| manifest 条目 | 29/29 一致（哈希与字节数），其中 6 项来自冻结目录 | 同上，独立于 `uid_r1_control` 计算 |
| 冻结父清单 | 7/7 一致；SHA-256 `a80bebd4…86ca`，与 manifest_r1 的 `parent_manifest_sha256` 相同 | 同上 |
| 配置哈希 | 原始 `d7ae76052d50776705f386655b0b501d403104d7d4ca118865a2b6ff3b71b0bd`，规范 `bb0111fab6310ab5430d931b781fc3c97cdb928da773d0cf8e71ea532fb4e5b3` | `print_plan.out.txt`（包内 `verify_package`） |
| 实际导入的文件 | 子进程按路径导入冻结的 `uid_common`/`uformer_ae`/`uformer_blocks` 和 r1 模块，并核对 6 个模块的哈希（列于 manifest）；禁止导入 `cg_fedllm`。final4 lifecycle 子进程记录的路径与哈希都与 manifest 一致 | `%TEMP%\uid_r1_final4_20261012\…\child_record.json`；`print_plan.out.txt` |
| 命令、目录、解释器、环境 | 与 `PROPOSED_LAUNCH.md` 一致：cwd 为 r1 目录；`D:/conda_envs/cgfedllm/python.exe -B`；`CUDA_VISIBLE_DEVICES=-1`、`HF_HUB_OFFLINE=1`、`PYTHONDONTWRITEBYTECODE=1`；本 shell 未设置 `PYTHONPATH`；supervisor 在设置了 `PYTHONPATH` 时会拒绝运行 | 19:47:25 |
| 没有正在运行的相关 worker | Python 进程数 0；没有命令行含 `uid_r1`/`uid_runner`/`uid_20261012` 的进程 | 19:47:25 |
| ledger 允许执行 | `--status`：invocations 0，starts 0，U-CHECK/U-FIT 均为 NOT_STARTED；两个 ledger 根目录和运行根目录都不存在 | 19:47:25 起 |
| 以往的登记尝试 | 仓库内（`.git` 除外）、`D:\cgfed-runs`、`%TEMP%\uid_20261012_v2` 中没有 `u_check/u_fit.json`，也没有 `start_/end_U-*` 文件。唯一命中的是 `phase5_p5a_v2_seed1\ledger\invocation_1.json`，属于已关闭的 P5-A，与本实验无关 | 19:47:25 |
| CPU、线程、确定性 | 计划为 `cpu = {intra 24, inter 24}`；子进程在构建模型前调用冻结的 `configure_cpu`，与请求不符即判 EXECUTION_ERROR。final4 fixture 运行中观察到的值与请求一致（2/1，deterministic True，CUDA False） | `uid_r1_child.py:54-66`；final4 子进程记录 |
| 时限 | U-CHECK 300 s → 软停止 220 s，强制结束 280 s；U-FIT 1,800 s → 1,720/1,780 s（均从各自的 start 记录算起）；累计 7,200 s，从 supervisor 导入时开始计时，两臂共用 | `print_plan.out.txt`；`uid_r1_supervise.py:29,75,261` |
| 启动次数 | 最多 2 次启动、1 次 invocation；U-FIT 的前提是 U-CHECK 为 COMPLETED_PASS | `uid_r1_control.py:153-172`；plan |
| RSS 保护 | 上限为 min(4 GiB, 可用内存 50%)。19:51 时可用 8,692,248,576 B，上限因此为 4,294,967,296 B；每 0.25 s 采样子进程树，超限即强制结束 | `print_plan.out.txt`；`uid_r1_supervise.py:195,230` |
| 证据大小 | 512 MiB，事后比较（`evidence_within_cap`） | `uid_r1_supervise.py:303` |
| 工程证据与批准字节的对应 | final4 开始于 19:23:03.718，晚于所有代码和 checks 的最后修改（19:23:03.355）。final4 之后写入的 manifest 条目只有 `review_package/runner_diff.patch`（文档）。诊断目录下没有 `__pycache__` | 19:47 起的 mtime 比较 |

已知的局限（用户已接受）：见 `EXECUTION_AUTHORIZATION.md`。

启动命令（原样取自 `PROPOSED_LAUNCH.md`）：

```powershell
Set-Location D:\projects-cd\cg-fedllm-reproduction\diagnostics\uid_20261012_v2_r1
$env:CUDA_VISIBLE_DEVICES = "-1"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONDONTWRITEBYTECODE = "1"
& "D:/conda_envs/cgfedllm/python.exe" -B uid_r1_supervise.py --status
& "D:/conda_envs/cgfedllm/python.exe" -B uid_r1_supervise.py --scientific `
    --approved-r1-manifest-sha256 9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044
```

`Remove-Item Env:PYTHONPATH` 不执行：本 shell 未设置该变量，而且在这个工具环境中该命令会被拦截（工程修订时见过一次）。
supervisor 自身也会在设置了 `PYTHONPATH` 时拒绝运行。
