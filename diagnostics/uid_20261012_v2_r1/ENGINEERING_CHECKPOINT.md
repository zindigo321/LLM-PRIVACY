# UID-20261012-v2 工程修订 r1：检查点

授权：用户选择方案 A（有界工程修订）。科学执行暂停，直到修订后的包经过展示并获得明确批准。SSBD 暂停。

## 已核实的事实

- 恢复（2026-10-12T17:09:34+08:00）：中断前，方案 A 的修订文件一个都没有落盘。原始冻结包 7/7 一致；`manifest.json` SHA-256
  `a80bebd414cad4aaf55c48cc5ca665ff658cf1388d687056db0c2ccc3cf286ca`。
- 登记的科学启动 **0/2**。依据：
  - `uid_r1_supervise.py --status`（只读）：invocations 0，starts 0，两个 ledger 根目录都不存在；
  - `D:\cgfed-runs\uid_20261012_v2_r1` 不存在；
  - 已知位置中没有 start、end、ledger 或 `u_check/u_fit.json`；
  - 本会话没有调用过 `uid_runner.py`，也没有以 `--scientific` 调用过 r1。
  - 局限：冻结的 runner 训练前不写任何文件；如果它在本会话之外运行后被强制终止，不会留下痕迹，无法排除。
- 冻结目录在 ruff format 前后哈希一致（`frozen_tree_unchanged=True`）。r1 只格式化了自己的文件。

## 进度

| 步骤 | 状态 | 证据（最终字节：`evidence/final3_20261012/`） |
|---|---|---|
| 1 ledger、启动计数、时限 | 已核实 | `test_r1_control.log`：13 passed |
| 2 无副作用测量 | 已核实 | `test_r1_measure.log`：7 passed |
| 3/4 子进程、supervisor、看门狗 | 已核实（真实子进程） | `test_r1_lifecycle.log`：13 passed |
| 5 上游对应性 + 负对照 | 已核实 | `upstream_correspondence.log`、`upstream_negative_control.log` |
| 6 采样层比较、CPU 成本探针 | 已核实 | `resnet_sampling.log`；`evidence/probe_cpu_cost.*`（格式化前运行，见下文） |
| 7 ruff | 已核实 | `ruff_check.log`、`ruff_format_check.log` |
| 7b 度量解析用例（`test_r1_metrics.py`） | 已核实：12 passed（第 1 次运行时 float32 舍入导致 1 项失败，已保留日志） | `evidence/test_r1_metrics_run1_failed.log`、`_run3.log` |
| 8 文档：修订说明、变更日志、协议附录、week_plan、diff | 完成 | `review_package/` |
| 9 manifest_r1 + 回读 | 完成并核实：29 个文件，0 处不符；错误哈希会被拒绝 | `manifest_r1.json` SHA-256 `9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044` |
| 最终字节全量检查 | 全部 exit 0 | `evidence/final4_20261012/`（manifest 构建前的最终字节） |

日志说明：`evidence/test_r1_metrics_run2.log` 的修改时间（19:21:35）晚于修复（19:21:20），但当时修复是否已被加载无法确定，
所以以 `_run3.log` 和 `final4` 为准。

## 状态

ENGINEERING REVISION READY FOR REVIEW（实现者自检，不是独立验收）。REGISTERED SCIENTIFIC STARTS: 0/2。
SCIENTIFIC EXECUTION PAUSED（需对 `9d3aba9a…8044` 单独批准）。SSBD EXECUTION PAUSED。

中间失败的记录保留不删：`evidence/test_r1_lifecycle.log`（2 failed，已修复）；`evidence/final_20261012/`
（PowerShell 的 Stop 设置导致中止）。

- CPU 成本探针在 ruff format 之前运行。它只读取冻结包，格式化没有改动冻结文件；探针脚本本身被格式化过，因此其结果对应格式化前
  的脚本字节。

## 下一步

等待人类评审 r1 包。获批后，先核对 manifest 哈希，再执行 `PROPOSED_LAUNCH.md` 中的命令。未经批准不得运行 `--scientific`。
