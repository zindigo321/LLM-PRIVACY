# UID-20261012-v2-r1 续作检查点（执行后）

更新时间：2026-10-12（Asia/Shanghai），运行结束之后。本文件是当前状态的权威记录。
`uid_20261012_v2_r1/ENGINEERING_CHECKPOINT.md` 和 `uid_20261012_v2/continuation_checkpoint.md` 是历史记录，保留原样不改。

## 已发生且已核实的事实

- 唯一一次科学 invocation：2026-10-12 19:53:29–20:01:59（+08:00），批准的 manifest 为 `9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044`，退出码 0。
- 结果：U-CHECK 为 COMPLETED_PASS（500 次更新，4.0 s）；U-FIT 为 COMPLETED_FIT_FAIL（3,000 次更新，506.3 s）。
  U-FIT 最终 eval 模式 RSE 0.2727、cosine 0.8540；train 模式 RSE 0.2721、cosine 0.8545。
- 执行后的 ledger（`--status`）：1 次 invocation，2/2 次启动，没有未完成的启动；镜像与主 ledger 逐字节相同。
- 没有发生资源停止、中断或执行错误；子进程和 supervisor 的 stderr 都为空。
- 冻结包与 r1 包都没有被修改。

## 当前状态

- 本协议已经关闭：启动配额用完，invocation 次数也用完。**不得重跑、续跑，也不得在本协议下追加运行。**
- 下一步需要人类决定。建议（见 `REPORT.md` §6）是另行登记一个只改更新预算的后续诊断，需要新的协议和新的批准。尚未开始。
- SSBD 暂停（B1–B3、E1–E3 不变）。没有执行 GPU、FAF 或 Git 发布操作。

## 证据位置

本目录：`REPORT.md`、`analysis.json`、`curve_u_fit.csv`、`launch/`、`prelaunch_check_r1.md`、`EXECUTION_AUTHORIZATION.md`、
`RUN_ARTIFACTS_SHA256.txt`、`FINAL_ARTIFACTS_SHA256.txt`。运行根目录：`D:/cgfed-runs/uid_20261012_v2_r1/`。
