# UID-20261012-v2 续作检查点

更新：2026-10-12（Asia/Shanghai）。阶段：执行前一致性检查 → **BLOCKED（runner 不符合批准条件）**。

## 当前事实（磁盘核实）

- 科学训练：**未开始**。训练启动计数 0；没有调用任何优化器更新；`output/` 不存在。
- 包身份：恢复后冻结清单 7/7 一致。`uformer_blocks.py` `3333f186…0b747`、`uformer_ae.py` `a3ad6e3f…c8606`；配置原始哈希
  `d7ae7605…71b0bd`，规范哈希 `bb0111fa…4e5b3`（完整值见 `prelaunch_20261012/prelaunch_check.md`）。
- 恢复：授权范围内只逆转两次 docstring 编辑，只改动这两个文件。证据与字节数解释见 `incident_evidence_20261012/`
  （`DISCREPANCY.md` 保留原样；更正见 `DISCREPANCY_RESOLUTION.md`；`restoration_log.json`）。
- 检查 1、2 已在恢复后的文件上重跑，输出保存在 `prelaunch_20261012/`（退出码均为 0）。
- 阻塞项 R1–R10：冻结 `uid_runner.py` 不执行单臂时限、累计时限和启动计数，没有 start 记录，不固定线程与确定性设置，
  不记录梯度、参数变化、第 1 步和峰值 RSS，第 0 步前向会改动 BN 缓冲区。详见 `prelaunch_check.md` §5。

## 下一步（需人类决定，见 prelaunch_check.md §7）

A：登记 runner 修订，然后进行工程检查、生成新清单、单独批准（建议）；B：书面接受偏差后按现状执行；C：停止。

## 禁止事项（不变）

不训练；不改冻结协议、配置、清单；不重试，不加 seed；不碰 SSBD、P5-A、GPU、FAF；不 commit、push、PR。

## 状态

UID：**BLOCKED**（启动 0/2）。SSBD：BLOCKING DEFECTS FOUND / HUMAN ACCEPTANCE PENDING / SSBD EXECUTION PAUSED（未改动）。
