# UID-20261012-v2-r1 执行授权记录

记录时间（本文件写入前取自系统时钟）：**2026-10-12T19:46:41.504+08:00**（China Standard Time）。
这是记录写入的时间，不是用户批准的时间：用户的批准消息不带时间戳，批准时间为 UNKNOWN（早于本记录）。

本授权依据实现者的评审包和其中报告的工程检查，**不是独立验收**。

## 被授权的对象

- 仓库：`D:/projects-cd/cg-fedllm-reproduction`
- 修订：`diagnostics/uid_20261012_v2_r1/`
- 清单：`manifest_r1.json`，批准的 SHA-256：`9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044`
- 范围：在 CPU 上执行一次有界运行：U-CHECK；当且仅当 U-CHECK 为 COMPLETED_PASS 时运行 U-FIT；然后做规定的最终评估并写报告。不得重试、续跑、调参、延长预算、挑选检查点，也不得开始其他实验。SSBD 继续暂停。

## 用户指示（摘录）

> I approve one bounded scientific execution of the revised Uformer diagnostic package identified below, conditional
> on the focused pre-launch checks in this message passing. [...] Approved manifest SHA-256:
> 9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044. This authorization is based on the implementation
> review package and reported engineering checks. Do not describe it as independent acceptance.

## 用户要求记录的澄清（用户已接受以下局限）

1. 原 runner 缺少 PREPARE 阶段报告中声称已实现、实际并未实现的执行保障。r1 包实现了这些要求。阅读 `review_package/CHANGELOG.md` 时须结合这一更正：凡是写"取代冻结 runner 的行为"的地方，应理解为取代 PREPARE 中没有依据的说法以及当时缺失的保障。按用户指示，不为修正这句描述而重新生成清单。
2. 已报告的 r1 ledger 中 invocation 为 0、start 为 0。这不能证明在已观察的会话之外从未发生过未被记录的执行。
3. RSS 证据是以 0.25 s 间隔采样得到的最大值，不是连续测量的峰值。

## 本文件之外的证据

- 启动前检查：`prelaunch_check_r1.md`（本目录）
- 运行输出：`D:/cgfed-runs/uid_20261012_v2_r1/`，以及 ledger 镜像 `diagnostics/uid_20261012_v2_r1/scientific_ledger_mirror/`
- 启动命令的原始 stdout/stderr 与退出码：本目录 `launch/`
