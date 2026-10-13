# CHANGELOG：UID-20261012-v2 → r1

父协议/清单：`uid_20261012_v2/protocol.md`（`de24e62f…dae0`）；`uid_20261012_v2/manifest.json`（`a80bebd4…86ca`）。
完整哈希以 `manifest_r1.json` 为准。

## A. 执行机制的变更（新文件，均位于 `uid_20261012_v2_r1/`）

- 新增 `uid_r1_supervise.py`：取代冻结的 `uid_runner.py` 作为启动入口。负责一次 invocation、按顺序依次执行各臂、为每个臂启动一个新的子进程，并执行单臂时限、累计时限和 RSS 上限。提供只读的 `--status`。
- 新增 `uid_r1_child.py`：每个臂在一个新进程中运行，执行登记的训练与测量，带 watchdog（supervisor 消失或到达截止时间即退出），并按哈希核对所导入的模块。
- 新增 `uid_r1_control.py`（不导入 torch）：包含双根目录、只写一次的 ledger，启动计数和前置条件，截止时间计算，包身份校验，以及登记计划（数值取自冻结 config 和 protocol）。
- 新增 `uid_r1_measure.py`：在隔离副本上做测量，计算分组梯度与参数变化，提供 RSS 采样器。
- 新增 `execution_r1.json`：只含执行参数（运行根目录、镜像、解释器、torch 版本、各类余量与间隔）。不含任何科学字段。
- 冻结的 `uid_runner.py` 和 `checks/check_*.py` 保留原样，不再作为启动入口。

## B. 工程检查（`checks/`，只用 fixture）

`test_r1_control.py`、`test_r1_measure.py`、`test_r1_lifecycle.py`（真实子进程）、`check_upstream_correspondence.py`、`check_upstream_negative_control.py`、`check_resnet_sampling.py`、`probe_cpu_cost.py`、`fixture_run.py`、`fixture_config.json`（`UID-R1-ENGINEERING-FIXTURE`）、`summarize_correspondence.py`、`run_all_checks.ps1`。

## C. 文档更正

- `REVISION_NOTE.md`：取代 PREPARE 阶段那些没有依据的说法（确定性、线程数、冻结时间、`week_plan`、上游等价、授权时间、检查时间线）。
- `protocol_addendum_r1.md`：只列执行机制的变更，以及测量记录方式的变更；后者不改变任何判定依据。
- `week_plan.md`：补交。
- 原有记录（`current_state.md`、`protocol.md`、`DISCREPANCY.md` 等）不修改。

## D. 未改变的科学设置（以冻结文件的字节为准）

- `uformer_blocks.py`、`uformer_ae.py`、`uid_common.py`、`config.json` 字节不变（r1 从冻结目录导入，不复制）。
- 输入：data seed 20261012，`1.6*rand([1,1,32,32])-0.8`；模型 seed 1；U-FIT 模型配置；优化器（U-CHECK Adam 1e-2，U-FIT Adam 2e-4，betas (0.9, 0.999)，eps 1e-8，wd 0）；更新次数 500/3,000；时限 300/1,800/7,200 s；RSS 上限 min(4 GiB, 可用内存 50%)；最多 2 次启动；判定门槛 RSE ≤ 0.01、cosine ≥ 0.99，以 eval 模式为准；结果分类；U-FIT 只有在 U-CHECK PASS 后才运行。
- 度量函数（`uid_common.metrics`/`passes`/`baselines`）与重建路径（`uid_common.reconstruct`）未变，由冻结模块提供。

## E. 运行行为的差异（与冻结 runner 相比；只影响测量方式，不影响判定依据）

- 第 0 步：冻结 runner 用训练模型以 train 模式测量（会改动 BN），r1 用隔离副本，以 eval 和 train 两种模式测量（R8）。
- 最终评估：两份独立副本从同一个最终 state_dict 加载，并从相同的 CPU RNG 状态开始。新增两项记录：两副本加载状态是否一致，以及 train 副本在前向后状态是否改变。
- 曲线：冻结 runner 记录每 100 次更新和最后一次，r1 另加第 1 次（协议要求），并区分更新前训练损失与更新后参数变化。
- 优化器：冻结 runner 的 eps 和 weight_decay 是硬编码值，r1 从冻结 config 读取，数值相同（1e-8、0）。
