# UID-20261012-v2 r1 修订说明（权威版本，取代下列不实说法）

- 父协议：`UID-20261012-v2`，`uid_20261012_v2/protocol.md`，SHA-256 `de24e62ffc32fc72bfb79b3d32e32bda5f9d21d1b2b1432c735d6ba4f3c6dae0`。
- 父清单：`uid_20261012_v2/manifest.json`，SHA-256 `a80bebd414cad4aaf55c48cc5ca665ff658cf1388d687056db0c2ccc3cf286ca`。
- 修订：`UID-20261012-v2-r1`，只涉及执行机制。原始文件、事件证据和以前的更正一律保留原样。
- 本说明由实现者编写，不是独立验收。新写入的时间点都来自命令输出；无法确定的时间记为 UNKNOWN。文件修改时间不能证明用户批准的时间。

## 1. 被取代的说法

| 原说法 | 出处 | 实际情况 |
|---|---|---|
| "确定性算法已启用" | `current_state.md`（资源限制一节） | 冻结 runner 从未调用 `configure_cpu`，确定性设置没有被应用。r1 的子进程在构建模型前调用它，并记录实际观察到的设置（`cpu.applied_as_requested`）。 |
| "CPU 线程固定为 24" | 同上 | 冻结 runner 没有设置线程数，用的是 torch 默认值。本机实测（2026-10-12，新进程）默认 intra 24、inter-op 24，`deterministic_algorithms` 默认为 False。线程数恰好相同，但并没有被固定；确定性默认是关闭的。r1 显式设置为 24/24 并启用确定性，然后记录观察值。 |
| "冻结时间 2026-10-12T09:45:00+08:00" | `protocol.md`、`manifest.json`、`uformer_architecture_map.md` | 文件修改时间：`manifest.json` 12:41:46，`protocol.md` 12:46:17，`uformer_architecture_map.md` 12:47:29。09:45 与第三方副本的修改时间（09:45:22）吻合，可能被误抄。真正的冻结时间点：UNKNOWN（只能确定在 12:41:46 之后）。 |
| `week_plan.md` 已交付 | PREPARE 交付清单 | 当时没有交付。现在补交：`week_plan.md`。 |
| vendored 块"与上游等价" | `uformer_architecture_map.md`、`current_state.md` | 当时没有做任何数值比较。r1 已做比较（§3），结论限于所比较的部件和配置。 |
| 授权时间 "2026-10-12T13:15:00+08:00" | `incident_evidence_20261012/DISCREPANCY.md` | 编造的时间（`DISCREPANCY_RESOLUTION.md` 已作废）。用户批准消息没有时间戳，实际时间 UNKNOWN。 |
| "检查 1、2 是用当前（已编辑）文件通过的" | 第一次执行前报告 | 不成立。两次检查早于那两次文档编辑，用的是获批字节（见 `DISCREPANCY_RESOLUTION.md` §4）。 |
| "runner 执行单臂与累计时限、启动计数" | PREPARE 报告 | 没有执行（`prelaunch_check.md` §5，R1–R10）。r1 已实现，见 §2。 |
| `dd_in=1` 证明没有全局捷径 | 第一次执行前报告 | 论据不适用，隔离实现里没有 `dd_in`。无捷径的结论依据代码路径（`prelaunch_check.md` §3）。 |

## 2. R1–R10 的处理

| ID | r1 中的实现 | 验证（`evidence/final3_20261012/`） |
|---|---|---|
| R1 单臂时限 | 从冻结 config 读取 `max_seconds`。软停止时间点 = 截止时间 − `eval_reserve_s`；到截止时间强制结束子进程。截止时间 = min(臂开始 + 臂时限, t0 + 7,200) − `end_margin_s` | lifecycle：软停止（`per_arm_soft_stop`）；在一次更新内挂起、在最终评估中挂起，都在截止时间被强制结束 |
| R2 累计时限 | 只有一个时钟：`time.monotonic()` 从 supervisor 被导入时开始计时，涵盖包校验、ledger 读取、各臂和 end 记录。两臂共用 | lifecycle：`aggregate_budget_is_shared`；`aggregate_clock_starts_before_the_invocation_record` |
| R3 启动计数 | 两个 ledger 根目录（运行根目录与仓库内镜像），记录用 O_EXCL 创建并 fsync，只写一次；最多 1 次 invocation、2 次启动；重复启动、超额启动、前置条件未满足都拒绝；两个根目录内容不一致或有残缺记录也拒绝 | control：13 passed；lifecycle：`second_invocation_is_refused` |
| R4 中断证据 | 子进程：KeyboardInterrupt/停止文件 → INCOMPLETE 记录。supervisor：Ctrl+C/Ctrl+Break → 通知子进程停止，宽限期到后强制结束，为每个已启动的臂写 end 记录。supervisor 被强制结束时不写任何东西，但留下的 start 记录会阻止任何再次进入 | lifecycle：`supervisor_interrupt_writes_end_records`；`killed_supervisor_leaves_an_unfinished_start`（真实进程被强制结束，残留的子进程自行退出，拒绝时不写任何字节） |
| R5 线程/确定性 | 子进程在导入 torch 之后、构建模型之前调用冻结的 `configure_cpu`，记录观察值，与请求不符则判 EXECUTION_ERROR | lifecycle：`completed_arm_records_required_fields_and_cpu` |
| R6 梯度与参数变化 | 曲线点记录四个分组（encoder/decoder × sampling/core）的梯度 L2、得到梯度的参数数、非有限值数；记录本次更新与相对初始的参数变化；还记录可训练参数及优化器覆盖情况 | lifecycle 与 measure |
| R7 第 1 步 | 曲线点取第 1 次、每 100 次（冻结协议）以及最后一次更新 | lifecycle（fixture 每 2 次取一点） |
| R8 第 0 步不改变状态 | 第 0 步在深拷贝上于 `fork_rng` 内测量（eval 和 train 两种模式），测量前后对训练模型状态加 RNG 做指纹比较，不一致则拒绝 | measure：参照组表明对训练模型本身做 train 模式前向会改变 BN，隔离后状态与 RNG 不变，训练轨迹逐位一致 |
| R9 峰值 RSS | supervisor 按 `poll_interval_s` 采样整个子进程树并执行上限；子进程自采样（标为"采样最大值"），另报告 Windows `peak_wset` | lifecycle：`rss_guard_kills_the_child`；measure：采样器 |
| R10 证据上限 | 记录运行根目录字节数，并与 512 MiB 比较（`evidence_within_cap`）。这是事后检查，不是写入前拒绝；预计只有数个 JSON 文件 | lifecycle 中 summary 包含该字段 |

另外实现的一项：子进程用 SHA-256 核对实际导入的每个模块（冻结的 `uid_common`/`uformer_ae`/`uformer_blocks` 以及 r1 模块），并禁止导入 `cg_fedllm`（lifecycle：`child_refuses_a_module_with_an_unapproved_hash`）。

## 3. 上游对应性（只针对所比较的部件）

- 方法：上游文件 `third_party/uformer_65fc970/model.py` 不作修改，按原字节执行。einops 0.8.1 从 PREPARE 时下载的 wheel 只读导入（已核对 SHA-256）。timm 用三个替身：`DropPath`（被比较的路径上不会构造）、`to_2tuple`、`trunc_normal_`（只影响初始化，会被权重映射覆盖）。权重按名称映射，状态键集合必须完全相同。在 float64 下随机生成权重和输入，比较前向、输入梯度和每个参数的梯度。容差在运行前声明：max_abs ≤ 1e-12，rel_l2 ≤ 1e-12。
- 结果：12 个用例全部完全一致（max_abs = 0）。用例包括 window_partition/reverse（往返精确）、移位窗口 mask、相对位置索引、q/k/v 投影、注意力缩放、W-MSA、带 mask 的 SW-MSA、LeFF、shift=0 与 shift=4 的 LeWin 块，以及 depth=2 的 BasicUformerLayer。配置为 dim 64、2 heads、窗口 8、网格 32。
- 负对照：权重改动 1e-9 会被检出（max_abs 6.4e-10，超过容差）；不映射权重时差异很大（max_abs 3.0）。比较并非空检验。
- 不在声明范围内：上游 `Uformer` 整体网络（多尺度、skip、`dd_in` 捷径），它与 CG-FedLLM 适配版的结构本来就不同；`uformer_ae.py` 的采样与 token 布局是项目自己的适配。采样层另行与仓库 ResNet-3 比较（在 train 和 eval 两种模式下，前向和梯度完全一致）。dropout/drop_path 为 0 的设置来自配置，没有改用上游默认的 0.1。

## 4. 仍未核实或未知

- 修订后的 r1 字节只做过 fixture 测试；登记的 3,000 次更新的实际耗时、峰值 RSS 仍未测量。成本探针只给出前向 + 反向的时间（不含 Adam 步、单进程、工程 seed）。
- RSS 采样可能漏掉短于采样间隔（0.25 s）的峰值。
- 软停止只在两次更新之间检查；单次更新内部的挂起由强制结束处理，这时子进程不写记录（由 supervisor 根据 start 记录收尾）。
- 测试只在本机 Windows 上运行；本修订没有在其他平台上验证。
