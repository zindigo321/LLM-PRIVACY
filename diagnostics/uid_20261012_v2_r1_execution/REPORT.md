# UID-20261012-v2（r1 执行机制）结果报告

执行于 2026-10-12，一次 invocation，19:53:29–20:01:59（+08:00）。本报告由实现者编写，不是独立验收。
批准的包：`manifest_r1.json`，SHA-256 `9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044`（启动前 29/29 核实一致）。

## 1. 问题

在明确的配置和预算下，按 CG-FedLLM App. B.4 适配的 Uformer encoder-decoder（x → Encoder → z → Decoder → x̂），能否学会重建一个固定的合成输入。

## 2. 实际测试的对象

- **架构**：保留 ResNet-3 的采样层（BN/ReLU、反射填充、Tanh 输出头），编码器和解码器两侧各有 3 个 BasicUformer，每个含 2 个 LeWinTransformerBlock，W-MSA 与 SW-MSA 交替。这些块 vendored 自 ZhendongWang6/Uformer@65fc970a（MIT 许可），所比较的部件在 float64 下与上游完全一致。配置为 dim 64、2 heads、窗口 8，前馈层用 LeFF（mlp_ratio 4），带相对位置偏置。参数共 685,507 个，全部可训练且都在优化器中。
- **重建路径**：x̂ = Decoder(Encoder(x))。没有全局输入捷径，也没有 encoder→decoder 的 skip。子进程按哈希核对了所导入的模块。
- **诊断修改**（执行前已登记）：6 级步长全部为 1，latent 为 [64,32,32]，元素数是输入的 64 倍，所以这是扩张，不是压缩；dropout、注意力 dropout 和 drop-path 全为 0；输入为 32×32 单通道。heads、窗口等论文未说明的值是实现选择。
- **输入与设置**：data seed 20261012，`x = 1.6·rand([1,1,32,32]) − 0.8`，输入 SHA-256 为 `fc5e1283d1a44cb3bedaf89cff96818b1db0116e036e7cfad75c4a121138d722`。模型 seed 1；Adam，lr 2e-4（U-CHECK 用 1e-2）；CPU float32。线程 24/24、确定性开启，均为子进程实测值。

## 3. 结果

判定以 eval 模式为准，需要输出与度量有限、RSE ≤ 0.01、cosine ≥ 0.99。

| | U-CHECK（1×1 卷积管道对照） | U-FIT（Uformer） |
|---|---|---|
| 启动次数 / 完成的更新 | 1 / 500（共 500） | 1 / 3,000（共 3,000） |
| 臂耗时（上限） | 4.0 s（300 s） | 506.3 s（1,800 s） |
| 采样最大 RSS（每 0.25 s 采样子进程树） | 638,971,904 B | 822,517,760 B |
| Windows 峰值工作集（子进程退出时读取） | 637,665,280 B | 824,033,280 B |
| 停止原因 / 状态 | 无 / **COMPLETED_PASS** | 无 / **COMPLETED_FIT_FAIL** |
| 第 0 步 eval：RSE / cosine | 1.128 / 0.470 | 1.006 / −0.017 |
| 第 0 步 train：RSE / cosine | 1.128 / 0.470 | 1.878 / −0.026 |
| 最终 eval：RSE / cosine / 范数比 | 7.2e-20 / 1.000 / 1.000 | **0.2727 / 0.8540** / 0.900 |
| 最终 train：RSE / cosine | 7.2e-20 / 1.000 | 0.2721 / 0.8545 |
| train 与 eval 输出之差的 SSE | 0 | 0.109（eval 的 SSE 为 62.0） |

- 两个基线（同一输入）：零预测 RSE 1.000，整张量共享一个标量均值时 RSE 0.9997。
- 整个 invocation 用时 510.5 s（上限 7,200 s），证据共 83,280 B（上限 512 MiB）。两个启动配额都已用完（2/2）。
- 第 0 步在隔离副本上测量。最终评估用两份独立副本，它们加载同一个最终状态，并从相同的 RNG 状态开始。

**U-FIT 曲线**（每个记录点：更新前的 train 损失 / RSE）：

| 更新次数 | 损失 | RSE |
|---|---|---|
| 1 | 0.417 | 1.878 |
| 100 | 0.117 | 0.529 |
| 1,000 | 0.0779 | 0.351 |
| 2,000 | 0.0707 | 0.319 |
| 3,000 | 0.0604 | 0.272 |

- 2,200 到 2,400 次之间下降较快（0.0700 → 0.0629），之后变慢。末段并不单调：2,700 到 2,800 小幅上升，到 3,000 仍在下降。
- 记录点中最低的损失出现在最后一个点（3,000）。按协议，判定使用最终状态，不挑最佳点。

**梯度与参数（U-FIT）**：分为四组（encoder/decoder × 采样/核心）。

| 分组 | 梯度 L2（首点 → 末点） | 参数相对初始的变化 L2（初值 L2） |
|---|---|---|
| encoder 采样 | 1.82 → 0.19 | 0.52（13.0） |
| encoder 核心 | 0.087 → 0.0085 | 2.02（38.0） |
| decoder 核心 | 0.086 → 0.0078 | 1.94（38.1） |
| decoder 采样 | 1.85 → 0.13 | 0.91（10.3） |

- 每个记录点上四组梯度都非零，没有非有限值。
- 采样层的元素全部有变化。两个核心各有 317,964 个元素，其中 317,963 个有变化；没变的那一个是哪个元素，没有查明。

**复核**：
- 用一个单独的脚本，只根据记录下来的 SSE、能量、点积和 N 重算 RSE、cosine、范数比和 q_n，与记录值的相对差为 0。按冻结门槛重新判定，结论与 ledger 一致。
- 恒等式 SSE = P + T − 2D 对 U-FIT 成立（相对误差 ≤ 8e-16）。对 U-CHECK 无法检验：其 SSE 约 1.6e-17，低于 float64 在 454.66 附近的分辨率（约 5.7e-14）。

## 4. 证据支持的结论

- 训练、数据和度量管道工作正常：U-CHECK 从非恒等的初始化（RSE 1.128）拟合到 RSE 7e-20。这只是管道对照，不是 Uformer 的结果。
- 在这个变体、输入、seed、lr 2e-4 和 3,000 次更新的条件下，Uformer encoder-decoder **没有达到**拟合门槛。不过 encoder 和 decoder 都参与训练并且都有更新，网络学到了部分重建：RSE 从约 1.0（eval）/ 1.88（train）降到 0.27，远低于两个基线；cosine 为 0.85。
- 初始化本身不满足门槛，所以不属于 INIT_PASS 情形。
- 最终的 train 模式与 eval 模式结果几乎相同，两者都远未达标。所以 BN 在 train 与 eval 之间的统计量切换不能解释这次失败。这不是对 BN 作用的一般结论。

## 5. 不能支持或仍未解决的问题

- 一次没有成功的优化运行，不证明这个网络在数学上无法表示恒等映射。更多的更新次数、别的学习率或结构改动能否达标，都没有检验。
- 结构上，输入在 stem 处要经过单通道的 Conv→BN→ReLU，在输出头之前又经过一次单通道的 BN→ReLU，两处都继承自 ResNet-3 的采样层。它们是否限制了拟合，没有检验，这里不作归因。
- 这次结果与历史 ResNet-3 的结果不可比较（架构和输入都不同）。它也不涉及压缩（latent 是 64 倍扩张）、泛化、论文复现或联邦训练。
- 最终权重没有保存。按附录 §5，输出只包含状态的 SHA-256，最终状态为 `fc3a8a7e8480bd3a1ebddbbdd46d7164e1bd9c552ca3ce56f4b84fec2d300985`。要分析训练后模型的内部，需要新登记一次运行。
- RSS 是 0.25 s 间隔下的采样最大值，OS 的峰值工作集另行列出。ledger 只能证明它记录下来的执行。本协议的配额已经用完，不能在本协议下再运行。

## 6. 建议的下一步（只此一项）

单独登记一个只改更新预算的后续诊断：模型、输入、seed 和 lr 都不变，并保存最终权重。目的是区分两种情况：预算不够，还是在这个配置下已经停滞。例如 30,000 次更新，按实测约 0.168 s/次估计，大约 85 分钟 CPU 时间。这需要新的协议和新的批准，目前什么都没有启动。

## 执行与证据（路径都相对于 `diagnostics/uid_20261012_v2_r1_execution/`，另有说明的除外）

- 授权与启动前检查：`EXECUTION_AUTHORIZATION.md`、`prelaunch_check_r1.md`、`print_plan.out.txt`。
- 启动：`launch/COMMANDS_AND_EXIT_CODES.txt`（`--status` 与 `--scientific` 都以退出码 0 结束）、`launch/*.stdout.txt`、`launch/*.stderr.txt`（stderr 为空）。
- 启动方式的两点说明：
  - 用 Start-Process 运行，只是为了把 stdout 和 stderr 保存到文件，参数与 `PROPOSED_LAUNCH.md` 一致；
  - 其中带条件的 `Remove-Item Env:PYTHONPATH` 一步没有执行，因为该变量本来就没有设置。
- 运行根目录：`D:/cgfed-runs/uid_20261012_v2_r1/`，包括 `ledger/` 和 `arms/{U-CHECK,U-FIT}/` 下的 `job.json`、`curve.jsonl`、`child_record.json` 以及子进程的 stdout/stderr（为空）。ledger 镜像与主 ledger 逐字节相同。
- 分析：`analyze_run.py`、`analysis.json`、`curve_u_fit.csv`、`analyze_run.out.txt`。第一次分析因为脚本文件尚未写入而失败，那次的输出保留为 `analyze_run.attempt1_file_missing.out.txt`。
- 文件哈希：`RUN_ARTIFACTS_SHA256.txt`（运行结束后记录）和 `FINAL_ARTIFACTS_SHA256.txt`。
