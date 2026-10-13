# UID-20261012-v2 协议附录 r1（执行机制）

- 父协议：`uid_20261012_v2/protocol.md`（SHA-256 `de24e62ffc32fc72bfb79b3d32e32bda5f9d21d1b2b1432c735d6ba4f3c6dae0`），保持不变，继续有效。
- 本附录只规定**如何执行**。研究问题、输入、seed、模型、优化器、更新次数、时限、RSS 上限、启动次数上限、度量、门槛、判定规则和解释限制都以父协议和冻结的 `config.json` 为准，本附录不作任何改动。
- 编写时间：r1 工程修订期间（2026-10-12）；尚未获得执行批准。

## 1. 执行入口与身份

- 唯一的科学入口：`uid_r1_supervise.py --scientific --approved-r1-manifest-sha256 <批准的 manifest_r1 SHA-256>`。
- 启动前拒绝的情形：
  - 冻结清单、冻结 config 的原始哈希或规范哈希，或 `manifest_r1.json` 中任一文件的 SHA-256 与批准值不一致；
  - 解释器不是 `execution_r1.json` 中指定的那个；
  - 设置了 `PYTHONPATH`。
- 子进程导入的每个模块都要核对 SHA-256，并禁止导入 `cg_fedllm`。
- 冻结的 `uid_runner.py` 不再用于科学执行。

## 2. 启动计数（取代父协议中"每次启动前持久化启动记录"的未实现部分）

- ledger 有两个根目录：`D:/cgfed-runs/uid_20261012_v2_r1/ledger` 和仓库内的 `diagnostics/uid_20261012_v2_r1/scientific_ledger_mirror`。记录只写一次（O_EXCL + fsync），两边必须逐字节相同。
- 1 次 invocation，最多 2 次启动。每臂在启动子进程之前写 start 记录，结束后写 end 记录。
- 只有 start、没有 end 的臂判为 INCOMPLETE，从此拒绝任何 invocation，不会自动重试，也不会续跑。
- U-FIT 的 start 记录要求 U-CHECK 的 end 状态为 COMPLETED_PASS。

## 3. 时间

- 只有一个时钟（`time.monotonic()`），t0 = supervisor 模块导入的时刻。包校验、ledger、两臂、最终评估和记录都计入 7,200 s。
- 每臂的截止时间 = min(start 时刻 + 臂时限, t0 + 7,200) − `end_margin_s`（20 s）。软停止 = 截止时间 − `eval_reserve_s`（60 s），在两次更新之间检查。到达截止时间时强制结束子进程（最多等待 `kill_wait_s` 10 s），因此 end 记录落在两个时限之内。
- 软停止 → RESOURCE_STOP（子进程写记录）；截止时间强制结束 → RESOURCE_STOP（由 supervisor 写记录）；超过 RSS 上限 → 强制结束并判 RESOURCE_STOP。Ctrl+C → 先发停止请求，20 s 后强制结束，判 INCOMPLETE。子进程报错 → EXECUTION_ERROR。只有在子进程状态与退出码一致、且没有被强制结束时，才采用子进程报告的结果。
- 已登记 `eval_reserve_s` = 60 s。这次登记早于任何结果。如果科学运行中最终评估没有在该余量内完成，结果为 RESOURCE_STOP，不改动余量重跑。

## 4. 测量（取代父协议中未实现的部分；不改变判定）

- CPU：子进程在构建模型前调用冻结的 `configure_cpu(24, 24)`，记录观察值，与请求不符则判 EXECUTION_ERROR。
- 第 0 步：在训练模型的隔离深拷贝上测量 eval 和 train 两种模式；训练模型的状态和 RNG 指纹必须不变，否则判 EXECUTION_ERROR。
- 曲线：记录第 1 次、每 100 次和最后一次更新。字段 `pre_update_train_*` 来自第 k 次更新**之前**、以 train 模式做的那次训练前向（在 k−1 次更新之后）；`param_change_*` 是第 k 次更新之后的变化。记录四个分组的梯度与参数变化，以及采样得到的 RSS。
- 最终：使用最后一次更新后的状态，不挑选最佳点。两份独立副本加载同一个 state_dict，并从相同的 RNG 状态开始；判定以 eval 副本为准，train 副本只报告。
- RSS：supervisor 每 0.25 s 采样整个子进程树并执行上限，得到的是"采样最大值"，另报告 Windows `peak_wset`。证据字节数在事后与 512 MiB 比较。

## 5. 输出位置

`D:/cgfed-runs/uid_20261012_v2_r1/`：`ledger/`；`arms/<ARM>/{job.json, curve.jsonl, child_record.json, child.stdout.txt, child.stderr.txt}`。
另有 ledger 镜像，见上文。
