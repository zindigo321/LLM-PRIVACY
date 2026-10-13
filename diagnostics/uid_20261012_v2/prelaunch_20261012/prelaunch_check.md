# UID-20261012-v2 执行前一致性检查（2026-10-12）

**结论：BLOCKED，在任何科学训练之前停止。** 包身份已恢复并核实。冻结的 `uid_runner.py` 没有实现批准要求的多项执行条件（§5）。
没有调用任何优化器更新；训练启动计数为 0；`output/` 未创建。本文件是实现者的执行前检查，不是独立验收。

## 1. 执行身份（完整 SHA-256，恢复后逐项核实，冻结清单 7/7 一致）

| 文件 | SHA-256 | 字节 |
|---|---|---:|
| `config.json`（原始字节） | `d7ae76052d50776705f386655b0b501d403104d7d4ca118865a2b6ff3b71b0bd` | 2,633 |
| `uformer_blocks.py` | `3333f1862288e720d83aadbe8f171898f5f6943d2c0f58c9bd0132aba60b0747` | 10,213 |
| `uformer_ae.py` | `a3ad6e3f2b6e6f009158276f5534e2c68bf42781df18a1c1d9fcd0f5867c8606` | 8,544 |
| `uid_common.py` | `f1f55dc90faddb0e4ac10be5fca0cc756592cf218dedb4fb90d1c4e82dc41230` | 9,691 |
| `uid_runner.py` | `a4ae1ed06948cd33996f06ccddf02bb0ef43c9ec1eb2dd0dfc9302f0026d9e6c` | 7,172 |
| `third_party/uformer_65fc970/model.py` | `03a2c6e50b4342e03a457dd10537c14d0f832ce186da49c781fbd9d7027904a9` | 54,011 |
| `third_party/uformer_65fc970/LICENSE` | `5bce7d17fb155d8c96d7b74901b710b43939a4851f60b0adb28e6adbd606f3a5` | 1,070 |

- 配置规范哈希（canonical JSON）：`bb0111fab6310ab5430d931b781fc3c97cdb928da773d0cf8e71ea532fb4e5b3`，与原始字节哈希分开记录。
- 解释器 `D:\conda_envs\cgfedllm\python.exe`，Python 3.11.16，torch 2.14.0+cu130，`torch.cuda.is_available()` = False。
- 进程环境：`PYTHONPATH` 指向诊断目录，`CUDA_VISIBLE_DEVICES=-1`，`HF_HUB_OFFLINE=1`，`PYTHONDONTWRITEBYTECODE=1`。
  `uid_common`、`uformer_ae`、`uformer_blocks`、`uid_runner` 都从诊断目录导入；`cg_fedllm` 未被加载。
- 清单覆盖：启动命令的项目内依赖只有 `uid_runner.py` → `uid_common.py` → `uformer_ae.py` → `uformer_blocks.py` 和
  `config.json`，均在清单内。第三方依赖为 torch、psutil、numpy（`tensor_sha256` 用到 `.numpy()`）。`checks/` 不被启动命令使用。
- 15:32:53 没有 Python 进程；诊断目录的修改时间只对应本会话的写入。VS Code 未保存的缓冲区无法访问，只核实了磁盘字节。
- 仓库：HEAD `0ad2795c7d03f6d0ea76992ee83864a00efd6b03`；`git status` 只比会话开始时多出 `?? diagnostics/`；
  抽查的 5 个 SSBD 相关文件哈希与 10-11 审查记录一致。

## 2. PREPARE 是否消耗了训练启动：否

- `checks/check_1_geometry_gradient.py`、`checks/check_2_manifest_ucheck.py` 中没有 optimizer、`.step(`、`zero_grad`
  或任何保存操作。每个检查在独立进程中新建模型，只做一次前向和一次 `backward()`，然后退出，不保留任何状态。
- 检查 1 在 train 模式下前向，更新了那个临时对象的 BN running 统计；该对象随进程结束，没有被持久化，也没有被 runner 使用
  （runner 每次都按种子新建模型）。
- 磁盘上没有 `output/`、start 或 ledger 类文件。
- 恢复后已重跑两个检查，输出保存在本目录（stdout/stderr，退出码均为 0，stderr 为空）。

## 3. 重建路径：没有全局输入捷径（依据代码，而不是 `dd_in`）

- `uid_common.reconstruct` → `model.decode(model.encode(x))`（`uid_common.py:232-235`）。
- `encode` → `encoder(x)`（`uformer_ae.py:184`）→ `_run_core(core, sampling(x))`（141）；`decode` → `decoder(z)`（187）
  → `sampling(_run_core(core, z))`（167）；`forward` 返回 `(decode(z), z)`（191）。
- 输出中没有任何项加上 `x`；解码器只接收 `z`；没有 skip、目标查表或依赖目标的路径。LeWin 块内的残差只作用于 latent token，
  属于内部结构残差。
- 本实验是**无空间下采样的拟合对照**，latent/input 元素比为 64:1 的扩张。它不是压缩结果，也不保证可逆。
- 更正：上一份报告用 `dd_in=1` 作论据并不适用，隔离实现里根本没有 `dd_in`；它引用的 `uformer_ae.py:241-244`、
  `162-166` 行号也是错的（文件只有 207 行）。

## 4. RSS 上限

启动前测得可用物理内存 10,198,917,120 B，其 50% 为 5,099,458,560 B，min(4 GiB, 50%) = 4,294,967,296 B，等于配置值，一致。
runner 在每次更新前检查当前 RSS，但不记录峰值。

## 5. 阻塞项：冻结 runner 与批准条件不符

| ID | 批准/协议要求 | 冻结 runner 的实际行为 | 证据 |
|---|---|---|---|
| R1 | U-CHECK ≤ 300 s，U-FIT ≤ 1,800 s | 从不读取 `max_seconds`；唯一的时间检查是 `global_limits.max_scientific_seconds` | `uid_runner.py:44-47`；全文无 `max_seconds` |
| R2 | 总科学时间 ≤ 7,200 s，含准备与评估 | 每个进程从 `Runner` 构造时单独计时，两臂不累计；构造前的导入和读配置不计时；最终评估阶段不检查时限 | `uid_runner.py:36`、`:44-47`、`:76-92` |
| R3 | 总共最多 2 次启动；训练前持久化 start 记录；中断也消耗启动 | 没有 start 记录；唯一的写入是训练结束后的结果 JSON。重跑会重新训练，直到写入时才被拒绝 | `uid_runner.py:130-131` |
| R4 | 停止时保留已有证据 | `ResourceStop` 和 `Exception` 路径会写 JSON；`KeyboardInterrupt` 或进程被终止时不写任何文件 | `uid_runner.py:117-131` |
| R5 | 执行前固定线程数与确定性设置 | `configure_cpu` 有定义但从未调用；`cpu_config` 从未读取；确定性算法未启用 | `uid_common.py:126`；runner 中无调用 |
| R6 | U-FIT 记录 encoder/decoder 梯度与参数变化证据 | 没有记录，只有初始和最终的 state 哈希 | `uid_runner.py:104`、`:112` |
| R7 | 记录第 0、1 步、每 100 步和最终状态（`protocol.md`） | 不记录第 1 步 | `uid_runner.py:73` |
| R8 | 诊断不得额外前向或改动归一化缓冲区 | 第 0 步测量在 train 模式下对待训练模型做 `no_grad` 前向，训练开始前更新一次 BN running 统计；第 0 步只有 train 模式数值 | `uid_runner.py:107-109` |
| R9 | 记录观测到的峰值 RSS | 不记录 | — |
| R10 | 新增保留证据 ≤ 512 MiB | 代码中不检查。预计只有两个小 JSON，可在事后核实；重要性低 | — |

R8 的数值影响预计很小：BN momentum 为 0.1，3,000 次更新后这次额外更新的权重约为 0.9^3000。但它违反了已登记的规则，
此处只记录，不作为结果解释的依据。

非阻塞事项：
- 曲线中第 k 步的数值来自第 k 次更新**之前**的 train 模式前向（即 k−1 次更新之后），字段名没有标明。
- U-CHECK → U-FIT 的门控和 `output/` 的创建由操作者完成，代码中没有实现。
- vendored 块与上游的等价性从未检验：docstring 引用的 `checks/check_upstream_equivalence.py` 不存在，einops 也未安装。状态 UNVERIFIED。
- `protocol.md` 和清单中的 "冻结于 09:45" 不准确：实际写入时间为 12:41–12:46。

## 6. 原计划命令（未执行）

```powershell
Set-Location D:\projects-cd\cg-fedllm-reproduction\diagnostics\uid_20261012_v2
$env:PYTHONPATH = "D:/projects-cd/cg-fedllm-reproduction/diagnostics/uid_20261012_v2"
$env:CUDA_VISIBLE_DEVICES = "-1"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONDONTWRITEBYTECODE = "1"
New-Item -ItemType Directory output
& "D:/conda_envs/cgfedllm/python.exe" -B uid_runner.py config.json --arm U-CHECK --root output
& "D:/conda_envs/cgfedllm/python.exe" -B uid_runner.py config.json --arm U-FIT --root output   # 仅在 U-CHECK PASS 后
```

## 7. 需要人类决定

- **A（建议）**：授权一次登记在案的 runner 修订，在看到任何结果之前补齐 R1–R10。模型、输入、种子、优化器、更新次数、门槛和判定
  规则都不变。修订后用非登记的 fixture 做不含登记模型更新的工程检查，生成新清单并回读，再对新哈希单独批准。
- **B**：按现有冻结 runner 执行，并把 R1–R10 书面接受为偏差。这样批准中明确列出的条件（单臂时限、启动计数、梯度证据、峰值 RSS）
  将无法满足，不建议。
- **C**：停止本诊断。
