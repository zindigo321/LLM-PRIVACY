# UID-20261012-v2 当前状态：准备完成，请求执行批准

**更新时间**：2026-10-12T09:52:00+08:00  
**阶段**：PREPARE 完成  
**状态**：等待一次科学执行批准

---

## 执行摘要

本诊断实现了一个隔离的 Uformer encoder-decoder，作为 CG-FedLLM 论文 Appendix B.4 的适配。目标是在明确的配置和预算下，确定该网络能否学习重建其输入。

**本次诊断不是生产 SSBD 运行**。它是一个独立的小规模拟合控制，仅用于基础能力验证。

- 无 GPU 使用
- 无 SSBD 依赖
- 冻结的协议、配置和源代码
- 工程检查已通过
- 资源限制：7,200 秒，4 GiB RSS
- 两臂控制：U-CHECK（管道健全性）和 U-FIT（实际 Uformer）

---

## 已核实的本地状态

### 仓库和解释器

- **仓库**：`D:/projects-cd/cg-fedllm-reproduction`
- **分支**：`phase5-ssbd-r3-single`
- **HEAD**：`0ad2795c7d03f6d0ea76992ee83864a00efd6b03`（协议冻结提交）
- **main**：`d8bf57ab9e0fe49cd504e01d57e9ba59575e512e`
- **解释器**：`D:/conda_envs/cgfedllm/python.exe` (Python 3.11.16)
- **torch**：2.14.0+cu130，CUDA 不可用（CPU 执行）
- **PYTHONPATH**：已设置为仓库 `src/`（避免陈旧 worktree）

### 文件完整性

24 个 SSBD 相关文件与 2026-10-11 独立审查记录的 SHA-256 一致。冻结的 `docs/` 与 HEAD 一致。

### 资源

- **CPU**：Intel Core i9-14900HX，24 核心，32 逻辑处理器
- **物理 RAM**：32,492 MB（可用 ~11 GB）
- **无运行中的 Python 进程**
- **无 `.git/index.lock`**

### Uformer 实现

**仓库 `src/` 中无 Uformer 代码**。本诊断使用隔离的 vendored 实现：

- **上游固定**：`ZhendongWang6/Uformer@65fc970a8ffc09605faca74ed016ee93c9ad8a36`
- **许可证**：MIT License (c) 2022 Zhendong Wang
- **上游 `model.py`**：SHA-256 `03a2c6e50b4342e03a457dd10537c14d0f832ce186da49c781fbd9d7027904a9`
- **字节副本**：`third_party/uformer_65fc970/`（已验证一致）

---

## 实际 Uformer 变体

### 架构摘要

- **输入**：[1, 1, 32, 32] 单通道
- **潜在**：[1, 64, 32, 32]，元素比 64:1（扩展，非压缩）
- **编码器**：6 个 Conv3x3 **全步长 1**（1→2→4→8→16→32→64）+ 3 个 BasicUformerLayer（每层 2 个 LeWin 块）
- **解码器**：3 个 BasicUformerLayer + 6 个 ConvTranspose3x3 全步长 1（64→32→16→8→4→2→1）
- **核心**：每侧 6 个 LeWinTransformerBlock（3 层 × 2 块）
- **注意力**：W-MSA/SW-MSA，2 个头，窗口 8，相对位置偏置
- **FFN**：LeFF（Linear-GELU, depthwise 3×3, Linear-GELU），MLP ratio 4.0
- **采样层**：ResNet-3 的 BatchNorm/ReLU、反射填充、Tanh 头
- **无全局快捷方式**：`forward` 返回 `(decode(encode(x)), z)`，不是 `x + decode(...)`
- **无编码器-解码器跳跃**：解码器仅接收 `z`，无跳跃特征
- **参数**：685,507 总计（编码器 342,797 + 解码器 342,710）

### 诊断修改（相对于选定的参考架构）

1. **全步长 1**：6 个下采样阶段全部使用步长 1，潜在网格 = 输入网格
   - **弱空间压缩瓶颈**（64:1 元素扩展）
   - 这是一个**诊断干预**，用于在没有强空间压缩的情况下测试 BasicUformer 块
   - 不是主管确认的要求或可逆性保证

2. **无随机正则化**：dropout=0、attn_drop=0、drop_path=0
   - 为确定性拟合控制而提出
   - **在执行前禁用**，而不是在观察不利结果后禁用

3. **32×32 输入**：从架构合法性推导（窗口分区、采样因子、填充）

4. **BatchNorm**：保留在 ResNet 采样层中
   - **train/eval 差异是预期的**且已记录

### 实现选择（论文未指定）

- **头数**：2（论文未说明；作者默认 8 用于更大模型）
- **窗口大小**：8（作者默认）
- **Shift**：shift_flag=true，shift_size=win_size//2（SW-MSA，作者默认）
- **MLP ratio**：4.0（作者默认）
- **QKV bias**：true（作者默认）
- **FFN 类型**：LeFF（作者默认；论文未指定 FFN 结构）
- **令牌布局**：行优先 [B, h*w, C]（上游选择）
- **位置**：仅相对位置偏置（无绝对位置嵌入）
- **初始化**：trunc_normal_(std=0.02) 用于 Linear/LayerNorm（上游）

完整架构映射：[uformer_architecture_map.md](uformer_architecture_map.md:1)

---

## 快捷方式和跳跃处理

### 无全局输入-输出快捷方式

上游 `Uformer` 类的 `forward` 方法（第 1305 行）：

```python
return x + y if self.dd_in ==3 else y
```

该快捷方式仅在 `dd_in == 3`（3 通道图像恢复）时激活。

**本诊断**：`dd_in=1`（单通道），`in_chans=1`（输出通道）。快捷方式 **不** 激活。

`UformerAutoEncoder.forward` 返回 `(self.decode(z), z)`，不包含 `x +`。

**验证**：[uformer_ae.py:241-244](uformer_ae.py:241-244)

```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = self.encode(x)
    return self.decode(z), z
```

### 无编码器-解码器跳跃

解码器仅接收潜在 `z`。上游 `Uformer` 在解码器阶段使用 `torch.cat([up, conv], -1)` 连接跳跃特征。

**本诊断**：解码器没有跳跃输入。`_run_core` 在核心层前后处理 `z` 的令牌，采样层在核心前后独立运行。

**验证**：[uformer_ae.py:162-166](uformer_ae.py:162-166)

```python
def forward(self, z: torch.Tensor) -> torch.Tensor:
    return self.sampling(_run_core(self.core, z))
```

### PASS 将确立什么

- 测试的 Uformer 变体可以在测试的预算下拟合测试的输入
- **不确立**：
  - 对未见输入的泛化
  - 有用的压缩（64:1 扩展不是压缩）
  - 步长 2 采样或更大输入的成功
  - 使用此架构的成功联邦训练

### PASS 不确立的原因

1. **弱瓶颈**：全步长 1，64:1 元素扩展
2. **小输入**：32×32，单通道
3. **单次运行**：一个种子，有限预算
4. **与论文配置不同**：这是全步长 1，不是论文设置

---

## 冻结的执行包

### 协议和配置

- **协议**：[protocol.md](protocol.md:1)（冻结于 2026-10-12T09:45:00+08:00）
- **配置**：[config.json](config.json:1)
  - **原始 SHA-256**：`d7ae76052d50776705f386655b0b501d403104d7d4ca118865a2b6ff3b71b0bd`
  - **规范 SHA-256**：`bb0111fab6310ab5430d931b781fc3c97cdb928da773d0cf8e71ea532fb4e5b3`

### 清单验证

[manifest.json](manifest.json:1) 列出 7 个文件的 SHA-256，全部已验证匹配：

- `config.json`（2,633 字节）
- `uformer_blocks.py`（10,213 字节）
- `uformer_ae.py`（8,544 字节）
- `uid_common.py`（9,691 字节）
- `uid_runner.py`（7,172 字节）
- `third_party/uformer_65fc970/model.py`（54,011 字节）
- `third_party/uformer_65fc970/LICENSE`（1,070 字节）

**工程检查 1 和 2 已通过**（几何、梯度、清单、U-CHECK 健全性）。

### 源代码

- [uformer_blocks.py](uformer_blocks.py:1)：LeWinTransformerBlock, BasicUformerLayer, WindowAttention, LeFF
- [uformer_ae.py](uformer_ae.py:1)：UformerAutoEncoder（App. B.4 适配 + ResNet 采样）
- [uid_common.py](uid_common.py:1)：哈希、输入、度量、标准、模型构建
- [uid_runner.py](uid_runner.py:1)：单次调用训练运行器，具有资源限制

### 执行命令

**在批准后运行**（从 `diagnostics/uid_20261012_v2/`）：

```bash
# 环境
export PYTHONPATH="D:/projects-cd/cg-fedllm-reproduction/diagnostics/uid_20261012_v2"
export CUDA_VISIBLE_DEVICES="-1"
export HF_HUB_OFFLINE="1"
export PYTHONDONTWRITEBYTECODE="1"

# U-CHECK（管道控制）
python -B uid_runner.py config.json --arm U-CHECK --root output/

# U-FIT（仅在 U-CHECK PASS 后）
python -B uid_runner.py config.json --arm U-FIT --root output/
```

输出：`output/u_check.json`，`output/u_fit.json`

---

## 资源限制

- **全局科学时间**：7,200 秒（2 小时）
- **RSS**：4 GiB（min(4 GiB, 50% 可用 RAM)）
- **训练启动**：总计最多 2 次（U-CHECK + U-FIT）
- **保留证据**：最多 512 MiB
- **CPU 线程**：固定为 24（intra_op 和 inter_op）
- **确定性算法**：已启用

**实际 U-FIT 墙钟时间**：未知（首次运行）。限制保守设定。

---

## 度量和标准

### 基线

- **零预测**：输出 = 0
- **共享标量均值**：输出 = mean(x)（整个张量共享一个标量）

### 最终评估（配对 train/eval 模式）

1. 克隆最终模型状态（参数 + 缓冲区）
2. 构建两个独立副本
3. 一个设置为 `train()`，一个设置为 `eval()`
4. 在相同输入上评估两者（无梯度）
5. 分别报告度量

### 严格拟合标准

- **有限输出和度量**
- **RSE** ≤ 0.01
- **余弦** ≥ 0.99
- 未定义的度量 → FAIL

**主要判定**：eval 模式

---

## 未解决的问题

### 无实质性执行阻塞

- Uformer 实现已完成并通过工程检查
- 清单已验证
- 协议已冻结
- 资源限制已明确
- 输出路径已记录

### 已知限制（已记录，非阻塞）

1. **全步长 1 不是论文的预期配置**  
   → 已记录为诊断干预

2. **BatchNorm train/eval 差异是预期的**  
   → 已记录；两种模式都会报告

3. **首次运行 Uformer，无历史时间/内存数据**  
   → 限制保守设定

4. **上游等效性检查尚未编写**  
   → 工程检查 1 和 2 确认前向/后向完成；完整等效性检查是可选的进一步验证

5. **ResNet 采样等效性检查尚未编写**  
   → 采样层从 `autoencoder.py` 逐字转录；等效性检查是可选的

---

## 依赖项

已验证：

- **torch**：2.14.0+cu130（CPU 执行）
- **numpy**：2.4.6
- **psutil**：7.2.2
- **safetensors**：0.8.0
- **pytest**：9.1.1
- **ruff**：0.16.9

**einops**：未安装。`rearrange`/`repeat` 在 `uformer_blocks.py` 中内联重新实现。

---

## 与 SSBD 状态的关系

- **SSBD**：BLOCKING DEFECTS FOUND / HUMAN ACCEPTANCE PENDING / EXECUTION PAUSED（未改动）
- **UID**：独立诊断，不依赖 SSBD，不解除 SSBD 阻塞
- **无 SSBD 导入**：诊断代码不导入 `cg_fedllm.phase5.ssbd_*`
- **共享包提交**：两者都在 `phase5-ssbd-r3-single` 分支上，但诊断是自包含的

---

## 请求批准

我现在请求**一次科学执行批准**用于以下冻结包：

- **协议 ID**：UID-20261012-v2
- **冻结时间**：2026-10-12T09:45:00+08:00
- **配置规范 SHA-256**：`bb0111fab6310ab5430d931b781fc3c97cdb928da773d0cf8e71ea532fb4e5b3`
- **清单**：7 个文件，全部已验证
- **实际 Uformer 变体**：全步长 1，弱瓶颈，685,507 参数
- **输入**：[1, 1, 32, 32]，数据种子 20261012
- **两臂**：U-CHECK（管道）和 U-FIT（Uformer）
- **资源限制**：7,200 秒，4 GiB RSS，2 次启动
- **输出**：`diagnostics/uid_20261012_v2/output/{u_check,u_fit}.json`

批准后，我将：

1. 验证批准的文件未更改（重新哈希清单）
2. 创建 `output/` 目录
3. 运行 U-CHECK
4. 如果 U-CHECK PASS，运行 U-FIT
5. 保存结构化 JSON 证据
6. 编写简短的结果报告（最多 2 页）

**无自动重试，无参数搜索，无额外种子。一次执行序列。**

---

**准备完成。等待您的批准以继续执行。**
