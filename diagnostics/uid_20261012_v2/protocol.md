# UID-20261012-v2 协议：Uformer encoder-decoder 重建诊断

**协议标识**：`UID-20261012-v2`  
**冻结时间**：2026-10-12T09:45:00+08:00  
**状态**：PREPARE 完成，等待一次执行批准

## 研究目标

确定 CG-FedLLM 论文 Appendix B.4 描述的 Uformer autoencoder 在明确指定的配置和预算下，能否学习重建其输入。

目标计算：`reconstructed_x = Decoder(Encoder(x))`。

**本次诊断仅用于基础拟合验证**，不构成泛化能力、有效压缩或联邦训练成功的证据。

## 实验设计

### 两臂控制

1. **U-CHECK**（训练管道控制）  
   - 随机初始化的 1×1 Conv2d，单输入输出通道，有 bias  
   - 共享逐点映射 `y = a*x + b`  
   - Adam lr=1e-2，最多 500 次更新，300 秒  
   - 验证训练和度量管道；不初始化为恒等解

2. **U-FIT**（实际 Uformer 控制）  
   - 仅在 U-CHECK PASS 后启动  
   - 使用已记录的 Uformer 诊断变体和相同输入  
   - 随机初始化  
   - Adam lr=2e-4，最多 3,000 次更新，1,800 秒  
   - 批大小 1，重建 MSE

### 共享输入

- CPU float32
- 独立数据和模型随机生成器
- 数据种子 20261012，模型初始化种子 1
- `x = 1.6 * rand([1, 1, 32, 32]) - 0.8`（范围 [-0.8, 0.8]）
- 保存输入哈希、形状、dtype、元素数、范围、均值、能量

### 全局限制

- 仅 CPU
- 科学墙钟时间最多 7,200 秒（包括设置、输入准备、训练、评估和清理）
- 总训练启动最多 2 次
- 每次启动前持久化启动记录；中断消耗启动次数
- 无自动重试、训练恢复、额外种子或参数搜索
- RSS 限制：min(4 GiB, 启动时可用物理 RAM 的 50%)
- 新保留证据最多 512 MiB
- 执行前固定 CPU 线程和确定性设置
- 使用最终规定的检查点，而非观察到的最佳检查点
- 在限制内为最终证据和评估预留时间

### 度量与配对模式评估

验证度量实现的解析案例：
- 预测等于目标：RSE=0
- 零预测：非零目标 RSE=1
- 预测等于 a*x：RSE=(a-1)²
- 零预测的余弦未定义

对于实际 float32 归一化目标，在约简前转换为 float64：

```
Z_n = sum(x_n²) / N
MSE_n = sum((reconstructed_x_n - x_n)²) / N
q_n = MSE_n / Z_n
```

同时报告原始空间：

```
RSE = sum((reconstructed_x - x)²) / sum(x²)
cosine = dot(reconstructed_x, x) / (norm(reconstructed_x) * norm(x))
norm_ratio = norm(reconstructed_x) / norm(x)
```

保存 SSE、目标能量、预测能量、点积和 N 以便独立重新计算。

记录步骤 0、步骤 1、每 100 次更新和最终状态。区分更新前训练损失和更新后评估。

最终优化器更新后：
- 创建具有相同参数、缓冲区和初始 RNG 状态的独立模型副本
- 在相同输入和批次上评估 train 和 eval 模式
- 分别报告两种模式

如果 BatchNorm 存在，记住 `no_grad` 不阻止 train 模式缓冲区更新。不要使用相同的变异对象评估 train 然后 eval。不要在训练后重新校准 BatchNorm 以获得通过结果。

如果必须保留非零随机正则化，在执行前登记 RNG 和重复评估策略。单次随机差异不能确立归一化失败。

### 严格拟合标准

- 有限重建和必需度量
- RSE ≤ 0.01
- 余弦 ≥ 0.99

使用 eval 模式作为主要判定并分别报告 train 模式。未定义的必需度量不能通过。

### 判定规则

- `NOT_RUN`：未执行
- `BLOCKED`：前置条件失败（例如 U-CHECK 失败）
- `COMPLETED_PASS`：eval 模式满足严格标准
- `COMPLETED_FIT_FAIL`：完成但未满足标准
- `RESOURCE_STOP`：资源限制停止
- `EXECUTION_ERROR`：执行错误
- `INCOMPLETE`：其他未完成原因

资源停止的运行不是完成的拟合失败或完成的协议成功。

## Uformer 架构

### 已确认（来自论文 arXiv:2405.13746v3 App. B.4 和 Table 7）

- 保留 ResNet AE 的上/下采样层
- 核心块替换为 BasicUformer，每个包含两个 LeWinTransformerBlock
- 编码器配备 DownSampling 层和三个 BasicUformer 块
- 解码器包含三个 BasicUformer 块和 UpSampling 层
- 与 Sect. B.3 中相同的预训练和微调

### 实现选择（未由论文指定）

本诊断使用的配置（记录在 `config.json` 和 `uformer_architecture_map.md` 中）：

- **输入**：32×32 单通道（从架构推导的合法大小）
- **采样层**：ResNet-3 重建（BatchNorm/ReLU、反射填充、Tanh 头）
  - 6 个 Conv3x3 下采样阶段，通道 1→2→4→8→16→32→64
  - 本诊断：**全步长 1**（弱空间压缩瓶颈）
  - 潜在 [64, 32, 32]，元素比 64:1 扩展
- **核心块**：3 层 BasicUformerLayer，每层深度 2（每侧 6 个 LeWin 块）
- **注意力**：num_heads=2，win_size=8（作者默认），qkv_bias=true
- **FFN**：LeFF（局部增强前馈），mlp_ratio=4.0，hidden_dim=4*dim
- **位置**：仅窗口注意力中的学习相对位置偏置；无绝对位置嵌入
- **shift_flag**：true（SW-MSA 与 W-MSA 交替）
- **跳跃连接**：无全局输入输出快捷方式，无编码器-解码器特征跳跃
- **正则化**：dropout=0、attn_drop=0、drop_path=0（确定性，无随机正则化）
- **初始化**：Linear/LayerNorm 按上游 `Uformer._init_weights`（截断正态 std=0.02）
- **令牌布局**：行优先 [B, h*w, C]

### 诊断修改

相对于选定的参考架构：

1. **全步长 1**：6 个下采样阶段全部使用步长 1，潜在网格与输入网格相同  
   - 弱空间压缩瓶颈（64:1 元素扩展）  
   - 这是一个**诊断干预**，不是主管确认的要求或可逆性保证
2. **无随机正则化**：dropout、attention_drop、drop_path 全部为 0  
   - 为确定性拟合控制而提出  
   - 在执行前禁用，而不是在观察不利结果后禁用
3. **32×32 输入**：从架构合法性推导（窗口分区、采样因子、填充）
4. **BatchNorm**：保留在 ResNet 采样层中；train/eval 差异是预期的

完整的架构映射和上游等效性检查在 `uformer_architecture_map.md` 中。

## 解释限制

使用这些判定规则：

- **U-CHECK 失败**：调查训练/数据/度量管道；不启动 U-FIT 或得出 Uformer 缺乏容量的结论
- **U-CHECK 通过且 U-FIT 通过**：仅为测试的 Uformer 变体、输入、种子和预算建立拟合
- **Train 通过而 eval 失败**：建立模式相关差异；在分配原因前调查实际模块和随机性
- **两种模式都失败**：eval 模式本身不能解释失败。一次不成功的优化运行不能证明数学表达能力不足
- **Uformer 缺失或其身份无法确立**：明确报告实现差距
- **资源或证据限制阻止完成**：报告限制而不制造科学判定

历史 ResNet 结果不能确立 Uformer 成功或失败。归档的 R3 训练 MSE 接近其零预测基线；它不能证明训练拟合良好而只有评估失败。

## 文件清单

冻结文件（`manifest.json` 中的 SHA-256 验证）：

- `config.json`（2,633 字节）— 冻结配置
- `uformer_blocks.py`（10,213 字节）— LeWinTransformerBlock / BasicUformerLayer
- `uformer_ae.py`（8,544 字节）— UformerAutoEncoder（CG-FedLLM App. B.4 适配）
- `uid_common.py`（9,691 字节）— 辅助函数：哈希、输入、度量、标准、模型构建
- `uid_runner.py`（7,172 字节）— 具有资源限制的单次调用训练运行器
- `third_party/uformer_65fc970/model.py`（54,011 字节）— 固定上游（MIT）
- `third_party/uformer_65fc970/LICENSE`（1,070 字节）— MIT License (c) 2022 Zhendong Wang

工程检查（执行前运行）：

- `checks/check_1_geometry_gradient.py` — Uformer 几何和梯度流
- `checks/check_2_manifest_ucheck.py` — 清单验证和 U-CHECK 健全性

## 依赖

- torch 2.14.0+cu130（CPU 执行，CUDA_VISIBLE_DEVICES=-1）
- numpy 2.4.6
- psutil 7.2.2
- einops **未安装**；`rearrange`/`repeat` 已内联重新实现

## 批准检查点

在请求科学执行批准之前，完成授权准备并保存文件。

向我展示：

- 已验证的本地仓库和解释器身份
- 实际 Uformer 变体及其来源
- 输入几何和所有诊断修改
- 快捷方式/跳跃处理以及 PASS 将确立什么
- 确切命令、输出路径和资源限制
- 协议、配置、脚本和依赖哈希
- 真实的工程检查结果
- 任何实质性影响执行的未解决问题

然后请求对该具体包的一次批准。不要重新讨论是否使用 Uformer 的已解决问题。

批准后，验证批准的文件未更改，并在约定的限制内执行 U-CHECK → U-FIT → 最终评估 → 报告。

在每个有意义的阶段后保存一个简短的检查点。在中断后，在继续之前检查实际文件和启动记录。永远不要假设计划的命令已运行，永远不要自动重启已消耗的科学运行。

---

**协议冻结**：本文档的内容在 2026-10-12T09:45:00+08:00 冻结。任何修改都会使批准失效并需要人工审查。
