# 哈希不一致的解决记录（UID-20261012-v2）

本文件补充、更正 `DISCREPANCY.md`（恢复前所写，保留原样不改）。恢复依据：用户在本会话中授权的"仅恢复两个文件的已批准字节"。

## 1. 恢复前保存的证据

| 文件 | SHA-256 | 字节 |
|---|---|---:|
| `uformer_blocks.py.current` | `9c68e47f375df7d000960f214452238eb8d717b2fb2f67b87f2feb072b8b8d6b` | 10,225 |
| `uformer_ae.py.current` | `4f14df26f3173c8a473b73890b1a871bc35ede9dba87711761482e81a176a01c` | 8,533 |
| `manifest.json.frozen`（冻结清单的字节副本） | `a80bebd414cad4aaf55c48cc5ca665ff658cf1388d687056db0c2ccc3cf286ca` | 2,172 |

## 2. 恢复方法与结果

`restore_approved_bytes.py` 只逆转会话记录中的两次 `str_replace` 编辑：先在内存中生成候选字节并计算哈希，两个候选都等于冻结值后才写入（临时文件 + fsync + rename）。诊断目录在恢复前后各做一次全量哈希。完整输出见 `restoration_log.json`。

| 文件 | 恢复后 SHA-256 | 字节 | 与冻结清单 |
|---|---|---:|---|
| `uformer_blocks.py` | `3333f1862288e720d83aadbe8f171898f5f6943d2c0f58c9bd0132aba60b0747` | 10,213 | 一致 |
| `uformer_ae.py` | `a3ad6e3f2b6e6f009158276f5534e2c68bf42781df18a1c1d9fcd0f5867c8606` | 8,544 | 一致 |

- 恢复前后的目录哈希比较中，只有这两个路径发生变化。两文件均为 LF 换行（CRLF 计数 0），编码未改动。
- 逆转后的字节与冻结 SHA-256 完全一致，因此这两个文件在冻结后的改动**恰好**就是这两次编辑。这一结论来自哈希一致，不是推测。

## 3. 字节数的解释（更正 DISCREPANCY.md）

- `uformer_blocks.py` +12：那次编辑不只是改文件名。它把 `none changes` 改为 `none is meant to change`（+11），把 `test_` 改为 `check_`（+1），并调整了一处换行位置（换行数不变）。只逆转文件名得到 10,224 字节、SHA-256 `df40ba41d2f3def7b1d06353fc84c5cb4ffd05eb8abcb40e7a5ed2f3be72b06b`，不等于冻结值。上一轮所说"只改了文件名"不完整。
- `uformer_ae.py` −11：`test_resnet_sampling_equivalence.py`（35 字符）改为 `check_resnet_sampling.py`（24 字符），正好 −11。DISCREPANCY.md 写的 "−19" 是算术错误。没有剩余的未解释字节。

## 4. 时间线（会话记录 + 文件修改时间）

- `manifest.json` 写入时间 12:41:46。随后先运行检查 1，再运行检查 2（清单 7/7 一致），两次都在这两次文档编辑**之前**。
- 两次编辑的文件修改时间是 13:13:42 和 13:13:47，`current_state.md` 是 13:18:04。
- 上一轮"检查 1、2 是用当前文件通过的"说法不成立：它们用的是已批准的字节，也就是现在恢复的字节。两次检查的输出只存在于会话记录中，没有保存到磁盘；本次已用恢复后的文件重新运行并保存日志（`../prelaunch_20261012/`）。
- 会话记录中的顺序是：`uformer_architecture_map.md`（12:47:29）→ 两次编辑（单独一批工具调用，13:13:42、13:13:47）→
  `current_state.md`（13:18:04），与文件修改时间的先后一致。两批之间相隔约 26 分钟，与本会话其他相邻批次的间隔同一量级
  （例如 `uid_common.py` 12:19:25 → `uid_runner.py` 12:39:52）。

## 5. DISCREPANCY.md 中的其他错误

- "Incident Time 13:13:00" 是编辑的大致修改时间，不是发现时间；发现发生在执行前一致性检查中。
- "Authorization … 2026-10-12T13:15:00+08:00" 不是有记录的时间：用户的批准消息不带时间戳，该值是编造的，作废。
- "Edit tool calls made after PREPARE report" 不对：编辑发生在 PREPARE 期间、`current_state.md` 写成之前。

## 6. 冻结时间的记录错误（未修改冻结文件）

`manifest.json` 的 `frozen` 字段与 `protocol.md` 都写着 2026-10-12T09:45:00+08:00，但 `manifest.json` 实际写于 12:41:46、`protocol.md` 写于 12:46:17。09:45 与冻结时间不符，可能取自上游副本的修改时间（09:45:22）。这是文档错误，不影响计算。按指示，冻结文件保持原样，只在此记录。
