# UCOV-20261013-v1 input selection rule (registered before any new tile value is read)

This rule is written before the snapshot payload is opened for this study, and before any value of the four tiles
below is seen. No model has been evaluated on any of them.

Seen beforehand:
- the URP-20261013-v1 record of the centre tile, which is not part of this study;
- the archived whole-snapshot summary in the P5-A record: normalized max |x_n| 0.8892.

`prepare_inputs.py` pins this file's SHA-256 and refuses to run if the file changes.

## Source (no choice)

Identical to URP-20261013-v1 (`diagnostics/urp_20261013_v1/input/SELECTION_RULE.md`):
- the R3 training snapshot of the historical P5-A single-snapshot control, `balanced_effective_delta_r8`;
- time index 7 and client 91, which is position 39 of the 80-record R3 training population;
- rebuilt by `load_states(verify=True)`, then `balanced_effective_delta(start, end, s=2.0, rank=8)[0]`, then
  `get_layout("layer_major_qkvo_AtB").forward(rep, geometry)`;
- the result is Phi `X` of shape [1, 2048, 1536], fp32. `tensor_sha256(X)` must equal the archived `phi_sha256`
  `485cbe5f3a450b43d7780fe9cd97afe5169ea81ce0bdaf21166938adfcac2546`; otherwise no tile is produced.

## Positional rule (four tiles; the quarter-grid rule)

- **Grid.** Non-overlapping 32 × 32 tiles of `X[0]` ([2048, 1536]), aligned at multiples of 32: 64 tile rows by
  48 tile columns.
- **Points.** The quarter points of both axes:
  - tile rows `64 // 4 = 16` and `3 · 64 // 4 = 48`;
  - tile columns `48 // 4 = 12` and `3 · 48 // 4 = 36`.
- **Arms**, in this fixed order, row-major over (tile row, tile column). Slices are half-open, `X[0, r0:r1, c0:c1]`.

  | Arm | Tile (row, col) | Slice | Layer | Blocks (8 columns each) |
  |---|---|---|---|---|
  | U-TILE-1 | (16, 12) | `Phi[0, 512:544, 384:416]` | 6 | q_proj A^T, q_proj B, k_proj A^T, k_proj B |
  | U-TILE-2 | (16, 36) | `Phi[0, 512:544, 1152:1184]` | 18 | same |
  | U-TILE-3 | (48, 12) | `Phi[0, 1536:1568, 384:416]` | 6 | same |
  | U-TILE-4 | (48, 36) | `Phi[0, 1536:1568, 1152:1184]` | 18 | same |

- **Each tile** is taken as a contiguous fp32 copy and reshaped to [1, 1, 32, 32], with no transpose, reordering,
  flip, padding, resampling or value change.
- **Layout meaning**, from `src/cg_fedllm/compression/layout.py` (`LayerMajorQKVOAtB.block_order`, blob
  `7b7b95479a7887ac20a8fbbfdcfc8af65fbac535`, unchanged at HEAD and at the P5-A execution commit):
  - column block i (8 columns) is `order[i]`, where `order` runs over the layers, then the modules (q, k, v, o),
    then the factors (A, B);
  - column c therefore belongs to layer `c // 64`, and to block `(c % 64) // 8` in the order qA^T qB kA^T kB vA^T vB
    oA^T oB;
  - each tile starts at a multiple of 64, so it covers the q/k blocks of one layer;
  - a row is a hidden index: an input-feature index in the A^T blocks and an output-feature index in the B blocks.
- **Relation to URP-v1.** The URP-v1 tile is the centre point of the same grid, (32, 24): `Phi[0, 1024:1056,
  768:800]`, layer 12, the same four blocks. It was chosen by a positional rule (the floor centre) registered before
  its values were read, not for favourable performance. It is not rerun here.
- **Reason, recorded before the values are known.** The quarter points spread the tiles over early and late layers
  (6, 18) and low and high hidden indices, while keeping the block composition of the URP-v1 tile. The rule has no
  free parameter except the stated floor convention.
- **Nothing else.** No other tile is extracted, inspected or evaluated by any model. Tiles are not chosen, replaced
  or reordered after their values are seen.

## Normalization (existing R3 definition, recorded scale)

- Mode `global_exact_maxabs_train`, with the frozen R3 scale `s_r = 0.015751632224572334`. That is `m_r / 0.95`,
  where `m_r` is the exact training-population maximum.
- It is applied with the production `Normalizer` arithmetic: `x_n = x / float32(s_r)` and
  `denormalize(y) = y · float32(s_r)`, in fp32. It is not refitted on any tile.

## Degeneracy (decided now; nothing is substituted)

A tile is **degenerate** if any of these holds:
1. an element of `x` is not finite;
2. `T = Σx² = 0`;
3. the scalar-mean baseline MSE is 0, i.e. the tile is constant;
4. `max |x_n| >= 1`: a Tanh output cannot reach that value.

- A degenerate tile is recorded as such, with the failed item. No model is trained on it.
- Under UPP-20261013-v1 it is the outcome DEGENERATE_INPUT, class F: an explicit non-pass, not an unknown.
- No other tile, snapshot or scale replaces it.
- Condition 4 cannot occur in this snapshot: its archived whole-Phi normalized max is 0.8891991972923279.
- A provenance failure, such as a Phi hash mismatch, produces no tile at all. The study then has no registered input
  and is not run.
- Reported only: the number of exactly-zero elements and the per-block energies.
