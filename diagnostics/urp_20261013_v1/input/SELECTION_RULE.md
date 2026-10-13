# URP-20261013-v1 input selection rule (fixed before any payload is read)

This rule is written before the snapshot payload is opened, before any patch value is seen and before any model is
evaluated on any patch. Only these were seen beforehand: the archived identities of the source snapshot and the
whole-snapshot normalized-range summary in its P5-A record (max |x_n| 0.8892, p50 0.0775, p99 0.3431). No per-position
value has been seen. `prepare_input.py` pins this file's SHA-256 and refuses to run if the file changes.

## Source (no choice)

The R3 training snapshot of the historical P5-A single-snapshot control:
- the representation `balanced_effective_delta_r8`;
- time index 7 and client 91, which is position 39 of the 80-record R3 training population;
- `results/phase5/p5a_v2/R3_balanced_effective_delta_r8_single_snapshot/p5a_record.json`.

It is rebuilt by the documented repository procedure, exactly as `cg_fedllm.phase5.p5a_inputs.build_population` does:
- `load_states(root, record, verify=True)` with file-byte and adapter-content hashes;
- `balanced_effective_delta(start, end, s=2.0, rank=8)[0]`;
- `get_layout("layer_major_qkvo_AtB").forward(rep, geometry)`.

The result is a Phi tensor `X` of shape [1, 2048, 1536], fp32. Its `tensor_sha256(X)` must equal the archived
`phi_sha256` `485cbe5f3a450b43d7780fe9cd97afe5169ea81ce0bdaf21166938adfcac2546`. Otherwise provenance is not
established and no patch is produced.

## Positional rule (one patch)

- Grid: non-overlapping 32 × 32 tiles of `X[0]` ([2048, 1536]), aligned at multiples of 32, so 64 tile rows by 48 tile
  columns.
- Tile: the floor-centre tile, `(ti, tj) = (64 // 2, 48 // 2) = (32, 24)`.
- Rows `[32·ti, 32·ti + 32) = [1024, 1056)`; columns `[32·tj, 32·tj + 32) = [768, 800)`.
- Patch: `x = X[0, 1024:1056, 768:800]`, taken as a contiguous fp32 copy and reshaped to [1, 1, 32, 32].
  - It is the same shape as the U-FIT input.
  - No transpose, reordering, flip, padding, resampling or value change is applied.
- Layout meaning (`layer_major_qkvo_AtB`, 64 columns per layer, eight 8-column blocks in the order
  qA^T qB kA^T kB vA^T vB oA^T oB):
  - columns 768–799 are layer 12: q_proj A^T (768–775), q_proj B (776–783), k_proj A^T (784–791) and k_proj B
    (792–799);
  - rows 1024–1055 are hidden indices 1024–1055. In the A^T blocks a row is an input-feature index; in the B blocks it
    is an output-feature index.
- Reason, recorded before the values are known: the tile is aligned with the layout's 8-column factor blocks and its
  64-column layer groups. It sits at the middle layer and middle hidden indices, and has no free parameter except the
  stated floor convention.
- No other tile is extracted, inspected or evaluated by any model.

## Normalization (existing R3 definition, recorded scale)

- Mode `global_exact_maxabs_train`, with the frozen R3 scale `s_r = 0.015751632224572334`. This is
  `m_r / 0.95`, where `m_r = 0.014964050613343716` is the exact training-population maximum, archived with
  `scale_check.bitwise_equal: true`.
- The scale is applied with the production `Normalizer` arithmetic: `x_n = x / float32(s_r)` and
  `denormalize(y) = y · float32(s_r)`, in fp32. It is not refitted on the patch.

## Degeneracy and feasibility (decided now; nothing is substituted)

The patch is **degenerate** if any of these holds:

1. an element of `x` is not finite;
2. `T = Σx² = 0`;
3. the scalar-mean baseline MSE is 0, i.e. the patch is constant;
4. `max |x_n| >= 1`: a Tanh output cannot reach that value.

A degenerate patch, or a provenance failure, blocks this diagnostic. It is reported with the specific failed item.
No other tile, snapshot or scale replaces it. The number of exactly-zero elements and the per-block energies are
reported only.
