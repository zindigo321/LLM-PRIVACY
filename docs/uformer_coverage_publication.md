# Uformer primary policy and four-tile coverage results

UCOV-20261013-v1 met its registered coverage criterion under **UPP-20261013-v1**: **P = 4, F = 0, U = 0**, against the requirement **P >= 3**. All four registered arms ran to completion; the supervisor did not stop after three passes.

See the [frozen report](../diagnostics/ucov_20261013_v1_execution/REPORT.md), [analysis](../diagnostics/ucov_20261013_v1_execution/analysis/analysis.json) and [policy](../diagnostics/upp_20261013_v1/POLICY.md).

## Primary policy

The authoritative [upp_policy.json](../diagnostics/upp_20261013_v1/upp_policy.json) requires all three inclusive bounds:

- **RSE <= 0.50**;
- **cosine >= 0.65**;
- **MSE <= 0.80 times the best constant baseline**, defined as the smaller of zero-prediction MSE and scalar-mean MSE.

The decision uses the **final registered state, in eval mode and original input space**. Evidence and policy identity must verify, the input must be usable, and the arm must complete its planned updates without interruption, guard action or exceeded limits. Undefined or non-finite metrics cannot pass.

Historical strict and EP-v1 verdicts are secondary references; neither can veto a primary pass. With the recorded metric definitions, RSE <= 0.50 mathematically implies cosine >= approximately 0.7071. The requested cosine bound of 0.65 is retained and checked separately.

This policy was chosen knowing the earlier URP-v1 result and was specified before the UCOV run. Applying it to older results is retrospective. Historical criteria, records and verdicts remain unchanged.

## Recorded results

Each arm trained a fresh Uformer on its own 32 x 32 tile, using the same seed, initial state and 3,000-update budget. Each slice below is `Phi[0, rows, columns]`.

| Arm | Rows; columns | Layer | RSE | Cosine | MSE / best constant | Primary | Strict | EP-v1 |
|---|---|---|---:|---:|---:|---|---|---|
| U-TILE-1 | `512:544; 384:416` | 6 | 0.22669 | 0.87940 | 0.22693 | PASS | FAIL | PASS |
| U-TILE-2 | `512:544; 1152:1184` | 18 | 0.21596 | 0.88546 | 0.21622 | PASS | FAIL | PASS |
| U-TILE-3 | `1536:1568; 384:416` | 6 | 0.21478 | 0.88613 | 0.21479 | PASS | FAIL | PASS |
| U-TILE-4 | `1536:1568; 1152:1184` | 18 | 0.19796 | 0.89557 | 0.19798 | PASS | FAIL | PASS |

Values are rounded; the linked analysis retains full precision. Strict means RSE <= 0.01 and cosine >= 0.99. EP-v1 uses its frozen rules, including cosine >= 0.80. All four arms also pass EP-v1, so their observed passes do not depend on lowering the cosine threshold to 0.65.

The single invocation completed all four arms without guard action, interruption or stderr output. The frozen report records agreement among the analyzer, independent metric recomputation and forward-only replay. UCOV-v1 is closed: its one invocation and four starts are consumed. Publication preserves these verification records; it does not rerun the scientific experiment or constitute independent scientific acceptance.

## Publication and validation

The [archive index](../diagnostics/uformer_coverage_archive_20261013/archive_index.json) covers **133 evidence files totaling 1,026,024 bytes**: 103 files from the policy, package and execution directories, plus 30 small original run records under `diagnostics/uformer_coverage_archive_20261013/raw_runs/`.

The 14 frozen dependency files are already published through PR #13. The externally referenced U-FIT and U-PATCH child records also have copies in the [earlier archive](../diagnostics/uformer_archive_20261013/archive_index.json). Eight UCOV state files and three R3 source artifacts remain outside the repository. Original absolute paths and frozen bytes are preserved; a checkout alone is **not self-contained for full package verification or replay**. The new index maps relocated records and distinguishes them from unavailable external payloads.

All nine recorded pre-freeze check steps passed. The policy has 133 tests covering boundaries, invalid outcomes and secondary criteria that cannot veto a primary pass. CI explicitly includes those tests alongside the repository suite:

```text
pytest -q tests diagnostics/upp_20261013_v1/test_upp_classify.py -m "not gpu and not model"
```

The archived repository-gate log's 445 passing tests came from the original working directory, which also contained paused SSBD tests. That number is not a test result for this publication branch; publication checks use its own source and exclude unrelated SSBD work.

## Interpretation and next step

The result supports successful individual fitting across four additional q/k tiles of one snapshot. It uses one seed per arm and a latent with **64 times as many elements as the input**. It establishes neither compression, held-out generalization, whole-snapshot reconstruction nor downstream or FAF benefit.

The [UHO-v1 proposal](../diagnostics/ucov_20261013_v1_execution/NEXT_EXPERIMENT_PROPOSAL.md) would train one shared model on eight tiles and evaluate four held-out tiles. It was a proposal only at the close of UCOV-v1 and is outside this publication.
