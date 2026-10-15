# ULR: genuine latent reduction, with limited reconstruction quality

ULR-20261015-v1 verified that its independent receiver reconstructs each 1,024-element tile from exactly **256, 64, or 16 fp32 values**. All three adapted arms passed their registered UPP screen, but every arm lost to input PCA at the same element count. Reducing the latent therefore established actual payload reduction without establishing a learned-compression advantage.

This publication archives completed work. Follow-up research is paused; no new training, model evaluation, or scientific replay was performed for publication.

## Results

| Method | Transmitted elements and type | Payload per tile | EARLY RSE | LATE RSE | TRAIN normalized MSE |
|---|---|---:|---:|---:|---:|
| UOA Leaky source | 256 fp32 | 1,024 B | 0.6839 | 0.6842 | 0.015294 |
| C16-CONTROL final | 256 fp32 | 1,024 B | 0.6587 | 0.6622 | 0.014632 |
| C4-REDUCED final | 64 fp32 | 256 B | 0.8682 | 0.8742 | 0.019484 |
| C1-REDUCED final | 16 fp32 | 64 B | 0.9511 | 0.9531 | 0.021720 |
| PCA-256 | 256 fp32 | 1,024 B | 0.5966 | 0.6006 | 0.007035 |
| PCA-64 | 64 fp32 | 256 B | 0.8328 | 0.8398 | 0.015772 |
| PCA-16 | 16 fp32 | 64 B | 0.9307 | 0.9377 | 0.020331 |
| SCALAR-4 | 1,024 two-bit indices | 256 B | 0.1201 | 0.1182 | 0.002831 |
| UOA Leaky int8 latent | 256 int8 | 256 B | 0.6839 | 0.6843 | 0.015295 |
| INT8-SHARED input codec | 1,024 int8 | 1,024 B | 0.0002155 | 0.0002058 | 0.000004662 |

Values are rounded from the archived [report tables](../diagnostics/ulr_20261015_v1_execution/report_tables.md). Element counts and byte counts are distinct comparisons. These payload figures exclude shared encoder/decoder, basis and quantizer state; they are not total federated communication costs.

The source had 12,000 training updates. Each ULR arm inherited that source, inserted a TRAIN-fitted channel-PCA bridge, and adapted for 6,000 more updates. The reduced arms recovered about 10–22% of the initial insertion damage across the reported cohorts. They remained worse than the similarly adapted C16 control and input PCA. The single seed and previously observed EARLY/LATE cohorts make the findings exploratory.

No learned arm reached the secondary full-TRAIN, eval-mode normalized-MSE target of 1e-4: C16, C4 and C1 were approximately 146, 195 and 217 times above it. Among these references, only the 1,024-B input int8 codec was below that target.

## Policy and interpretation

ULR uses **UPP-20261014-v1**: valid completed evidence, pooled RSE <= 1.0 and cosine >= 0.1, with the frozen cohort-combination rules. The old constant-baseline condition is report-only. The pinned [policy JSON](../diagnostics/ulr_20261015_v1/vendor/uoa_20261014_v1/vendor/upp_20261014_v1/upp_policy.json) and classifier are vendored in the frozen package. This is a different, more permissive policy than UCOV's UPP-20261013-v1; it does not replace UCOV's historical verdicts.

All three final ULR arms passed at study level. C1 had one failing LATE tile (335/336 passed). A PASS establishes neither superiority to PCA nor reaching 1e-4. C1's RSE near 0.95 means its squared reconstruction error is about 95% of target energy; it does not measure the percentage of information retained.

The saved receiver replay reports 126/126 bit-identical checks, and the independent recomputation reports 57/57 byte checks. These are historical scientific verification results; this publication rechecks archived bytes, not model outputs.

Three interpretation clarifications apply to the preserved historical discussion:

- The UOA source already had a 256-element bottleneck. It had not been trained under the new 64/16-element constraint.
- The pooled 16-dimensional channel spectrum is not the complete flattened 256-dimensional latent spectrum. Its top-four variance fraction cannot establish a global compressibility limit.
- The tested curves establish limited recovery at the tested budget, not that additional training cannot change the ranking.

In the historical tables, the C16 insertion-damage denominator is near zero; its very large negative "fraction of damage recovered" values are numerically unstable and should not be interpreted. The quoted 10–22% range concerns C4/C1. Frozen reports and tables remain unmodified.

## Published evidence and reproduction limits

Start with the [frozen report](../diagnostics/ulr_20261015_v1_execution/REPORT.md), [tables](../diagnostics/ulr_20261015_v1_execution/report_tables.md), [element-count figure](../diagnostics/ulr_20261015_v1_execution/element_count_vs_quality.svg), and [archive index](../diagnostics/uformer_latent_reduction_archive_20261015/archive_index.json).

The index maps **149 original evidence files, totaling 5,701,408 bytes**, to their repository locations: 111 package/execution files and 38 small original run records. Original file bytes, absolute paths, historical instructions and saved failures are preserved. Local-path fingerprints contain project filenames and hashes, not the contents of unrelated unpublished work.

The frozen ULR manifest SHA-256 is:

```text
8db0e72f9f9315c105dd0e82b5bc189b15f078ff4b300b18222f666bd9b00bb6
```

Publication preparation checked all 61 package manifest entries, all 12 external bound entries and all 178 original inventory entries against their local sources. These checks do not mean all 178 entries are uploaded. Thirty run-root safetensors files (166,470,796 bytes), plus external UOA/UCX artifacts, remain outside Git. The index distinguishes published files from external references.

UOA and UCX are still unpublished as full studies. The required source modules are vendored here, but external data and states are not. A checkout alone is **not replay-complete**; frozen absolute paths would also require an explicit path mapping. No claim is made about complete LoRA updates, task responses, FAF utility, communication savings or innovation.

## Publication checks

The archive includes a standard-library-only byte verifier:

```text
python diagnostics/uformer_latent_reduction_archive_20261015/verify_archive.py
python diagnostics/uformer_latent_reduction_archive_20261015/verify_archive.py --git-tree HEAD
```

These commands validate the 149 published copies against the index and the 61 manifest-bound package files. They do not load tensors, train models or invoke scientific analysis. The second command checks actual committed blobs and therefore detects Git line-ending conversion.

Read-only lint inspection preserved the existing post-run script issues: three B023 diagnostics in one expression in report_tables.py, and three scripts that would be reformatted. The [publication check record](../diagnostics/uformer_latent_reduction_archive_20261015/publication_checks.json) records these as existing archive findings; no executed evidence was silently reformatted. Existing repository CI still covers its configured source/test scope.

Future experiments suggested in historical continuation documents are not started or authorized by this publication. SSBD remains paused.
