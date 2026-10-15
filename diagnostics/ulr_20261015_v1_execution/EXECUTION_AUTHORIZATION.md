# ULR-20261015-v1 execution authorization (record)

- **Authorization.** The user's message of 2026-10-15 authorized, subject to tool permissions and the stated limits:
  - a new isolated study (Uformer Latent Reduction), its implementation and verification;
  - one TRAIN-only channel-PCA initialization;
  - at most one synthetic pilot per arm;
  - exactly THREE scientific adaptation starts of 6,000 updates each.

  It asked that the work proceed without another routine "go".
- **Executed identity.** The frozen package `diagnostics/ulr_20261015_v1/manifest_ulr.json`, SHA-256
  `8db0e72f9f9315c105dd0e82b5bc189b15f078ff4b300b18222f666bd9b00bb6` (61 files, 12 external files), frozen
  2026-10-15 11:08:58 +08:00.
  - Pre-launch verification: all_ok (`prelaunch_verify.out.txt`).
  - The digest is the identity executed under that advance authorization; it was not human-approved.
- **Scope.** One invocation of five steps. Consumed before the freeze: three synthetic pilots, all GO (76.2 s). Not
  used: any new input-PCA or scalar-quantizer fit.
- **Not authorized and not done.** Commit, push, publication, SSBD, LLM / FAF / GPU / DP runs, a fourth arm, another
  seed, another bottleneck family, QAT, downstream experiments, and any relaunch of UOA, UCX or an earlier closed
  protocol.
