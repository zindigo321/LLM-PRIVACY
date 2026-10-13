# UCOV-20261013-v1: execution authorization record

This directory is outside every manifest-covered path. It holds the authorization, the pre-launch checks, the launch,
the analysis and the handoff.

## Authorization, as given by the user (task message of 2026-10-13, about 19:50 +08:00)

- **Approved: the design, before any manifest existed.**
  - The primary policy UPP-20261013-v1 (RSE ≤ 0.50, cosine ≥ 0.65, MSE ≤ 0.80 × best constant baseline), and its
    implementation.
  - Engineering checks, input preparation and the package freeze.
  - Bounded scientific execution of the four-tile input-coverage study, once the frozen package is verified to match
    the scope. No duplicate approval is to be requested.
- **No digest was approved in advance.** The user explicitly approved the design, not a digest. The frozen digest is
  recorded below as the identity of the executed package. No earlier human approval of that digest exists, and none
  is claimed.
- **Superseded:** the separate hash-approval requirement in
  `diagnostics/urp_20261013_v1_execution/NEXT_EXPERIMENT_PROPOSAL.md`, for this exact study only. That proposal is
  sealed and unchanged.

## Frozen package (recorded at freeze)

- `diagnostics/ucov_20261013_v1/manifest_ucov.json`, SHA-256
  `3bc2ab302f6cd77b63b9bb8b77894041fd97cbdd63d5fa7f380a4dae080fe824` (17,562 bytes).
  - It lists 65 files plus 5 external files bound by hash.
  - Built at 2026-10-13T21:18:48 +08:00 and read back 65/65 and 5/5. Only the manifest itself is unlisted.
- Primary policy: `upp_policy.json` `74b277b952e0af68dee00692cde6b95f8ed747d641843bbe98517c70d98c5770` and
  `upp_classify.py` `31904623223cc18b216b97bf2cfaa27734487c38a93ebe57efca5de84abecbb1`.
- Resolved configuration: `resolved_config.json` `3db88e97fbe0c27806104d2b98ded6e17544976a446a232a909a9540c3056337`.

## Scope of execution (exactly as authorized)

- One scientific invocation, with at most four scientific starts: U-TILE-1, U-TILE-2, U-TILE-3 and U-TILE-4, in
  order, run one after another.
- Per-arm limit 1,800 s; aggregate limit 9,000 s; RSS min(4 GiB, 50% of available RAM at launch); retained evidence
  512 MiB.
- All four registered tiles run after ordinary PASS or FIT_FAIL outcomes. Global resource or integrity failures stop
  safely, and the remaining arms are recorded.
- Not authorized: a retry, a resume, state transfer, an extra seed, a replacement tile, an extra arm, a budget
  extension, GPU work, SSBD, other experiments, dependency changes, and any Git write (staging, commit, push, PR,
  merge, branch switch, reset, clean, stash).

## Planned evidence layout (fixed before launch)

- `prelaunch_verify.py`, `prelaunch_verify.out.txt`, `prelaunch_check.md`: read-only verification.
- `launch/`: written by `launch_scientific.ps1`, started once and detached.
- `analysis/`: written by `ucov_analyze.py --evidence-dir <this directory>\analysis`.
- `independent_verify.py` and `replay_final_states.py` with their outputs: torch-free recomputation, and a
  forward-only replay of each saved final state.
- `REPORT.md`, figures, `NEXT_EXPERIMENT_PROPOSAL.md`, `PUBLICATION_HANDOFF.md`, `CONTINUATION_CHECKPOINT.md`, and
  `EVIDENCE_INVENTORY_SHA256.txt` last.
