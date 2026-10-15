# ULR-20261015-v1 continuation checkpoint (working notes; the evidence is REPORT.md)

## Final state

- **Closed.** ULR-20261015-v1 is CLOSED. Never relaunch it, rename it, reset its counters, or add an arm, seed or
  width under it. `launch/` exists, so the start is consumed.
- **Consumed.**
  - 1 / 1 invocation;
  - 3 / 3 neural starts (C16-CONTROL, C4-REDUCED, C1-REDUCED, 6,000 updates each, 18,000 in all);
  - 1 / 1 channel-PCA fit, and 0 input-PCA or quantizer fits;
  - 3 / 3 synthetic pilots, plus one engineering fixture test of the pilot runner (not a pilot).
- **Time.** About 2,030 s of the 9,000-s scientific wall time were used, and about 504 s of the 2,000-s evaluation
  and verification budget.
- **Git.** No commit, push or publication. The worktree `D:/projects-cd/cg-fedllm-ulr` holds two untracked
  directories; see `PUBLICATION_HANDOFF.md`.
- **Protected work.** It is unchanged (`protected_fingerprint_comparison.out.txt`): the paused SSBD work, UCX, UHO,
  UOA and the closed run roots.

## Timeline (2026-10-15, +08:00)

- 10:35: state verified; worktree created; protected-work fingerprint taken.
- 10:58-10:59: three pilots, all GO.
- 11:08:58: frozen (`8db0e72f9f9315c105dd0e82b5bc189b15f078ff4b300b18222f666bd9b00bb6`).
- 11:09:43: launched once (detached).
- 11:38:17: `LAUNCH_SCRIPT_END scientific_exit_code=0`, with all five steps COMPLETED.
- 11:39-11:43: the analyzer, independent verification (all_ok) and replay (126 / 126 bit-identical) were each
  executed once.
- The session was then interrupted and resumed; nothing was re-executed. The tables, figures, protected-work
  comparison (identical), manifest re-check (61 / 61) and documents followed.

## Single next recommendation (evidence-based; not executed)

Before any further neural start, test whether the learned codec can beat the TRAIN-fitted input PCA at a matched
element count when the bottleneck is part of training from the start. For example, make the bottleneck the training
objective's own geometry, with the same budget and the same input-PCA comparator. Do not adapt it from the 16-channel
source.

The evidence for this:

- At every tested count (16, 64, 256), the warm-started reduced compressor lost to input PCA with the same number of
  fp32 values.
- The fixed adaptation recovered only 10% to 22% of the bottleneck damage.
- The source latent's channel spectrum is flat: 4 of 16 components carry 37% of its variance.

So cutting channels from a model trained without a bottleneck is a poor starting point, and further training of
these arms is unlikely to change the ranking.

This would be a new protocol with its own authorization. It is not a fourth arm of ULR. Whether a learned codec can
beat the simple codecs at equal bytes remains the prerequisite for the innovation hypotheses.

## Not authorized (unchanged)

SSBD, FAF, LLM, GPU, DP, online adaptation, residual replay, extra seeds, 1/64 runs outside this study, QAT, another
bottleneck family in this session, downstream experiments, and any Git write.
