# Clarification of UOA-20261014-v1 wording (new note; the frozen UOA files are unchanged)

These clarifications narrow statements in `D:/projects-cd/cg-fedllm-uoa/diagnostics/uoa_20261014_v1_execution/`
(`REPORT.md`, `CONTINUATION_CHECKPOINT.md`). The UOA package, evidence, counters and recorded numbers are not
edited. Where this note and the UOA report differ in interpretation, this note is the more careful statement.

1. **"Stem gain."**
   - The UOA report says the 1-channel stem's "effective gain went from 7.97 at 3,000 updates to 0.0141 at 12,000"
     and that "the stem then passes a strongly attenuated linear signal" (a "1/70 attenuation").
   - The quantity reported is only the BatchNorm affine coefficient gamma / sqrt(running_var + eps) of the stem BN. It
     excludes the 3x3 convolution before it.
   - It is not the gain of the stem as a whole, nor any input-to-output gain of the network: later BN layers and
     weights can rescale the signal.
   - A small coefficient in one layer does not by itself show that information is lost. The offset observation (the
     stem BN output became all positive in the ReLU arm) stands as an observation.
2. **"Train- and eval-mode BatchNorm give almost the same loss."** That paired check (normalized MSE 0.021586 versus
   0.021592) covered the UCX-HIST model, plus the initial models, on the four registered batches. It was not run on
   the final UOA models, so it does not exclude a train / eval BatchNorm difference for U-RELU-LONG or U-LEAKY-LONG.
   ULR-20261015-v1 records that check for the final U-LEAKY-LONG model before adapting it.
3. **"The neural codec's error is representational, not quantization error."**
   - What was shown: the int8 latent adds very little error, so most of the error already exists in the float model.
   - This does not establish an architectural capacity ceiling. Optimization, normalization and structural
     constraints were not separated.
4. **Longer training and LeakyReLU.**
   - Both improved this configuration: 26.5% from four times the updates, and 6% to 9% from the activation change at
     matched budgets.
   - Neither identifies the dominant remaining failure mechanism.
   - The hypothesis about the rectified 1-channel stages remains a hypothesis.
5. **"Simple scalar quantizers need ... no training."**
   - They need no neural training, but they are not fit-free. UNIFORM-2BIT calibrated its range on TRAIN over a fixed
     grid, and SCALAR-4 ran 50 Lloyd iterations on all TRAIN elements.
   - The PCA baselines also required a TRAIN decomposition.
6. **"No train / held-out gap."** It is more accurate to say the gap was small: held-out RSE was about 0.015 to 0.023
   above TRAIN at 12,000 updates.
7. **"Fix the supervisor's target definition before any further neural start."** This UOA recommendation was
   superseded by the user's explicit new scope of 2026-10-15, which authorizes ULR-20261015-v1. The UOA recommendation
   itself remains as written.
