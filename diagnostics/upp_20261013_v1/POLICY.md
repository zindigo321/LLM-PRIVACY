# UPP-20261013-v1: primary policy for Uformer reconstruction diagnostics

`upp_policy.json` in this directory is the authoritative source of every value below. `upp_classify.py` reads it
and refuses to run if its content differs from the values it was written for. A package that uses this policy pins
the SHA-256 of both files and records the policy ID and hashes in its resolved configuration, execution records and
analysis.

## Status and disclosure

- **Prospective** from UCOV-20261013-v1, the four-tile input-coverage study. It is that study's primary criterion.
- **Chosen by the user on 2026-10-13, with knowledge of the URP-20261013-v1 result**: final eval RSE 0.1935 and
  cosine 0.8981 on the centre tile.
- **Earlier results are not changed.** Applying the policy to an earlier result is retrospective, is labelled as
  such and never replaces the recorded verdict. URP-20261013-v1 remains COMPLETED_FIT_FAIL under its strict criterion
  and PASS under the pre-specified EP-20261012-v1.

## Primary criterion

The basis is one arm's **final registered state** (never a best intermediate point), in **eval mode**, in the
**original space** of its input. The quantities are float64 functions of the recorded sums SSE, T, P, D and N:

| Quantity | Definition |
|---|---|
| RSE | SSE / T (undefined if T ≤ 0) |
| cosine | D / sqrt(P · T) (undefined if P ≤ 0 or T ≤ 0) |
| MSE | SSE / N |
| best baseline | min(zero-prediction MSE, scalar-mean MSE); the scalar mean predicts one scalar, the mean of all N elements, across the whole input |

A valid completed arm is a **primary PASS** when all three hold:
- **RSE ≤ 0.50**;
- **cosine ≥ 0.65**;
- **MSE ≤ 0.80 × best baseline**.

The comparisons are inclusive, in float64, against these literal values, with no epsilon.

**Relationship between the bounds.** With these definitions, SSE = P + T − 2D. For a fixed RSE the cosine is
smallest when the norm ratio sqrt(P/T) equals sqrt(1 − RSE). That minimum is sqrt(1 − RSE), so RSE ≤ 0.50 implies
cosine ≥ sqrt(0.5) ≈ 0.7071 in exact arithmetic. In practice the cosine bound 0.65 is therefore implied by the RSE
bound, apart from rounding. It is kept exactly as requested, and is checked and reported separately; the metric
definitions are unchanged.

## Outcomes (one per arm, checked in this order)

| Outcome | Condition | Class |
|---|---|---|
| UNVERIFIED | evidence not verified: identity, hashes, record agreement or policy identity fail, or a required quantity is absent | U |
| DEGENERATE_INPUT | the registered input is degenerate (non-finite element, T = 0, constant, or normalized max \|x_n\| ≥ 1); no model is trained | F |
| NOT_RUN | the arm was never started | U |
| NOT_ELIGIBLE | not a valid completion: not COMPLETED, updates done ≠ planned, interrupted, a guard acted, or a limit was exceeded | U |
| FIT_FAIL | valid completion, but a required final quantity is non-finite or undefined, or RSE, cosine or the baseline condition fails | F |
| NONINFORMATIVE | valid completion with RSE and cosine within bounds, but the best baseline MSE is 0, so the baseline condition carries no information | F |
| PASS | valid completion and all three conditions hold | P |

- **Two kinds of non-pass.** An execution or evidence failure (U) is an unknown outcome, not a statement about model
  capacity. A valid fitting failure, or a registered degenerate input, is an explicit non-pass (F).
- **Undefined is not a pass.** An undefined or non-finite required quantity cannot pass.

## Aggregate decision for a multi-arm study

Count P, F and U over the registered arms; every arm counts, including the unknowns. For UCOV-20261013-v1 (four
arms):
- **P ≥ 3:** the pre-specified coverage criterion is met.
- **F ≥ 2:** the criterion is not met.
- **Otherwise:** inconclusive.

With four arms, P ≥ 3 and F ≥ 2 cannot both hold.

## Secondary references (reported only; neither can veto a primary PASS)

- **Historical strict near-identity.** The UID-20261012-v2 criteria: RSE ≤ 0.01 and cosine ≥ 0.99 (eval mode,
  finite).
- **EP-20261012-v1.** RSE ≤ 0.50, cosine ≥ 0.80 and MSE ≤ 0.80 × best baseline, with its validity rules. It is
  evaluated by the frozen `ep_classify.classify`.

## What a PASS means

A primary PASS means only this: a completed, verified run reconstructed its one fixed input, from its final state,
clearly better than any constant prediction. It does not mean identity reconstruction, compression, generalization
to held-out inputs, whole-snapshot reconstruction, downstream usefulness or FAF utility.
