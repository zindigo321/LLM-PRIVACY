# Exploratory progression policy EP-20261012-v1

This is a derived-analysis policy. It decides whether a completed small reconstruction diagnostic justifies further
small diagnostics. It lives outside every frozen package and changes no protocol, manifest, record, status or
criterion. The classifier is `ep_classify.py`, and its first application is `assessment_uid_r1_ufit.json`.

## Disclosure: selected after the result was known

This policy was selected on 2026-10-12, after the UID-20261012-v2 U-FIT result had been observed and reported. That
result came from the r1 execution, 19:53 to 20:01 +08:00: final eval RSE 0.2727, cosine 0.8540. The task instructions
of that day set the thresholds, with that result in view. For that run, the policy is a post-hoc classification. It is
not a pre-registered test and does not change the run's outcome. Meeting it is not evidence that the run succeeded.

- U-FIT remains COMPLETED_FIT_FAIL under its frozen strict criteria: finite output and metrics, RSE <= 0.01 and
  cosine >= 0.99, in eval mode. These criteria remain the standard for any claim of near-identity reconstruction.
- U-CHECK's pipeline sanity criterion is unchanged.
- P5-A and SSBD constants, historical manifests and manifest-covered files are untouched.
- The exploratory outcome is stored separately and never replaces a recorded status.

The policy can be named as a pre-specified criterion only for a diagnostic registered later, before that diagnostic
runs.

## What a PASS means

A PASS means only this:
- the run completed within its limits;
- its evidence was verified;
- its final reconstruction of its one fixed input was clearly better than trivial constant predictions.

It does not establish identity reconstruction, compression, generalization to other inputs, downstream usefulness or
the superiority of one architecture. It is not a prerequisite for other exploratory work. A FAIL does not, on its own,
block small follow-up diagnostics.

## Scope

The policy applies to a completed single-input reconstruction diagnostic of the U-FIT type:
- one fixed input tensor x with N elements;
- one reconstruction x_hat = Decoder(Encoder(x)) of the final state, never of a best point;
- eval mode as the basis. Train mode may be reported but is not gated.

## Quantities

The quantities come from the recorded float64 sufficient statistics of the final eval-mode reconstruction:
SSE = sum((x_hat - x)^2), T = sum(x^2), P = sum(x_hat^2), D = sum(x_hat * x), and N.

- RSE = SSE / T, undefined if T = 0.
- cosine = D / sqrt(P * T), undefined if P = 0 or T = 0.
- MSE = SSE / N.
- MSE_zero = T / N, the MSE of the zero prediction.
- MSE_mean is the MSE of a single scalar broadcast to the input's shape: the mean of all N elements of the tested input.
- best = min(MSE_zero, MSE_mean).

The policy does not use a per-position mean over a one-input population. That mean equals the input, so its error is 0.
Every quantity is in the space of the tested input. The policy never compares raw MSE across inputs of different
scales.

## Conditions

| ID | Condition |
|---|---|
| V1 | Valid evidence. The run's final artifact hash list must name every file the assessment reads. Every listed file must match the list in SHA-256 and size. The package identity recorded by the run must match the approved manifest hash, and the records must agree on status, update count, N and T. |
| V2 | Completed and compliant. The status is COMPLETED_PASS or COMPLETED_FIT_FAIL, and updates done equal updates planned. If the protocol has a prerequisite control, it is COMPLETED_PASS under its own unchanged criterion. The run was not interrupted and no guard acted. For each applicable limit, used <= limit: arm wall time, aggregate wall time, training starts, invocations, RSS and retained evidence bytes. RSS is the larger of the sampled maximum and the OS peak working set. |
| M0 | Defined and finite. The reconstruction is recorded as finite. SSE, T, P, D and MSE_mean are finite, with SSE >= 0, MSE_mean >= 0, T > 0 and P > 0. N is a positive integer. RSE and cosine are finite. |
| M1 | RSE <= 0.50 |
| M2 | cosine >= 0.80 |
| M3 | MSE <= 0.80 * best, with the right side computed in float64. Requires best > 0. |

Every comparison is inclusive and in float64, against the literal threshold, with no epsilon or tolerance.

## Outcomes

There is exactly one outcome, checked in this order:

1. UNVERIFIED: V1 fails, or a required quantity was not recorded. No metric result counts.
2. NOT_ELIGIBLE: V2 fails, for example after RESOURCE_STOP, EXECUTION_ERROR, INCOMPLETE or an exceeded limit.
3. FAIL: M0 fails, since an undefined or non-finite required quantity cannot pass; or M1, M2 or M3 is evaluated and
   fails.
4. UNINFORMATIVE: M0, M1 and M2 hold, but best = 0, so the baseline-improvement assessment carries no information.
   This is not a pass.
5. PASS: V1, V2, M0, M1, M2 and M3 all hold.

## Rationale

- The strict criterion (RSE <= 0.01, cosine >= 0.99) asks whether a network reproduces its input almost exactly. A
  completed run can miss it and still fit its input substantially. This policy separates partial fit that is worth one
  more small diagnostic from fit that is no better than a constant.
- Two absolute floors (M1, M2) are combined with one relative floor (M3). M3 covers inputs where a constant already
  explains most of the energy, such as data with a large mean; there, a low RSE alone would be trivial.
- In squared error, the mean is the best constant predictor, up to rounding to the input dtype. The zero prediction is
  included because it defines RSE. The minimum of the two is used.
- There is no epsilon. A value exactly at a threshold passes, and the next representable value beyond it fails.
- Evidence and compliance are checked first, so a metric value from an unverified or non-compliant run never yields
  PASS.
