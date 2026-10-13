"""ENGINEERING CHECK: the protocol's analytical metric cases, applied to the frozen ``uid_common.metrics`` / ``passes``,
and an independent recomputation from the recorded reductions (as the post-run evaluation will do). Synthetic
fixture tensors only; nothing registered.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1.parent / "uid_20261012_v2"))

import uid_common as C  # noqa: E402  (frozen)

CRIT = {"max_rse": 0.01, "min_cosine": 0.99}


@pytest.fixture()
def x():
    g = torch.Generator().manual_seed(555)  # engineering seed
    return torch.rand((1, 1, 8, 8), generator=g) * 1.6 - 0.8


def test_identity_prediction(x):
    m = C.metrics(x.clone(), x)
    assert m["rse"] == 0.0 and m["sse"] == 0.0 and m["cosine"] == pytest.approx(1.0, abs=1e-12)
    assert m["norm_ratio"] == pytest.approx(1.0, abs=1e-12) and C.passes(m, CRIT)


def test_zero_prediction(x):
    m = C.metrics(torch.zeros_like(x), x)
    assert m["rse"] == pytest.approx(1.0, abs=1e-12)
    assert m["cosine"] is None and not C.passes(m, CRIT)  # undefined cosine cannot pass


@pytest.mark.parametrize("a", [0.0, 0.5, 0.95, 1.05, 2.0, -1.0])
def test_scaled_prediction(x, a):
    # built in float64: a float32 product a*x is rounded (~1e-8 per element), which is not small relative to
    # (a-1)^2 = 0.0025 at a = 1.05 (first run of this test: rel. error 1.8e-6 -> failure log kept in evidence/)
    xd = x.double()
    m = C.metrics(a * xd, xd)
    assert m["rse"] == pytest.approx((a - 1.0) ** 2, rel=1e-6, abs=1e-12)
    if a > 0:
        assert m["cosine"] == pytest.approx(1.0, abs=1e-6)
    elif a < 0:
        assert m["cosine"] == pytest.approx(-1.0, abs=1e-6)
    else:
        assert m["cosine"] is None


def test_threshold_edges(x):
    # RSE = (a-1)^2: a = 0.95 -> 0.0025 (passes), a = 0.85 -> 0.0225 (fails); cosine is 1 in both
    assert C.passes(C.metrics(0.95 * x, x), CRIT)
    assert not C.passes(C.metrics(0.85 * x, x), CRIT)


def test_nonfinite_prediction_fails(x):
    y = x.clone()
    y.view(-1)[0] = float("nan")
    m = C.metrics(y, x)
    assert m["finite"] is False and not C.passes(m, CRIT)


def test_independent_recomputation_from_recorded_reductions(x):
    """RSE, cosine, norm ratio and q_n recomputed from the saved sums alone agree with the reported values."""
    g = torch.Generator().manual_seed(556)
    y = x + 0.05 * torch.randn(x.shape, generator=g)
    m = C.metrics(y, x)
    rse = m["sse"] / m["target_energy"]
    cos = m["dot"] / math.sqrt(m["pred_energy"] * m["target_energy"])
    ratio = math.sqrt(m["pred_energy"] / m["target_energy"])
    q_n = (m["sse"] / m["n"]) / (m["target_energy"] / m["n"])
    assert (rse, cos, ratio, q_n) == pytest.approx(
        (m["rse"], m["cosine"], m["norm_ratio"], m["q_n"]), rel=1e-12
    )
    # and the sums themselves against a float64 numpy-free reference
    a, b = y.double(), x.double()
    assert m["sse"] == pytest.approx(float(((a - b) ** 2).sum()), rel=1e-12)
    assert m["dot"] == pytest.approx(float((a * b).sum()), rel=1e-12)


def test_shared_scalar_mean_baseline_is_one_scalar(x):
    b = C.baselines(x)
    assert b["zero_prediction"]["rse"] == pytest.approx(1.0)
    mean = float(x.double().mean())
    assert b["shared_scalar_mean"]["rse"] == pytest.approx(
        float(((x.double() - mean) ** 2).sum() / (x.double() ** 2).sum()), rel=1e-12
    )
