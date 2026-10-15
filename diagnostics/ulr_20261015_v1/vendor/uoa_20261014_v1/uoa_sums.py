"""UOA-20261014-v1 production sums (numpy, imported lazily): per-tile float64 SSE, T, P, D; per-block sums over the
four 8-column groups of a tile; ratio-of-sums pooling with math.fsum. The independent verifier re-implements these
without importing this module."""

from __future__ import annotations

import math
from typing import Any

import uoa_spec as S

SUM_KEYS = ("sse", "target_energy", "pred_energy", "dot")


def _rows(a):
    import numpy as n

    x = n.asarray(a)
    return x.reshape(x.shape[0], -1).astype(n.float64)


def sums4(pred, target):
    """float64 [M, 4]: (SSE, T, P, D) of every tile (row) of ``pred`` against ``target``."""
    import numpy as n

    a, b = _rows(pred), _rows(target)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return n.stack(
        [n.square(a - b).sum(axis=1), n.square(b).sum(axis=1), n.square(a).sum(axis=1), (a * b).sum(axis=1)],
        axis=1,
    )


def block_sums(pred, target):
    """float64 [M, 4 blocks, 4]: the same sums within each 8-column block (q A^T, q B, k A^T, k B) of a tile."""
    import numpy as n

    a = _rows(pred).reshape(-1, S.TILE, len(S.BLOCKS), S.BLOCK_COLS)
    b = _rows(target).reshape(-1, S.TILE, len(S.BLOCKS), S.BLOCK_COLS)
    ax = (1, 3)
    return n.stack(
        [
            n.square(a - b).sum(axis=ax),
            n.square(b).sum(axis=ax),
            n.square(a).sum(axis=ax),
            (a * b).sum(axis=ax),
        ],
        axis=2,
    )


def finite_rows(pred):
    import numpy as n

    return n.isfinite(_rows(pred)).all(axis=1)


def pooled(arr4, finite=None, n_per_row: int = S.N_TILE) -> dict[str, Any]:
    """Ratio of math.fsum-pooled sums; plus the means of the per-row ratios (defined rows only)."""
    import numpy as n

    a = n.asarray(arr4, dtype=n.float64)
    out: dict[str, Any] = {"members": int(a.shape[0]), "n": int(a.shape[0]) * n_per_row}
    out["finite"] = bool(n.isfinite(a).all()) and (True if finite is None else bool(n.asarray(finite).all()))
    for j, k in enumerate(SUM_KEYS):
        out[k] = math.fsum(a[:, j].tolist())
    sse, t, p, d = (out[k] for k in SUM_KEYS)
    out["mse"] = sse / out["n"]
    out["rse"] = sse / t if t > 0 else None
    out["cosine"] = d / math.sqrt(p * t) if p > 0 and t > 0 else None
    out["norm_ratio"] = math.sqrt(p / t) if t > 0 else None
    ok = (a[:, 1] > 0) & (a[:, 2] > 0)
    if ok.any():
        out["mean_of_row_rse"] = math.fsum((a[ok, 0] / a[ok, 1]).tolist()) / int(ok.sum())
        out["mean_of_row_cosine"] = math.fsum((a[ok, 3] / n.sqrt(a[ok, 1] * a[ok, 2])).tolist()) / int(
            ok.sum()
        )
    out["rows_with_undefined_cosine"] = int((~ok).sum())
    return out


def normalized_from_raw(pool_raw: dict[str, Any], s32: float) -> float:
    """The crosswalk normalized MSE of a raw-space pooled unit: raw MSE / s32^2 (float64)."""
    return pool_raw["mse"] / (s32 * s32)
