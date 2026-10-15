"""UOA-20261014-v1 codecs and payload formats (protocol sections 6 and 8). numpy only (imported lazily). Every
payload is an explicit byte string; a decoder receives only those bytes and the persisted shared state.

Arithmetic, as registered:
- **2-bit packing.** A tile's 1,024 indices (row-major order of the [1, 32, 32] tile) go into 256 bytes. Element e is
  stored in byte e // 4 at bits 2 (e % 4) and 2 (e % 4) + 1, least significant pair first.
- **4-level scalar quantizer.** Levels L0 < L1 < L2 < L3 are fp32 shared state. The thresholds are the float64
  midpoints t_k = (L_k + L_{k+1}) / 2 of the fp32 levels. The index of x is the number of thresholds <= float64(x),
  so a value exactly on a threshold goes to the upper level. The decode is levels[index] in fp32.
  - **UNIFORM-2BIT.** Levels float32([-a, -a/3, a/3, a]). a is chosen on TRAIN from (k / 10) x TRAIN RMS for
    k = 5..60, plus a = TRAIN max |x|, by the smallest TRAIN SSE of the deployed codec. Equal SSE goes to the smaller a.
  - **SCALAR-4.** A Lloyd fit on all TRAIN elements in float64. Initial centroids are the linear quantiles at
    0.125 / 0.375 / 0.625 / 0.875. There are exactly 50 iterations, each assigning by the midpoint rule above and
    then replacing each centroid by the math.fsum mean of its members. An empty cell keeps its centroid. The centroids
    must stay strictly increasing, or the fit refuses. The final centroids are cast to fp32.
- **PCA.** One fresh float64 decomposition (vendored ``ucx_codecs.pca_fit``, 128 components). The persisted fp32 mean
  and basis are used by both encoder and decoder.
  - **PCA-64.** c = float32((x64 - mean) @ basis[:64]^T), 64 little-endian fp32 values = 256 B.
  - **PCA-128-HALF.** c = float16((x64 - mean) @ basis[:128]^T) with a single IEEE round-to-nearest-even from
    float64, 128 little-endian fp16 values = 256 B. A non-finite fp16 coefficient refuses.
  - **Decode.** x_hat = float32(mean64 + float64(c) @ basis64[:k]), where mean64 and basis64 are the float64 values of
    the fp32 state.
- **INT8 latent.** There are 16 per-channel fp32 scales, scale[c] = float32(max_abs_train_latent[c] / 127).
  - The codes are k = clip(rint(z64 / float64(scale[c])), -127, 127), with rint rounding half to even.
  - The 256 int8 bytes follow the latent's row-major [16, 4, 4] order: byte c * 16 + h * 4 + w.
  - The decode is z_hat = float32(k) * scale[c] in fp32.
  - A channel whose scale is exactly 0 always has code 0 and decodes to exactly 0.
"""

from __future__ import annotations

import math
from typing import Any

import uoa_spec as S


def np():
    import numpy

    return numpy


# ---- 2-bit packing -------------------------------------------------------------------------------------------------
def pack2(idx) -> bytes:
    n = np()
    a = n.asarray(idx, dtype=n.uint8).reshape(-1)
    if a.size != S.N_TILE or int(a.max(initial=0)) > 3:
        raise ValueError(
            f"2-bit payload needs {S.N_TILE} indices in 0..3, got {a.size} (max {a.max(initial=0)})"
        )
    q = a.reshape(-1, 4)
    return (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).astype(n.uint8).tobytes()


def unpack2(data: bytes):
    n = np()
    if len(data) != S.N_TILE // 4:
        raise ValueError(f"2-bit payload has {len(data)} bytes, not {S.N_TILE // 4}")
    b = n.frombuffer(data, dtype=n.uint8)
    return n.stack([(b >> s) & 3 for s in (0, 2, 4, 6)], axis=1).reshape(-1).astype(n.uint8)


# ---- 4-level scalar quantizers --------------------------------------------------------------------------------------
def thresholds(levels32):
    n = np()
    lv = n.asarray(levels32, dtype=n.float32).astype(n.float64)
    if lv.shape != (S.LEVELS,) or not all(lv[i] < lv[i + 1] for i in range(S.LEVELS - 1)):
        raise ValueError(f"levels {lv} are not {S.LEVELS} strictly increasing values")
    return (lv[:-1] + lv[1:]) / 2


def sq_indices(x_rows, levels32):
    """[M, 1024] values -> uint8 indices (the number of thresholds <= float64(x))."""
    n = np()
    return n.searchsorted(thresholds(levels32), n.asarray(x_rows, dtype=n.float64), side="right").astype(
        n.uint8
    )


def sq_encode(x_rows, levels32) -> list[bytes]:
    return [pack2(r) for r in sq_indices(n_rows(x_rows), levels32)]


def sq_decode(payloads: list[bytes], levels32):
    n = np()
    lv = n.asarray(levels32, dtype=n.float32)
    thresholds(lv)
    return n.stack([lv[unpack2(b)] for b in payloads]).astype(n.float32)


def n_rows(x_rows):
    n = np()
    x = n.asarray(x_rows)
    return x.reshape(x.shape[0], -1)


def sq_sse(x_rows, levels32) -> float:
    """TRAIN SSE of the deployed codec: math.fsum of the per-tile float64 sums."""
    n = np()
    x = n_rows(x_rows)
    lv = n.asarray(levels32, dtype=n.float32)
    pred = lv[sq_indices(x, lv)].astype(n.float64)
    return math.fsum(n.square(pred - x.astype(n.float64)).sum(axis=1).tolist())


def uniform_levels(a: float):
    n = np()
    a = float(a)
    return n.asarray([-a, -a / 3.0, a / 3.0, a], dtype=n.float64).astype(n.float32)


def uniform_calibrate(x_rows) -> dict[str, Any]:
    """The registered grid on TRAIN; the smallest deployed SSE wins, equal SSE goes to the smaller a."""
    n = np()
    x = n_rows(x_rows)
    x64 = x.astype(n.float64)
    n_el = x64.size
    rms = math.sqrt(math.fsum(n.square(x64).sum(axis=1).tolist()) / n_el)
    max_abs = float(n.abs(x.astype(n.float32)).max())
    cands = [(k / 10.0 * rms, f"{k / 10.0:.1f} x TRAIN RMS") for k in S.UNIFORM_GRID_TENTHS]
    cands.append((max_abs, "TRAIN max |x|"))
    rows = []
    for a, label in cands:
        lv = uniform_levels(a)
        rows.append({"a": a, "label": label, "levels32": [float(v) for v in lv], "train_sse": sq_sse(x, lv)})
    best = min(rows, key=lambda r: (r["train_sse"], r["a"]))
    ties = [r for r in rows if r["train_sse"] == best["train_sse"]]
    return {
        "train_rms": rms,
        "train_max_abs": max_abs,
        "candidates": rows,
        "chosen": best,
        "tied_candidates": len(ties),
        "levels32": uniform_levels(best["a"]),
    }


def lloyd_step(xs, c):
    """One Lloyd iteration on sorted float64 values ``xs`` with centroids ``c``: midpoint assignment (a value on a
    midpoint goes to the upper cell), math.fsum means, an empty cell keeps its centroid. Returns
    (new centroids, counts, empty cells, distortion before the update)."""
    n = np()
    t = (c[:-1] + c[1:]) / 2
    b = [0, *(int(v) for v in n.searchsorted(xs, t, side="left")), xs.size]
    new = c.copy()
    counts, empty, distortion = [], [], 0.0
    for k in range(len(c)):
        seg = xs[b[k] : b[k + 1]]
        counts.append(int(seg.size))
        distortion += float(n.square(seg - c[k]).sum())
        if seg.size:
            new[k] = math.fsum(seg.tolist()) / seg.size
        else:
            empty.append(k)
    return new, counts, empty, distortion


def lloyd_fit(x_rows) -> dict[str, Any]:
    """The registered 50-iteration Lloyd fit on every TRAIN element (float64)."""
    n = np()
    xs = n.sort(n_rows(x_rows).astype(n.float64).reshape(-1))
    c = n.quantile(xs, list(S.LLOYD_INIT_QUANTILES), method=S.QUANTILE_METHOD).astype(n.float64)
    init = [float(v) for v in c]
    if not all(c[i] < c[i + 1] for i in range(S.LEVELS - 1)):
        raise ValueError(f"initial centroids {init} are not strictly increasing")
    history = []
    for it in range(1, S.LLOYD_ITERATIONS + 1):
        new, counts, empty, distortion = lloyd_step(xs, c)
        shift = float(n.abs(new - c).max())
        c = new
        if not all(c[i] < c[i + 1] for i in range(S.LEVELS - 1)):
            raise ValueError(f"Lloyd iteration {it}: centroids {list(c)} are not strictly increasing")
        history.append(
            {
                "iteration": it,
                "counts": counts,
                "empty_cells": empty,
                "distortion_before_update": distortion,
                "max_centroid_shift": shift,
                "centroids": [float(v) for v in c],
            }
        )
    lv = c.astype(n.float32)
    thresholds(lv)
    return {
        "init_quantiles": list(S.LLOYD_INIT_QUANTILES),
        "init_centroids": init,
        "iterations": S.LLOYD_ITERATIONS,
        "history": history,
        "final_centroids64": [float(v) for v in c],
        "levels32": lv,
        "train_sse_deployed": sq_sse(x_rows, lv),
    }


# ---- PCA payloads ---------------------------------------------------------------------------------------------------
def _pca_coef64(x_rows, mean32, basis32, k: int):
    n = np()
    xc = n_rows(x_rows).astype(n.float64) - n.asarray(mean32, dtype=n.float32).astype(n.float64)
    return xc @ n.asarray(basis32, dtype=n.float32)[:k].astype(n.float64).T


def _pca_decode64(coef64, mean32, basis32, k: int):
    n = np()
    m = n.asarray(mean32, dtype=n.float32).astype(n.float64)
    return (m + coef64 @ n.asarray(basis32, dtype=n.float32)[:k].astype(n.float64)).astype(n.float32)


def pca64_encode(x_rows, mean32, basis32) -> list[bytes]:
    n = np()
    c = _pca_coef64(x_rows, mean32, basis32, 64).astype(n.float32)
    return [r.astype("<f4").tobytes() for r in c]


def pca64_decode(payloads: list[bytes], mean32, basis32):
    n = np()
    if any(len(b) != 256 for b in payloads):
        raise ValueError("PCA-64 payloads must have 256 bytes")
    c = n.stack([n.frombuffer(b, dtype="<f4") for b in payloads]).astype(n.float64)
    return _pca_decode64(c, mean32, basis32, 64)


def pca128h_encode(x_rows, mean32, basis32) -> list[bytes]:
    n = np()
    c = _pca_coef64(x_rows, mean32, basis32, 128).astype(n.float16)
    if not bool(n.isfinite(c).all()):
        raise ValueError("a PCA-128-HALF coefficient is not finite in fp16")
    return [r.astype("<f2").tobytes() for r in c]


def pca128h_decode(payloads: list[bytes], mean32, basis32):
    n = np()
    if any(len(b) != 256 for b in payloads):
        raise ValueError("PCA-128-HALF payloads must have 256 bytes")
    c = n.stack([n.frombuffer(b, dtype="<f2") for b in payloads]).astype(n.float64)
    return _pca_decode64(c, mean32, basis32, 128)


def baseline_codecs(mean32, basis32, uniform32, scalar32) -> dict[str, tuple]:
    """The four registered 256-B baselines as (encoder: rows -> payloads, decoder: payloads -> fp32 rows). Each
    decoder sees only the payload bytes and the persisted shared state."""
    return {
        "PCA-64": (lambda r: pca64_encode(r, mean32, basis32), lambda b: pca64_decode(b, mean32, basis32)),
        "PCA-128-HALF": (
            lambda r: pca128h_encode(r, mean32, basis32),
            lambda b: pca128h_decode(b, mean32, basis32),
        ),
        "UNIFORM-2BIT": (lambda r: sq_encode(r, uniform32), lambda b: sq_decode(b, uniform32)),
        "SCALAR-4": (lambda r: sq_encode(r, scalar32), lambda b: sq_decode(b, scalar32)),
    }


# ---- INT8 latent ----------------------------------------------------------------------------------------------------
def latent_scales(max_abs_per_channel):
    """16 fp32 scales float32(max_abs / 127) from the exact fp32 per-channel maxima (as float64 values)."""
    n = np()
    m = [float(v) for v in max_abs_per_channel]
    if len(m) != S.LATENT_SHAPE[0] or any(not math.isfinite(v) or v < 0 for v in m):
        raise ValueError(f"per-channel maxima {m} are not 16 finite non-negative values")
    return n.asarray([n.float32(v / S.INT8_MAX_CODE) for v in m], dtype=n.float32)


def latent_int8_codes(z, scales32) -> tuple[Any, Any, Any]:
    """(int8 codes [256], saturated count per channel [16], non-zero values in zero-scale channels [16])."""
    n = np()
    c_n = S.LATENT_SHAPE[0]
    z64 = n.asarray(z, dtype=n.float32).astype(n.float64).reshape(c_n, -1)
    q = n.asarray(scales32, dtype=n.float32).astype(n.float64)
    codes = n.zeros(z64.shape, dtype=n.int8)
    sat = n.zeros(c_n, dtype=n.int64)
    zero_ch = n.zeros(c_n, dtype=n.int64)
    for c in range(c_n):
        if q[c] == 0.0:
            zero_ch[c] = int((z64[c] != 0).sum())
            continue
        r = n.rint(z64[c] / q[c])
        sat[c] = int((n.abs(r) > S.INT8_MAX_CODE).sum())
        codes[c] = n.clip(r, -S.INT8_MAX_CODE, S.INT8_MAX_CODE).astype(n.int8)
    return codes.reshape(-1), sat, zero_ch


def latent_int8_to_bytes(codes) -> bytes:
    a = np().asarray(codes, dtype="i1").reshape(-1)
    if a.size != S.LATENT_N:
        raise ValueError(f"INT8 latent payload has {a.size} codes, not {S.LATENT_N}")
    return a.tobytes()


def latent_int8_from_bytes(data: bytes):
    if len(data) != S.LATENT_N:
        raise ValueError(f"INT8 latent payload has {len(data)} bytes, not {S.LATENT_N}")
    return np().frombuffer(data, dtype="i1").copy()


def latent_int8_dequant(codes, scales32):
    """fp32 [1, 16, 4, 4]: float32(code) * scale[c], one fp32 multiplication."""
    n = np()
    k = n.asarray(codes, dtype=n.int8).astype(n.float32).reshape(S.LATENT_SHAPE[0], -1)
    q = n.asarray(scales32, dtype=n.float32).reshape(-1, 1)
    return (k * q).astype(n.float32).reshape(1, *S.LATENT_SHAPE)
