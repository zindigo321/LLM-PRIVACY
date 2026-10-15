"""UCX-20261014-v1 codecs and payload formats (rule section 6): normalization, PCA-256, INT8-SHARED, INT8-TILE, and
the U-QUARTER latent payload. numpy only (imported lazily); every payload is an explicit little-endian byte string, and
only those bytes reach a decoder.

Arithmetic, as registered:
- normalization: s = m / 0.95 (float64); x_n = x / float32(s), y = y_n * float32(s), in fp32 (production Normalizer);
- PCA encoder: c = float32((float64(x) - float64(mean32)) @ float64(basis32)^T); decoder:
  x_hat = float32(float64(mean32) + float64(c) @ float64(basis32));
- int8: k = clip(rint(float64(x) / float64(q)), -127, 127) as int8 (rint: round half to even); x_hat = float32(k) * q
  in fp32. INT8-SHARED: q = float32(m / 127) (one-time constant). INT8-TILE: q_t = float32(float64(max |x_t|) / 127),
  sent as 4 bytes before the codes; q_t = 0 gives all-zero codes and a zero decode.
"""

from __future__ import annotations

from typing import Any

import ucx_spec as S

N = S.TILE * S.TILE
LATENT_N = 16 * 4 * 4


def np():
    import numpy

    return numpy


# ---- normalization ----------------------------------------------------------------------------------------------
def scale_from_max(m: float) -> float:
    return float(m) / S.MAXABS_TARGET


# ---- U-QUARTER latent payload ------------------------------------------------------------------------------------
def latent_to_bytes(z) -> bytes:
    """One tile's latent [16, 4, 4] (or [1, 16, 4, 4]) fp32 as 1,024 little-endian bytes, row-major."""
    a = np().asarray(z, dtype="<f4").reshape(-1)
    if a.size != LATENT_N:
        raise ValueError(f"latent has {a.size} elements, not {LATENT_N}")
    return a.tobytes()


def latent_from_bytes(data: bytes):
    if len(data) != S.PAYLOAD_BYTES["U-QUARTER"]:
        raise ValueError(f"U-QUARTER payload has {len(data)} bytes")
    return np().frombuffer(data, dtype="<f4").reshape(1, *S.LATENT_SHAPE).astype("float32")


# ---- PCA-256 -----------------------------------------------------------------------------------------------------
def pca_fit(x_rows, components: int = S.PCA_COMPONENTS) -> dict[str, Any]:
    """The one registered PCA fit on a [M, 1024] matrix (float64 copy of fp32 values): float64 column mean, thin SVD of
    the centered matrix, the ``components`` (registered: 256) leading right singular vectors with the sign rule,
    numerical rank, all singular values. Returns float64 results and the fp32 decoder."""
    n = np()
    x = n.asarray(x_rows, dtype=n.float64)
    m_rows, n_cols = x.shape
    if components > min(m_rows, n_cols):
        raise ValueError(f"{components} components need at least that many rows and columns, got {x.shape}")
    mean = x.mean(axis=0)
    xc = x - mean
    _, sv, vt = n.linalg.svd(xc, full_matrices=False)
    basis = vt[:components].copy()
    flips = []
    for i in range(basis.shape[0]):
        j = int(n.argmax(n.abs(basis[i])))
        if basis[i, j] < 0:
            basis[i] = -basis[i]
            flips.append(i)
    tol = max(m_rows, n_cols) * n.finfo(n.float64).eps * float(sv[0])
    energy = float(n.square(sv).sum())
    kept = float(n.square(sv[:components]).sum())
    return {
        "mean64": mean,
        "basis64": basis,
        "mean32": mean.astype(n.float32),
        "basis32": basis.astype(n.float32),
        "singular_values": [float(v) for v in sv],
        "numerical_rank": int((sv > tol).sum()),
        "rank_tolerance": tol,
        "centered_energy_train": energy,
        "retained_energy_fraction_train": kept / energy if energy > 0 else None,
        "sign_flips": len(flips),
        "shape": [m_rows, n_cols],
        "orthonormality_max_abs_error_fp32": float(
            n.abs(
                basis.astype(n.float32).astype(n.float64) @ basis.astype(n.float32).astype(n.float64).T
                - n.eye(basis.shape[0])
            ).max()
        ),
    }


def pca_encode(x_rows, mean32, basis32):
    """[M, 1024] -> fp32 coefficients [M, 256]."""
    n = np()
    xc = n.asarray(x_rows, dtype=n.float64) - n.asarray(mean32, dtype=n.float64)
    return (xc @ n.asarray(basis32, dtype=n.float64).T).astype(n.float32)


def pca_decode(coef, mean32, basis32):
    """fp32 coefficients [M, 256] -> fp32 tiles [M, 1024]."""
    n = np()
    c = n.asarray(coef, dtype=n.float64)
    return (n.asarray(mean32, dtype=n.float64) + c @ n.asarray(basis32, dtype=n.float64)).astype(n.float32)


def pca_to_bytes(coef_row, components: int = S.PCA_COMPONENTS) -> bytes:
    a = np().asarray(coef_row, dtype="<f4").reshape(-1)
    if a.size != components:
        raise ValueError(f"PCA payload has {a.size} coefficients, not {components}")
    return a.tobytes()


def pca_from_bytes(data: bytes, components: int = S.PCA_COMPONENTS):
    if len(data) != 4 * components:
        raise ValueError(f"PCA payload has {len(data)} bytes, not {4 * components}")
    return np().frombuffer(data, dtype="<f4").astype("float32")


# ---- int8 --------------------------------------------------------------------------------------------------------
def int8_step(max_abs: float):
    """float32(max_abs / 127); max_abs is an exact fp32 maximum converted to float64."""
    return np().float32(float(max_abs) / S.INT8_MAX_CODE)


def int8_encode(x_row, q) -> tuple[Any, int]:
    """(int8 codes [1024], number of saturated elements |rint(x / q)| > 127). q = 0 gives zero codes."""
    n = np()
    x = n.asarray(x_row, dtype=n.float64).reshape(-1)
    qf = float(n.float32(q))
    if qf == 0.0:
        return n.zeros(x.size, dtype=n.int8), 0
    r = n.rint(x / qf)
    saturated = int((n.abs(r) > S.INT8_MAX_CODE).sum())
    return n.clip(r, -S.INT8_MAX_CODE, S.INT8_MAX_CODE).astype(n.int8), saturated


def int8_decode(codes, q):
    n = np()
    return n.asarray(codes, dtype=n.int8).astype(n.float32) * n.float32(q)


def int8_shared_to_bytes(codes) -> bytes:
    a = np().asarray(codes, dtype="i1").reshape(-1)
    if a.size != N:
        raise ValueError(f"int8 payload has {a.size} codes")
    return a.tobytes()


def int8_shared_from_bytes(data: bytes):
    if len(data) != S.PAYLOAD_BYTES["INT8-SHARED"]:
        raise ValueError(f"INT8-SHARED payload has {len(data)} bytes")
    return np().frombuffer(data, dtype="i1").copy()


def int8_tile_to_bytes(q, codes) -> bytes:
    n = np()
    return n.asarray([q], dtype="<f4").tobytes() + int8_shared_to_bytes(codes)


def int8_tile_from_bytes(data: bytes):
    n = np()
    if len(data) != S.PAYLOAD_BYTES["INT8-TILE"]:
        raise ValueError(f"INT8-TILE payload has {len(data)} bytes")
    return n.frombuffer(data[:4], dtype="<f4")[0], n.frombuffer(data[4:], dtype="i1").copy()


def tile_max_abs(x_row) -> float:
    """Exact max |x| of the fp32 tile, converted to float64 unchanged."""
    n = np()
    return float(n.abs(n.asarray(x_row, dtype=n.float32)).max())
