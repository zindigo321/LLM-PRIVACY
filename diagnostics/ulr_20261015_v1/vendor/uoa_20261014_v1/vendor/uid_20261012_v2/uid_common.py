"""Shared helpers of the UID-20261012-v2 diagnostic: hashing, records, input, metrics, criteria and models.

Self-contained: nothing here imports ``cg_fedllm``, so the diagnostic depends neither on uncommitted package code
nor on the SSBD runner. ``canonical_json`` and ``tensor_sha256`` follow ``cg_fedllm.utils.hashing``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROTOCOL_ID = "UID-20261012-v2"
DIAG_DIR = Path(__file__).resolve().parent
STATUSES = (
    "NOT_RUN",
    "BLOCKED",
    "COMPLETED_PASS",
    "COMPLETED_FIT_FAIL",
    "RESOURCE_STOP",
    "EXECUTION_ERROR",
    "INCOMPLETE",
)


class ProtocolError(RuntimeError):
    """A frozen-protocol precondition does not hold; nothing scientific may start."""


# ---- hashing and records ------------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def tensor_sha256(tensor: torch.Tensor, name: str = "") -> str:
    """SHA-256 over name, dtype, shape and raw bytes (the ``cg_fedllm.utils.hashing`` definition)."""
    t = tensor.detach().to("cpu").contiguous()
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(t.dtype).encode("utf-8"))
    h.update(repr(tuple(t.shape)).encode("utf-8"))
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def state_sha256(module: nn.Module) -> str:
    """SHA-256 over every state tensor (parameters and buffers) in sorted-key order."""
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(tensor_sha256(value, key).encode("ascii"))
    return h.hexdigest()


def json_safe(obj: Any) -> Any:
    """Non-finite floats become the strings "nan" / "inf" / "-inf" so records stay strict JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return "inf" if obj > 0 else ("-inf" if obj < 0 else "nan")
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def write_json_durable(path: str | Path, obj: Any) -> int:
    """Write pretty JSON through a temporary file, fsync, then rename; refuses to overwrite. Returns bytes."""
    path = Path(path)
    if path.exists():
        raise ProtocolError(f"record already exists, never overwritten: {path}")
    data = (json.dumps(json_safe(obj), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return len(data)


def load_config(path: str | Path) -> tuple[dict[str, Any], str, str]:
    """The parsed config, the SHA-256 of its raw bytes and its canonical-JSON SHA-256 (kept separate)."""
    raw = Path(path).read_bytes()
    cfg = json.loads(raw.decode("utf-8"))
    if cfg.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolError(f"config protocol_id {cfg.get('protocol_id')!r} != {PROTOCOL_ID}")
    return cfg, hashlib.sha256(raw).hexdigest(), canonical_sha256(cfg)


def verify_manifest(manifest_path: str | Path, root: str | Path = DIAG_DIR) -> dict[str, Any]:
    """Every ``verify`` entry of the frozen manifest must match the file on disk byte for byte."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    mismatches = []
    for entry in manifest["files"]:
        if not entry.get("verify", False):
            continue
        p = Path(root) / entry["path"]
        actual = sha256_file(p) if p.is_file() else None
        if actual != entry["sha256"]:
            mismatches.append({"path": entry["path"], "expected": entry["sha256"], "actual": actual})
    if mismatches:
        raise ProtocolError(f"approved files changed or missing: {mismatches}")
    return manifest


# ---- CPU execution settings ---------------------------------------------------------------------------
def configure_cpu(intra_op_threads: int, inter_op_threads: int) -> dict[str, Any]:
    """Fix thread counts and deterministic algorithms; call once at process start, before any tensor work."""
    torch.set_num_interop_threads(int(inter_op_threads))
    torch.set_num_threads(int(intra_op_threads))
    torch.use_deterministic_algorithms(True)
    return {
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }


# ---- input --------------------------------------------------------------------------------------------
def make_input(cfg: dict[str, Any]) -> torch.Tensor:
    """x = scale * rand([1, 1, H, W]) + offset from a dedicated CPU generator (never the global RNG)."""
    spec = cfg["input"]
    g = torch.Generator(device="cpu")
    g.manual_seed(int(spec["data_seed"]))
    shape = tuple(int(v) for v in spec["shape"])
    x = torch.rand(shape, generator=g, dtype=torch.float32)
    return x * float(spec["scale"]) + float(spec["offset"])


def describe_input(x: torch.Tensor) -> dict[str, Any]:
    x64 = x.detach().double()
    return {
        "sha256": tensor_sha256(x, "x"),
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "numel": x.numel(),
        "min": float(x64.min()),
        "max": float(x64.max()),
        "mean": float(x64.mean()),
        "energy_sum_sq": float(x64.square().sum()),
        "z_n": float(x64.square().sum()) / x.numel(),
    }


# ---- metrics (float64 reductions on CPU) --------------------------------------------------------------
def metrics(x_hat: torch.Tensor, x: torch.Tensor) -> dict[str, Any]:
    """Raw sums plus RSE, cosine and norm ratio. Cosine is undefined (None) when either norm is zero."""
    if x_hat.shape != x.shape:
        raise ValueError(f"shape mismatch {tuple(x_hat.shape)} vs {tuple(x.shape)}")
    a = x_hat.detach().to("cpu", torch.float64)
    b = x.detach().to("cpu", torch.float64)
    n = b.numel()
    sse = float((a - b).square().sum())
    target_energy = float(b.square().sum())
    pred_energy = float(a.square().sum())
    dot = float((a * b).sum())
    finite = bool(torch.isfinite(a).all()) and all(math.isfinite(v) for v in (sse, pred_energy, dot))
    rse = sse / target_energy if target_energy > 0 else None
    cosine = dot / math.sqrt(pred_energy * target_energy) if pred_energy > 0 and target_energy > 0 else None
    norm_ratio = math.sqrt(pred_energy / target_energy) if target_energy > 0 else None
    return {
        "n": n,
        "sse": sse,
        "target_energy": target_energy,
        "pred_energy": pred_energy,
        "dot": dot,
        "mse": sse / n,
        "z_n": target_energy / n,
        "q_n": (sse / n) / (target_energy / n) if target_energy > 0 else None,
        "rse": rse,
        "cosine": cosine,
        "norm_ratio": norm_ratio,
        "max_abs_err": float((a - b).abs().max()),
        "finite": finite,
    }


def passes(m: dict[str, Any], criteria: dict[str, Any]) -> bool:
    """Strict fit: finite output and metrics, RSE <= max_rse, cosine >= min_cosine; undefined metrics fail."""
    if not m["finite"] or m["rse"] is None or m["cosine"] is None:
        return False
    if not (math.isfinite(m["rse"]) and math.isfinite(m["cosine"])):
        return False
    return m["rse"] <= float(criteria["max_rse"]) and m["cosine"] >= float(criteria["min_cosine"])


def baselines(x: torch.Tensor) -> dict[str, Any]:
    """Zero prediction and one scalar mean shared by the whole tensor (not a per-coordinate mean)."""
    zero = torch.zeros_like(x)
    shared_mean = torch.full_like(x, float(x.detach().double().mean()))
    return {"zero_prediction": metrics(zero, x), "shared_scalar_mean": metrics(shared_mean, x)}


# ---- models -------------------------------------------------------------------------------------------
def build_model(cfg: dict[str, Any], arm: str) -> nn.Module:
    """Build an arm's model under ``torch.manual_seed(model_init_seed)`` inside a forked RNG, so model
    initialisation never shares state with the data generator."""
    seed = int(cfg["model_init_seed"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if arm == "U-CHECK":
            spec = cfg["u_check"]["model"]
            return nn.Conv2d(int(spec["in_channels"]), int(spec["out_channels"]), kernel_size=1, bias=True)
        if arm == "U-FIT":
            from uformer_ae import UformerAEConfig, UformerAutoEncoder

            return UformerAutoEncoder(UformerAEConfig(**cfg["u_fit"]["model"]))
    raise ValueError(f"unknown arm {arm!r}")


def reconstruct(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The output path of record: Decoder(Encoder(x)) for U-FIT, the 1x1 conv for U-CHECK."""
    if hasattr(model, "encode") and hasattr(model, "decode"):
        return model.decode(model.encode(x))
    return model(x)
