"""UOA-20261014-v1 model helpers (torch imported lazily): the registered UCX Uformer through the vendored frozen
builder, the one registered activation ablation, state files, the neural payload paths and activation statistics.

- ``build`` makes the registered model with ``ucx_model.build_model`` (seed 1 in a forked RNG). For the LeakyReLU arm
  it then replaces exactly the 13 sampling-path ``nn.ReLU`` modules (``uoa_spec.RELU_PATHS``) with
  ``nn.LeakyReLU(0.1, inplace=False)``. Neither module has a parameter or buffer, so both arms start from the same
  tensors. ``check_activations`` verifies the scope after every build and load.
- ``train_update``, ``schedule``, ``normalize`` and ``denormalize`` are the vendored UCX functions, unchanged.
- ``float_outputs`` and ``int8_outputs`` are the eval-mode payload paths. Every tile is processed alone, and the
  decoder receives only the parsed payload bytes (dequantized for INT8) and the persisted scales.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import uoa_common as K
import uoa_spec as S

K.setup_paths()
import ucx_model as Mo  # noqa: E402  (vendored, byte-identical)

schedule = Mo.schedule
schedule_sha256 = Mo.schedule_sha256
normalize = Mo.normalize
denormalize = Mo.denormalize
train_update = Mo.train_update
train_update_source_sha256 = Mo.train_update_source_sha256


def expected_counts(model_cfg: dict) -> dict[str, int]:
    blocks = 2 * int(model_cfg["core_layers"]) * int(model_cfg["core_depth"])
    return {"GELU": 2 * blocks, "LayerNorm": 2 * blocks, "Tanh": 1, "BatchNorm2d": 13, "activations": 13}


def build(model_cfg: dict, seed: int, activation: str):
    from torch import nn

    model = Mo.build_model(Mo.build_cfg(model_cfg, seed))
    if activation == "leaky_relu":
        for path in S.RELU_PATHS:
            side, part, idx = path.split(".")
            seq = getattr(getattr(model, side), part)
            if type(seq[int(idx)]) is not nn.ReLU:
                raise K.UOAError(f"{path} is {type(seq[int(idx)]).__name__}, not ReLU")
            seq[int(idx)] = nn.LeakyReLU(negative_slope=S.LEAKY_SLOPE, inplace=False)
    elif activation != "relu":
        raise K.UOAError(f"unknown activation {activation!r}")
    check_activations(model, activation, model_cfg)
    return model


def activation_inventory(model) -> list[dict[str, Any]]:
    from torch import nn

    out = []
    for name, m in model.named_modules():
        if type(m) in (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Tanh, nn.BatchNorm2d, nn.LayerNorm):
            row = {"path": name, "class": type(m).__name__}
            if type(m) is nn.LeakyReLU:
                row["negative_slope"] = m.negative_slope
            if hasattr(m, "inplace"):
                row["inplace"] = m.inplace
            out.append(row)
    return out


def check_activations(model, activation: str, model_cfg: dict) -> dict[str, Any]:
    inv = activation_inventory(model)
    want = expected_counts(model_cfg)
    relu = [r["path"] for r in inv if r["class"] == "ReLU"]
    leaky = [r for r in inv if r["class"] == "LeakyReLU"]
    counts = {c: sum(1 for r in inv if r["class"] == c) for c in ("GELU", "LayerNorm", "Tanh", "BatchNorm2d")}
    ok_other = all(counts[c] == want[c] for c in counts)
    if activation == "relu":
        ok = tuple(relu) == S.RELU_PATHS and not leaky
    else:
        ok = (
            not relu
            and tuple(r["path"] for r in leaky) == S.RELU_PATHS
            and all(r["negative_slope"] == S.LEAKY_SLOPE and r["inplace"] is False for r in leaky)
        )
    if not (ok and ok_other):
        raise K.UOAError(
            f"activation scope check failed for {activation}: relu {relu}, leaky {leaky}, {counts}"
        )
    return {"activation": activation, "replaced_paths": list(S.RELU_PATHS), "counts": counts, "ok": True}


def state_sha256(model) -> str:
    import uid_common as C  # vendored

    return C.state_sha256(model)


def save_state(path: Path, model, meta: dict[str, Any]) -> dict[str, Any]:
    """All parameters and buffers as safetensors, created exclusively and fsynced; read back bit for bit."""
    import torch
    from safetensors.torch import load, save

    state = {k: v.detach().to("cpu").contiguous().clone() for k, v in model.state_dict().items()}
    digest = state_sha256(model)
    meta = {
        "format": "uoa_20261014_v1.state/v1",
        "state_sha256": digest,
        "model_config": json.dumps(model.cfg.to_dict(), sort_keys=True),
        **{k: v if isinstance(v, str) else json.dumps(v, sort_keys=True) for k, v in meta.items()},
    }
    data = save(state, metadata=meta)
    file_sha = K.write_exclusive(path, data)
    back = load(Path(path).read_bytes())
    if set(back) != set(state) or not all(
        back[k].dtype == state[k].dtype and torch.equal(back[k], state[k]) for k in state
    ):
        raise K.UOAError(f"state file {path} does not read back bit for bit")
    return {"path": str(path), "bytes": len(data), "file_sha256": file_sha, "state_sha256": digest}


def load_state(path: Path, model_cfg: dict, seed: int, activation: str, file_sha256: str, state_sha256_: str):
    """A fresh model of the arm's architecture with the saved state loaded strictly, after the file bytes and the
    loaded state match their recorded hashes; returned in eval mode."""
    from safetensors.torch import load

    data = Path(path).read_bytes()
    if K.sha256_bytes(data) != file_sha256:
        raise K.UOAError(f"{path}: file SHA-256 differs from its record")
    model = build(model_cfg, seed, activation)
    model.load_state_dict(load(data), strict=True)
    if state_sha256(model) != state_sha256_:
        raise K.UOAError(f"{path}: loaded state hash differs from its record")
    check_activations(model, activation, model_cfg)
    model.eval()
    return model


def fingerprint(model) -> dict[str, str]:
    import uid_r1_measure as M  # vendored

    return M.state_fingerprint(model)


def rng_sha256() -> str:
    import torch

    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


# ---- payload paths (eval mode, gradients disabled by the caller) ---------------------------------------------------
def float_outputs(model, x, scale: float) -> dict[str, Any]:
    """Every tile alone: x -> x_n -> z -> 1,024 fp32 LE bytes -> parse -> y_n -> y (vendored UCX arithmetic)."""
    import torch
    import ucx_codecs as X  # vendored

    if model.training:
        raise K.UOAError("float_outputs needs a model in eval mode")
    pays, zs, yns, ys = [], [], [], []
    for i in range(x.shape[0]):
        x_n = normalize(x[i : i + 1], scale)
        z = model.encode(x_n)
        b = X.latent_to_bytes(z.numpy())
        z_back = torch.from_numpy(X.latent_from_bytes(b))
        if not torch.equal(z_back, z):
            raise K.UOAError("the FLOAT latent payload does not round-trip bit for bit")
        y_n = model.decode(z_back)
        pays.append(b)
        zs.append(z_back)
        yns.append(y_n)
        ys.append(denormalize(y_n, scale))
    return {"payloads": pays, "z": torch.cat(zs), "y_n": torch.cat(yns), "y": torch.cat(ys)}


def int8_outputs(model, z_all, scales32, scale: float) -> dict[str, Any]:
    """Every tile alone: FLOAT latent -> int8 codes with the persisted scales -> 256 bytes -> parse -> dequantize ->
    decoder -> y_n -> y. Also returns per-tile saturation counts per channel."""
    import numpy as np
    import torch
    import uoa_codecs as Q

    if model.training:
        raise K.UOAError("int8_outputs needs a model in eval mode")
    pays, yns, ys, sats, zeros, zq = [], [], [], [], [], []
    for i in range(z_all.shape[0]):
        codes, sat, zc = Q.latent_int8_codes(z_all[i].numpy(), scales32)
        b = Q.latent_int8_to_bytes(codes)
        zh = torch.from_numpy(Q.latent_int8_dequant(Q.latent_int8_from_bytes(b), scales32))
        y_n = model.decode(zh)
        pays.append(b)
        sats.append(sat)
        zeros.append(zc)
        zq.append(zh)
        yns.append(y_n)
        ys.append(denormalize(y_n, scale))
    return {
        "payloads": pays,
        "z_hat": torch.cat(zq),
        "y_n": torch.cat(yns),
        "y": torch.cat(ys),
        "saturated": np.stack(sats),
        "zero_scale_nonzero": np.stack(zeros),
    }


def decode_float_payloads(model, payloads: list[bytes], scale: float):
    """Decoder only, from FLOAT payload bytes (used for the wrong-payload control)."""
    import torch
    import ucx_codecs as X

    yns = [model.decode(torch.from_numpy(X.latent_from_bytes(b))) for b in payloads]
    y_n = torch.cat(yns)
    return y_n, denormalize(y_n, scale)


def decode_int8_payloads(model, payloads: list[bytes], scales32, scale: float):
    import torch
    import uoa_codecs as Q

    yns = [
        model.decode(torch.from_numpy(Q.latent_int8_dequant(Q.latent_int8_from_bytes(b), scales32)))
        for b in payloads
    ]
    y_n = torch.cat(yns)
    return y_n, denormalize(y_n, scale)


# ---- activation statistics (descriptive; on the caller's model, which must be a disposable copy) -----------------
class ActStats:
    """Forward hooks that accumulate per-module output statistics in float64 (count, sum, sum of squares, min, max,
    exact zeros, negatives) and, for the Tanh head, the pre-activation magnitude and saturation |tanh(u)| >= 0.99."""

    def __init__(self, model) -> None:
        from torch import nn

        self.rows: dict[str, dict[str, Any]] = {}
        self.handles = []
        for name, m in model.named_modules():
            if ".sampling." in f"{name}." and name.count(".") == 2 and not isinstance(m, nn.ReflectionPad2d):
                self.handles.append(m.register_forward_hook(self._hook(name, type(m).__name__)))
        self.handles.append(model.encoder.register_forward_hook(self._hook("latent", "encoder")))
        self.handles.append(
            model.decoder.core[-1].register_forward_hook(self._hook("decoder.core.out", "core"))
        )

    def _hook(self, name: str, cls: str):
        def hook(mod, inp, out):
            import torch

            o = out.detach().double()
            r = self.rows.setdefault(
                name,
                {
                    "class": cls,
                    "n": 0,
                    "sum": 0.0,
                    "sumsq": 0.0,
                    "min": math.inf,
                    "max": -math.inf,
                    "zeros": 0,
                    "negatives": 0,
                    "shape": list(out.shape[1:]),
                },
            )
            r["n"] += o.numel()
            r["sum"] += float(o.sum())
            r["sumsq"] += float(o.square().sum())
            r["min"] = min(r["min"], float(o.min()))
            r["max"] = max(r["max"], float(o.max()))
            r["zeros"] += int((o == 0).sum())
            r["negatives"] += int((o < 0).sum())
            if cls == "Tanh":
                u = inp[0].detach().double()
                r["pre_max_abs"] = max(r.get("pre_max_abs", 0.0), float(u.abs().max()))
                r["saturated_0p99"] = r.get("saturated_0p99", 0) + int((torch.tanh(u).abs() >= 0.99).sum())

        return hook

    def close(self) -> dict[str, Any]:
        for h in self.handles:
            h.remove()
        out = {}
        for name, r in self.rows.items():
            n = r["n"]
            mean = r["sum"] / n
            row = {
                "class": r["class"],
                "shape": r["shape"],
                "elements": n,
                "mean": mean,
                "std": math.sqrt(max(r["sumsq"] / n - mean * mean, 0.0)),
                "min": r["min"],
                "max": r["max"],
                "zero_fraction": r["zeros"] / n,
                "negative_fraction": r["negatives"] / n,
            }
            if "pre_max_abs" in r:
                row["tanh_pre_max_abs"] = r["pre_max_abs"]
                row["tanh_saturated_0p99_fraction"] = r["saturated_0p99"] / n
            out[name] = row
        return out


def eval_copy_stats(model, x_n_probe, batch: int = 160) -> dict[str, Any]:
    """Activation statistics of an eval-mode deep copy over the probe tiles (batched; descriptive only). The source
    model, its buffers and the global RNG are left unchanged (checked)."""
    import torch

    before, rng0 = fingerprint(model)["state_sha256"], rng_sha256()
    with torch.random.fork_rng(devices=[]):
        twin = copy.deepcopy(model)
        twin.eval()
        st = ActStats(twin)
        with torch.no_grad():
            for i in range(0, x_n_probe.shape[0], batch):
                twin.decode(twin.encode(x_n_probe[i : i + batch]))
        out = st.close()
    del twin
    if fingerprint(model)["state_sha256"] != before or rng_sha256() != rng0:
        raise K.UOAError("activation statistics changed the source model or the RNG")
    return out
