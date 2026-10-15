"""ULR-20261015-v1 model helpers (torch imported lazily): the bridged Uformer, the shape-aware wire format, the
standalone receiver, and the sender / receiver payload paths.

- ``BridgedUformer(base, c)``: ``encode(x) = reduce_c(base.encode(x))`` gives the wire latent [B, c, 4, 4];
  ``decode(z) = base.decode(expand_c(z))``. ``reduce_c`` and ``expand_c`` are 1x1 convolutions with bias. Nothing
  else is added: no activation, normalization, residual, bypass or extra branch between them. The training forward
  (``train_update``) and every evaluation pass through both.
- **Wire format.** One tile's wire latent is 16 c fp32 values in the row-major order of [c, 4, 4], as little-endian
  bytes (64 c B).
  - A payload of any other length is malformed and raises (the step fails).
  - Non-finite values are transmitted unchanged; they make that tile's prediction non-finite, which the policy cannot
    pass.
- ``Receiver(c)``: the decoder side only, built standalone, is ``expand_c`` plus the frozen ``UformerDecoder`` with
  its six sampling LeakyReLUs. It is loaded strictly from the ``expand.*`` and ``base.decoder.*`` tensors of a saved
  state. It holds no encoder or reduce tensor, and it sees only parsed payload bytes and the shared configuration.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import ulr_common as K
import ulr_spec as S

K.setup_paths()
import uoa_model as UM  # noqa: E402  (vendored UOA model helpers, byte-identical)

normalize = UM.normalize
denormalize = UM.denormalize
schedule = UM.schedule
schedule_sha256 = UM.schedule_sha256
train_update = UM.train_update
train_update_source_sha256 = UM.train_update_source_sha256
fingerprint = UM.fingerprint
rng_sha256 = UM.rng_sha256
DECODER_LEAKY_INDICES = (2, 5, 8, 11, 14, 17)


def _nn():
    from torch import nn

    return nn


def bridge_layers(c: int):
    """The two 1x1 convolutions, created inside a forked RNG so their default initialization consumes no global RNG
    (their values are always overwritten by the persisted channel-PCA initialization)."""
    import torch

    nn = _nn()
    with torch.random.fork_rng(devices=[]):
        return nn.Conv2d(16, c, kernel_size=1, bias=True), nn.Conv2d(c, 16, kernel_size=1, bias=True)


_CLS: dict[str, Any] = {}


def make_bridged_class():
    if "bridged" in _CLS:
        return _CLS["bridged"]
    nn = _nn()

    class BridgedUformer(nn.Module):
        def __init__(self, base, c: int) -> None:
            super().__init__()
            self.base = base
            self.c = int(c)
            self.cfg = base.cfg
            self.reduce, self.expand = bridge_layers(self.c)

        def encode(self, x):
            return self.reduce(self.base.encode(x))

        def decode(self, z):
            if tuple(z.shape[1:]) != S.wire_shape(self.c):
                raise ValueError(f"wire latent {tuple(z.shape)} is not [B, {self.c}, 4, 4]")
            return self.base.decode(self.expand(z))

        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z

    _CLS["bridged"] = BridgedUformer
    return BridgedUformer


def bridged(base, c: int):
    return make_bridged_class()(base, c)


def set_bridge(model, init: dict, c: int) -> None:
    """Copy the persisted fp32 channel-PCA initialization of width c into the bridge (no other tensor changes)."""
    import torch

    with torch.no_grad():
        model.reduce.weight.copy_(torch.from_numpy(init[f"reduce_weight_c{c}"]))
        model.reduce.bias.copy_(torch.from_numpy(init[f"reduce_bias_c{c}"]))
        model.expand.weight.copy_(torch.from_numpy(init[f"expand_weight_c{c}"]))
        model.expand.bias.copy_(torch.from_numpy(init[f"expand_bias_c{c}"]))


def bridge_equals(model, init: dict, c: int) -> bool:
    import numpy as np

    pairs = (
        (model.reduce.weight, f"reduce_weight_c{c}"),
        (model.reduce.bias, f"reduce_bias_c{c}"),
        (model.expand.weight, f"expand_weight_c{c}"),
        (model.expand.bias, f"expand_bias_c{c}"),
    )
    return all(np.array_equal(t.detach().numpy(), init[k]) and t.dtype.is_floating_point for t, k in pairs)


def load_source(src: dict, cfg: dict):
    """The UOA U-LEAKY-LONG final state in its own architecture, eval mode, after both hashes match."""
    import uoa_common as UC

    try:
        return UM.load_state(
            Path(src["path"]),
            cfg,
            S.MODEL_INIT_SEED,
            S.SOURCE_ACTIVATION,
            src["file_sha256"],
            src["state_sha256"],
        )
    except UC.UOAError as exc:
        raise K.ULRError(str(exc)) from exc


def build_arm(src: dict, init: dict, c: int, cfg: dict):
    """An arm at step zero: the verified source backbone (every parameter and buffer) plus the PCA bridge of width c.
    No initializer runs on the loaded backbone; the checks refuse any difference."""
    base = load_source(src, cfg)
    model = bridged(base, c)
    set_bridge(model, init, c)
    if backbone_sha256(model) != src["state_sha256"]:
        raise K.ULRError("the bridged model's backbone differs from the source state")
    if not bridge_equals(model, init, c):
        raise K.ULRError("the bridge differs from the persisted channel-PCA initialization")
    return model


def backbone_sha256(model) -> str:
    return UM.state_sha256(model.base)


def state_sha256(model) -> str:
    return UM.state_sha256(model)


def save_state(path: Path, model, meta: dict[str, Any]) -> dict[str, Any]:
    """Every parameter and buffer (backbone and bridge), created exclusively; read back bit for bit."""
    import torch
    from safetensors.torch import load, save

    state = {k: v.detach().to("cpu").contiguous().clone() for k, v in model.state_dict().items()}
    meta = {
        "format": "ulr_20261015_v1.state/v1",
        "state_sha256": state_sha256(model),
        "backbone_sha256": backbone_sha256(model),
        "model_config": json.dumps(model.cfg.to_dict(), sort_keys=True),
        "backbone_internal_shape": json.dumps(list(S.BACKBONE_INTERNAL_SHAPE)),
        "wire_latent_shape": json.dumps(list(S.wire_shape(model.c))),
        "c": str(model.c),
        **{k: v if isinstance(v, str) else json.dumps(v, sort_keys=True) for k, v in meta.items()},
    }
    data = save(state, metadata=meta)
    sha = K.write_exclusive(path, data)
    back = load(Path(path).read_bytes())
    if set(back) != set(state) or not all(
        back[k].dtype == state[k].dtype and torch.equal(back[k], state[k]) for k in state
    ):
        raise K.ULRError(f"state file {path} does not read back bit for bit")
    return {
        "path": str(path),
        "bytes": len(data),
        "file_sha256": sha,
        "state_sha256": meta["state_sha256"],
        "backbone_sha256": meta["backbone_sha256"],
        "c": model.c,
    }


def load_arm_state(f: dict, c: int, cfg: dict):
    """A saved arm state (full sender + receiver model), eval mode, after both hashes match."""
    from safetensors.torch import load

    data = Path(f["path"]).read_bytes()
    if K.sha256_bytes(data) != f["file_sha256"]:
        raise K.ULRError(f"{f['path']}: file SHA-256 differs from its record")
    model = bridged(UM.build(cfg, S.MODEL_INIT_SEED, S.SOURCE_ACTIVATION), c)
    model.load_state_dict(load(data), strict=True)
    if state_sha256(model) != f["state_sha256"]:
        raise K.ULRError(f"{f['path']}: loaded state hash differs from its record")
    UM.check_activations(model.base, S.SOURCE_ACTIVATION, cfg)
    model.eval()
    return model


# ---- wire format -----------------------------------------------------------------------------------------------
def wire_to_bytes(z, c: int) -> bytes:
    import numpy as np

    a = np.asarray(z, dtype=np.float32)
    if tuple(a.shape) not in ((1, *S.wire_shape(c)), S.wire_shape(c)):
        raise ValueError(f"wire latent of shape {a.shape} is not [{c}, 4, 4]")
    out = a.astype("<f4").reshape(-1).tobytes()
    if len(out) != S.wire_bytes(c):
        raise ValueError(f"wire payload has {len(out)} bytes, not {S.wire_bytes(c)}")
    return out


def wire_from_bytes(data: bytes, c: int):
    import numpy as np

    if not isinstance(data, (bytes, bytearray)) or len(data) != S.wire_bytes(c):
        raise ValueError(
            f"malformed wire payload: {len(data) if hasattr(data, '__len__') else data!r} bytes, expected {S.wire_bytes(c)}"
        )
    return np.frombuffer(bytes(data), dtype="<f4").reshape(1, *S.wire_shape(c)).astype(np.float32)


# ---- the standalone receiver -------------------------------------------------------------------------------------
def make_receiver(c: int, cfg: dict):
    import torch
    import uformer_ae as A  # vendored frozen Uformer

    nn = _nn()

    class Receiver(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c = c
            with torch.random.fork_rng(devices=[]):
                self.expand = nn.Conv2d(c, 16, kernel_size=1, bias=True)
                self.decoder = A.UformerDecoder(A.UformerAEConfig(**cfg))
            for i in DECODER_LEAKY_INDICES:
                if type(self.decoder.sampling[i]) is not nn.ReLU:
                    raise K.ULRError(f"receiver decoder.sampling.{i} is not the expected ReLU slot")
                self.decoder.sampling[i] = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        def forward(self, z):
            if tuple(z.shape[1:]) != S.wire_shape(self.c):
                raise ValueError(f"wire latent {tuple(z.shape)} is not [B, {self.c}, 4, 4]")
            return self.decoder(self.expand(z))

    return Receiver()


def receiver_from_state(f: dict, c: int, cfg: dict):
    """The receiver loaded strictly from ONLY the decoder-side tensors of a saved state (file hash checked)."""
    from safetensors.torch import load

    data = Path(f["path"]).read_bytes()
    if K.sha256_bytes(data) != f["file_sha256"]:
        raise K.ULRError(f"{f['path']}: file SHA-256 differs from its record")
    full = load(data)
    keep = {}
    for k, v in full.items():
        if k.startswith("expand."):
            keep[k] = v
        elif k.startswith("base.decoder."):
            keep["decoder." + k[len("base.decoder.") :]] = v
    rec = make_receiver(c, cfg)
    rec.load_state_dict(keep, strict=True)
    rec.eval()
    kinds = {k.split(".")[0] for k in keep}
    if kinds != {"expand", "decoder"}:
        raise K.ULRError(f"receiver tensors {kinds} are not exactly expand + decoder")
    return rec, {
        "tensors": len(keep),
        "elements": int(sum(v.numel() for v in keep.values())),
        "bytes": int(sum(v.numel() * v.element_size() for v in keep.values())),
    }


def send(model, x, scale: float, c: int) -> dict[str, Any]:
    """Sender, every tile alone, eval mode: x -> x_n -> encoder -> reduce -> wire bytes (64 c B)."""
    import torch

    if model.training:
        raise K.ULRError("send needs a model in eval mode")
    pays, zs = [], []
    for i in range(x.shape[0]):
        z = model.encode(normalize(x[i : i + 1], scale))
        b = wire_to_bytes(z.numpy(), c)
        pays.append(b)
        zs.append(torch.from_numpy(wire_from_bytes(b, c)))
    return {"payloads": pays, "z": torch.cat(zs)}


def receive(receiver, payloads: list[bytes], c: int, scale: float):
    """Receiver, every payload alone: parse -> expand -> decoder -> y_n -> y. Sees nothing but the bytes."""
    import torch

    if receiver.training:
        raise K.ULRError("receive needs a receiver in eval mode")
    yns = [receiver(torch.from_numpy(wire_from_bytes(b, c))) for b in payloads]
    y_n = torch.cat(yns)
    return y_n, denormalize(y_n, scale)


def roundtrip_copy(model, x, scale: float, c: int):
    """In-training evaluation path on an eval-mode deep copy (the copy's own decoder; bitwise the receiver's)."""
    import torch

    with torch.random.fork_rng(devices=[]):
        twin = copy.deepcopy(model)
        twin.eval()
        with torch.no_grad():
            s = send(twin, x, scale, c)
            yns = [twin.decode(torch.from_numpy(wire_from_bytes(b, c))) for b in s["payloads"]]
    y_n = torch.cat(yns)
    return {
        "z": s["z"],
        "y_n": y_n,
        "y": denormalize(y_n, scale),
        "payload_bytes": sorted({len(b) for b in s["payloads"]}),
    }


# ---- training measurements ---------------------------------------------------------------------------------------
GROUPS = (
    "base.encoder.sampling",
    "base.encoder.core",
    "reduce",
    "expand",
    "base.decoder.core",
    "base.decoder.sampling",
)


def group_of(name: str) -> str:
    for g in GROUPS:
        if name.startswith(g + "."):
            return g
    raise K.ULRError(f"parameter {name} belongs to no registered group")


def grad_stats(model) -> dict[str, Any]:
    res = {g: {"grad_l2_sq": 0.0, "numel": 0, "numel_with_grad": 0, "nonfinite": 0} for g in GROUPS}
    for name, p in model.named_parameters():
        r = res[group_of(name)]
        r["numel"] += p.numel()
        if p.grad is not None:
            gd = p.grad.detach().double()
            r["numel_with_grad"] += p.numel()
            r["nonfinite"] += int((~gd.isfinite()).sum())
            r["grad_l2_sq"] += float(gd.square().sum())
    return {g: {"grad_l2": math.sqrt(r.pop("grad_l2_sq")), **r} for g, r in res.items()}


def snapshot(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def param_change(model, ref: dict) -> dict[str, Any]:
    res = {g: {"d2": 0.0, "r2": 0.0, "changed": 0, "numel": 0} for g in GROUPS}
    for name, p in model.named_parameters():
        r = res[group_of(name)]
        d = p.detach().double() - ref[name].double()
        r["d2"] += float(d.square().sum())
        r["r2"] += float(ref[name].double().square().sum())
        r["changed"] += int((d != 0).sum())
        r["numel"] += p.numel()
    return {
        g: {
            "delta_l2": math.sqrt(r["d2"]),
            "ref_l2": math.sqrt(r["r2"]),
            "numel_changed": r["changed"],
            "numel": r["numel"],
        }
        for g, r in res.items()
    }


def bridge_stats(model, z) -> dict[str, Any]:
    """Composed bridge map M = W_expand W_reduce (16 x 16) singular values, and per-channel wire-latent statistics."""
    import numpy as np

    we = model.expand.weight.detach().double().reshape(16, model.c).numpy()
    wr = model.reduce.weight.detach().double().reshape(model.c, 16).numpy()
    sv = np.linalg.svd(we @ wr, compute_uv=False)
    zz = z.double().numpy().transpose(1, 0, 2, 3).reshape(model.c, -1)
    return {
        "composed_singular_values": [float(v) for v in sv],
        "composed_rank_1e-6": int((sv > 1e-6 * sv[0]).sum()) if sv[0] > 0 else 0,
        "wire_channel_mean": [float(v) for v in zz.mean(axis=1)],
        "wire_channel_std": [float(v) for v in zz.std(axis=1)],
        "wire_max_abs": float(np.abs(zz).max()),
    }
