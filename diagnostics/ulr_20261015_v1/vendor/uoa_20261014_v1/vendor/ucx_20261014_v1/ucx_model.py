"""UCX-20261014-v1 model helpers (torch imported lazily): the registered Uformer through the frozen builder, the
sampling schedule, normalization arithmetic, the one training update and the state files.

The model is built only by the frozen ``uid_common.build_model(cfg, "U-FIT")`` (the builder's model key) from the
registered configuration, under the registered seed in a forked RNG. ``train_update`` is shared unchanged by the
training child and the resource pilot, so the pilot measures the registered update.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import ucx_spec as S

FROZEN_DIR = Path(__file__).resolve().parent.parent / "uid_20261012_v2"


def build_cfg(model: dict | None = None, seed: int | None = None) -> dict[str, Any]:
    return {
        "model_init_seed": S.MODEL_INIT_SEED if seed is None else seed,
        "u_fit": {"model": model or S.MODEL},
    }


def build_model(cfg: dict[str, Any]):
    import uid_common as C

    if Path(C.__file__).resolve().parent != FROZEN_DIR.resolve():
        raise RuntimeError(f"uid_common imported from {C.__file__}")
    return C.build_model(cfg, "U-FIT")


def schedule(n_tiles: int, updates: int, batch: int, seed: int) -> list[list[int]]:
    """Rule section 6: epoch permutations from one CPU generator seeded ``seed``, drawn in order and concatenated;
    update k takes positions batch (k - 1) .. batch k - 1."""
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    need = updates * batch
    flat: list[int] = []
    while len(flat) < need:
        flat += torch.randperm(n_tiles, generator=g).tolist()
    return [flat[i * batch : (i + 1) * batch] for i in range(updates)]


def schedule_sha256(sched: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(sched, separators=(",", ":")).encode("ascii")).hexdigest()


def normalize(x, scale: float):
    """Production ``Normalizer.normalize`` arithmetic: x / float32(s) in fp32."""
    import torch

    return x / torch.tensor(float(scale), dtype=torch.float32, device=x.device)


def denormalize(y, scale: float):
    import torch

    return y * torch.tensor(float(scale), dtype=torch.float32, device=y.device)


def train_update(model, opt, x_n, F):
    """The forward and backward of one optimizer update: mean normalized-space MSE over the batch, train mode. The
    caller then measures (at curve points) and calls ``opt.step()``."""
    import uid_common as C

    opt.zero_grad(set_to_none=True)
    y_n = C.reconstruct(model, x_n)
    loss = F.mse_loss(y_n, x_n)
    loss.backward()
    return loss, y_n


def train_update_source_sha256() -> str:
    import inspect

    return hashlib.sha256(inspect.getsource(train_update).encode("utf-8")).hexdigest()


def save_state(path: Path, model, meta: dict[str, Any]) -> dict[str, Any]:
    """All parameters and buffers as safetensors, created exclusively and fsynced; read back bit for bit."""
    import torch
    import ucx_common as K
    import uid_common as C
    from safetensors.torch import load, save

    state = {k: v.detach().to("cpu").contiguous().clone() for k, v in model.state_dict().items()}
    digest = C.state_sha256(model)
    meta = {
        "format": "ucx_20261014_v1.state/v1",
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
        raise RuntimeError(f"state file {path} does not read back bit for bit")
    return {
        "path": str(path),
        "bytes": len(data),
        "file_sha256": file_sha,
        "state_sha256": digest,
        "tensors": len(state),
        "parameter_elements": sum(p.numel() for p in model.parameters()),
        "buffer_elements": sum(b.numel() for b in model.buffers()),
    }


def load_state(path: Path, cfg: dict[str, Any], file_sha256: str, state_sha256: str):
    """A fresh model with the saved state loaded strictly, after the file bytes match the recorded hash."""
    import ucx_common as K
    import uid_common as C
    from safetensors.torch import load

    data = Path(path).read_bytes()
    if K.sha256_bytes(data) != file_sha256:
        raise RuntimeError(f"{path}: file SHA-256 differs from the training record")
    model = build_model(cfg)
    model.load_state_dict(load(data), strict=True)
    if C.state_sha256(model) != state_sha256:
        raise RuntimeError(f"{path}: loaded state hash differs from the training record")
    model.eval()
    return model
