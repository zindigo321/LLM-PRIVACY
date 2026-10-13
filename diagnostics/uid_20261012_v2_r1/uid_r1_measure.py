"""UID-20261012-v2 r1: non-mutating measurements, gradient / parameter-change evidence and sampled RSS.

Every pure measurement runs on a deep copy of the training model, under the RNG fork of ``isolated``, so it changes
neither the training model's parameters or BatchNorm buffers nor the global RNG state; ``check_r1_measure.py`` verifies
this. Gradient statistics are read from the ``.grad`` tensors left by the training step's own backward: no extra pass.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import threading
import time
from typing import Any

import psutil
import torch
from torch import nn


@contextlib.contextmanager
def isolated(model: nn.Module, mode: str):
    """A deep copy of ``model`` in ``mode`` ("train"/"eval") with the CPU RNG state restored on exit."""
    with torch.random.fork_rng(devices=[]):
        twin = copy.deepcopy(model)
        twin.train(mode == "train")
        yield twin
    del twin


def state_fingerprint(module: nn.Module) -> dict[str, Any]:
    """Exact digest of every state tensor plus the global CPU RNG state (used by the isolation check)."""
    import hashlib

    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        t = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(t.dtype).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {
        "state_sha256": h.hexdigest(),
        "rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }


def groups(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Encoder/decoder sampling and core groups for the AE; one group for a model without encode/decode."""
    named = list(model.named_parameters())
    if not (hasattr(model, "encoder") and hasattr(model, "decoder")):
        return {"all": named}
    out: dict[str, list[tuple[str, nn.Parameter]]] = {
        "encoder.sampling": [],
        "encoder.core": [],
        "decoder.core": [],
        "decoder.sampling": [],
    }
    for name, p in named:
        side, part = name.split(".")[0], name.split(".")[1]
        out[f"{side}.{part}"].append((name, p))
    return out


def grad_stats(model: nn.Module) -> dict[str, Any]:
    """Per group: L2 norm of the existing gradients, how many parameters got a gradient, non-finite count."""
    res = {}
    for g, params in groups(model).items():
        sq, with_grad, nonfinite, numel = 0.0, 0, 0, 0
        for _, p in params:
            numel += p.numel()
            if p.grad is not None:
                gd = p.grad.detach().double()
                with_grad += p.numel()
                nonfinite += int((~torch.isfinite(gd)).sum())
                sq += float(gd.square().sum())
        res[g] = {
            "grad_l2": math.sqrt(sq),
            "numel": numel,
            "numel_with_grad": with_grad,
            "nonfinite": nonfinite,
        }
    return res


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def param_change(model: nn.Module, ref: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Per group: ||theta - theta_ref||_2, ||theta_ref||_2 and the number of elements that moved."""
    res = {}
    for g, params in groups(model).items():
        d2, r2, changed, numel = 0.0, 0.0, 0, 0
        for name, p in params:
            d = p.detach().double() - ref[name].double()
            d2 += float(d.square().sum())
            r2 += float(ref[name].double().square().sum())
            changed += int((d != 0).sum())
            numel += p.numel()
        res[g] = {
            "delta_l2": math.sqrt(d2),
            "ref_l2": math.sqrt(r2),
            "numel_changed": changed,
            "numel": numel,
        }
    return res


def trainable_coverage(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    in_opt = {id(p) for grp in optimizer.param_groups for p in grp["params"]}
    params = list(model.parameters())
    return {
        "parameters": sum(p.numel() for p in params),
        "requires_grad": sum(p.numel() for p in params if p.requires_grad),
        "in_optimizer": sum(p.numel() for p in params if id(p) in in_opt),
        "tensors_missing_from_optimizer": sum(1 for p in params if id(p) not in in_opt),
    }


class RSSSampler:
    """Polls the RSS of this process every ``interval_s``. The maximum is a sampled peak: a spike shorter than the
    interval can be missed. ``limit`` is enforced by the supervisor (which kills the child); this only reports."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = float(interval_s)
        self.proc = psutil.Process(os.getpid())
        self.samples = 0
        self.max_rss = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="rss-sampler", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.poll()

    def poll(self) -> int:
        rss = self.proc.memory_info().rss
        self.samples += 1
        self.max_rss = max(self.max_rss, rss)
        return rss

    def start(self) -> RSSSampler:
        self._t.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._t.join(timeout=5)
        self.poll()
        return {
            "sampled_max_rss_bytes": self.max_rss,
            "samples": self.samples,
            "interval_s": self.interval_s,
            "label": "sampled maximum of the working set; spikes shorter than the interval can be missed",
            # Windows keeps its own peak working-set counter; reported separately, None where unavailable
            "os_peak_wset_bytes": getattr(self.proc.memory_info(), "peak_wset", None),
        }


def monotonic() -> float:
    return time.monotonic()
