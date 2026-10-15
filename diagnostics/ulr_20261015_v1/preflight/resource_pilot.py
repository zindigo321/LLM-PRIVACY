"""ULR-20261015-v1 resource pilot child: the full-size registered update of ONE arm's bridged model on SYNTHETIC data.

``python -B preflight/resource_pilot.py <arm> <mode> <out_json>``, started only by ``run_resource_pilot.py``. ``arm``
is one of C16-CONTROL / C4-REDUCED / C1-REDUCED; ``mode`` is ``full`` (the one authorized pilot of that arm) or
``fixture`` (a small engineering model; never evidence about the budget).

**Same as the training child:**
- the registered architecture (LeakyReLU sampling paths) plus the bridge of the arm's width;
- batch 8, CPU fp32, 24 / 24 threads, deterministic algorithms;
- the vendored update (mean normalized-space MSE, train mode, through the bottleneck) and Adam lr 2e-4 with its
  states allocated.

**Different on purpose:** a fresh model under seed 20261016 (not the UOA source), a random orthonormal bridge, and
synthetic data. Nothing here can seed or inform the scientific run.

**Measured:** 5 warm-up and 20 measured updates. Then, reported only:
- three curve-point measurements;
- two state saves into a deleted temporary directory;
- one checkpoint TRAIN evaluation of 1,440 synthetic tiles on a deep copy;
- the per-tile sender and standalone-receiver costs.

The model, optimizer and states are discarded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
PILOT_SEED = 20261016
WARMUP, MEASURED = 5, 20
INTERNAL_BUDGET_S = 100.0
FIXTURE_MODEL = {"down_channels": [2, 2, 4, 8, 8, 16], "core_layers": 1, "core_depth": 1}


def stats(v: list[float]) -> dict:
    s = sorted(v)
    n = len(s)
    if not n:
        return {"n": 0}
    return {"n": n, "min": s[0], "median": s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]),
            "p90_nearest_rank": s[max(0, math.ceil(0.9 * n) - 1)], "max": s[-1], "mean": math.fsum(s) / n}  # fmt: skip


def main(arm: str, mode: str, out_json: str) -> int:
    import psutil
    import ulr_common as K
    import ulr_spec as S

    proc = psutil.Process(os.getpid())
    rec: dict = {"arm": arm, "mode": mode, "pid": os.getpid()}
    import uid_common as C

    rec["cpu"] = C.configure_cpu(24, 24) if mode == "full" else C.configure_cpu(2, 1)
    import numpy as np
    import torch
    import ulr_model as Mo
    import uoa_model as UM
    from torch.nn import functional as F

    c = S.ARMS[arm]
    cfg = S.MODEL if mode == "full" else {**S.MODEL, **FIXTURE_MODEL}
    rec["code"] = {
        "train_update_source_sha256": Mo.train_update_source_sha256(),
        "ulr_model_sha256": hashlib.sha256(Path(Mo.__file__).read_bytes()).hexdigest(),
        "pilot_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    g = torch.Generator().manual_seed(PILOT_SEED)
    model = Mo.bridged(UM.build(cfg, PILOT_SEED, "leaky_relu"), c)
    q, _ = torch.linalg.qr(torch.randn((16, 16), generator=g, dtype=torch.float64))
    V, mu = q.T.numpy(), torch.randn(16, generator=g, dtype=torch.float64).numpy() * 0.1
    init = {f"reduce_weight_c{c}": V[:c].astype(np.float32).reshape(c, 16, 1, 1), f"reduce_bias_c{c}": (-(V[:c] @ mu)).astype(np.float32),
            f"expand_weight_c{c}": V[:c].T.astype(np.float32).reshape(16, c, 1, 1).copy(), f"expand_bias_c{c}": mu.astype(np.float32)}  # fmt: skip
    Mo.set_bridge(model, init, c)
    rec["model"] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "c": c,
        "wire_latent_numel": S.wire_numel(c),
        "seed": PILOT_SEED,
    }
    o = S.OPTIMIZER
    opt = torch.optim.Adam(
        model.parameters(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"], weight_decay=o["weight_decay"]
    )
    x_n = (torch.rand((1440, 1, 32, 32), generator=g) * 1.6 - 0.8).contiguous()
    sched = Mo.schedule(1440, WARMUP + MEASURED, S.BATCH_SIZE, PILOT_SEED)
    model.train()
    steps, losses = [], []
    t_loop = time.perf_counter()
    for k in range(WARMUP + MEASURED):
        if time.perf_counter() - t_loop > INTERNAL_BUDGET_S:
            break
        t0 = time.perf_counter()
        loss, _ = Mo.train_update(model, opt, x_n[sched[k]], F)
        opt.step()
        steps.append(time.perf_counter() - t0)
        losses.append(float(loss.detach()))
    rec["steps"] = {"warmup": WARMUP, "measured": MEASURED, "completed": len(steps), "per_step_s": steps,
                    "measured_s": stats(steps[WARMUP:]), "all_losses_finite": all(math.isfinite(v) for v in losses)}  # fmt: skip
    meas = []
    for k in range(3):
        t0 = time.perf_counter()
        loss, y_n = Mo.train_update(model, opt, x_n[sched[k]], F)
        _ = (C.metrics(y_n.detach(), x_n[sched[k]]), Mo.grad_stats(model))
        snap = Mo.snapshot(model)
        opt.step()
        _ = (Mo.param_change(model, snap), Mo.param_change(model, snap))
        meas.append(time.perf_counter() - t0)
    rec["curve_point_s"] = stats(meas)
    xs = (torch.rand((1440, 1, 32, 32), generator=g) * 0.03 - 0.015).contiguous()
    with tempfile.TemporaryDirectory(prefix="ulr_pilot_") as tmp:
        t0 = time.perf_counter()
        f = Mo.save_state(Path(tmp) / "a.safetensors", model, {"point": "pilot"})
        Mo.save_state(Path(tmp) / "b.safetensors", model, {"point": "pilot"})
        rec["state_save_s"] = (time.perf_counter() - t0) / 2
        t0 = time.perf_counter()
        _ = Mo.roundtrip_copy(model, xs, 0.02, c)
        rec["checkpoint_eval_1440_s"] = time.perf_counter() - t0
        recv, _ = Mo.receiver_from_state(f, c, cfg)
        snd = Mo.load_arm_state(f, c, cfg)
    with torch.no_grad():
        sub = xs[:200]
        t0 = time.perf_counter()
        s = Mo.send(snd, sub, 0.02, c)
        rec["send_per_tile_s"] = (time.perf_counter() - t0) / sub.shape[0]
        t0 = time.perf_counter()
        _ = Mo.receive(recv, s["payloads"], c, 0.02)
        rec["receive_per_tile_s"] = (time.perf_counter() - t0) / sub.shape[0]
        src = UM.build(cfg, PILOT_SEED, "leaky_relu").eval()
        t0 = time.perf_counter()
        _ = UM.float_outputs(src, sub, 0.02)
        rec["source_float_per_tile_s"] = (time.perf_counter() - t0) / sub.shape[0]
    rec["payload_bytes"] = sorted({len(b) for b in s["payloads"]})
    del model, opt, snd, recv, src
    mi = proc.memory_info()
    rec["memory"] = {
        "os_peak_wset_bytes": getattr(mi, "peak_wset", None),
        "rss_end_bytes": mi.rss,
        "available_end_bytes": psutil.virtual_memory().available,
    }
    rec["discarded"] = {"model_and_optimizer_deleted": True, "states_written_only_to_deleted_temp_dir": True}
    K.check_imports(rec)
    with open(out_json, "xb") as fh:
        fh.write((json.dumps(rec, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
