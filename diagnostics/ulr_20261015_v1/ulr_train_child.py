"""ULR-20261015-v1 training child: ONE adaptation start (C16-CONTROL, C4-REDUCED or C1-REDUCED), in one fresh process.

``python -B ulr_train_child.py <job.json>``. Installs the split-enforcing open guard first: no held-out file, and no
UCX or UOA evaluation output.

- **Start.** The arm starts from the verified source (UOA U-LEAKY-LONG, 12,000 updates, every parameter and buffer)
  plus the persisted channel-PCA bridge of its width c. Each arm starts independently from the source, never from
  another arm. The child refuses unless the backbone equals the source state and the bridge equals the persisted
  initialization.
- **Optimizer.** A fresh Adam over EVERY parameter (backbone and bridge): lr 2e-4, betas (0.9, 0.999), eps 1e-8,
  weight decay 0, batch 8. The loss is the vendored UOA / UCX update: mean normalized-space MSE, train mode, through
  the reduced bottleneck.
- **Schedule.** The newly registered schedule (seed 20261015), checked by hash. Exactly 6,000 updates.
- **Checkpoints.** States at 0 / 1,000 / 3,000 / 6,000 updates. Each is followed by a full-TRAIN evaluation through
  the wire path on an eval-mode deep copy. The training model, its buffers, its mode and the RNG are checked
  unchanged afterwards.
- **Isolation.** No held-out quantity exists in this process.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ulr_common as K  # noqa: E402


class SoftStop(Exception):
    """The soft stop of the arm arrived before training finished."""


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    attempts: list[str] = []
    K.install_open_guard(job["forbidden_paths"], attempts)
    out = Path(job["out_dir"])
    rec: dict = {
        "phase": "train",
        "arm": job["arm"],
        "c": job["c"],
        "mode": job["mode"],
        "pid": os.getpid(),
        "status": "EXECUTION_ERROR",
    }
    t0 = time.monotonic()
    curve = (out / "curve.jsonl").open("x", encoding="utf-8")
    ckpts = (out / "checkpoints.jsonl").open("x", encoding="utf-8")
    try:
        K.start_watchdog(job, f"ulr_train_child[{job['arm']}]")
        if job["mode"] != "engineering" and job.get("fixture_hooks"):
            raise K.ULRError("fixture hooks are refused in scientific mode")
        K.check_env(rec)
        import uid_common as C

        cpu = job["cpu"]
        rec["cpu"] = C.configure_cpu(cpu["intra_op_threads"], cpu["inter_op_threads"])
        if not (
            rec["cpu"]["intra_op_threads"] == cpu["intra_op_threads"]
            and rec["cpu"]["inter_op_threads"] == cpu["inter_op_threads"]
            and rec["cpu"]["deterministic_algorithms"] is True
            and rec["cpu"]["cuda_available"] is False
            and rec["cpu"]["torch_version"] == job["torch_version"]
        ):
            raise K.ULRError(f"CPU configuration not applied as requested: {rec['cpu']}")
        run(job, rec, out, curve, ckpts)
        K.check_imports(rec)
        rec["status"] = "COMPLETED"
    except SoftStop as exc:
        rec["status"] = "RESOURCE_STOP"
        rec["stop_reason"] = str(exc)
    except KeyboardInterrupt:
        rec["status"] = "INCOMPLETE"
    except Exception as exc:
        rec["status"] = "EXECUTION_ERROR"
        rec["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        curve.close()
        ckpts.close()
        rec["split_enforcement"] = {"forbidden_open_attempts": attempts, "audit_hook_installed": True}
        rec["memory"] = K.peak_memory()
        rec["child_wall_s"] = time.monotonic() - t0
        K.write_json(out / "train_record.json", rec)
    return K.EXIT[rec["status"]]


def fixture_hook(job: dict, k: int) -> None:
    h = job.get("fixture_hooks") or {}
    if not h or job["mode"] != "engineering":
        return
    if h.get("raise_at_update") == k:
        raise RuntimeError(f"fixture error at update {k}")


def write_line(fh, line: dict) -> None:
    fh.write(json.dumps(K.json_safe(line), sort_keys=True, allow_nan=False) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def checkpoint(model, k: int, job: dict, out: Path, x, x_n_flat, scale: float) -> dict:
    import numpy as np
    import ulr_model as Mo
    import uoa_sums as U

    c = int(job["c"])
    fp0, rng0, mode0 = Mo.fingerprint(model)["state_sha256"], Mo.rng_sha256(), model.training
    meta = {
        "protocol_id": job["protocol_id"],
        "arm": job["arm"],
        "update": k,
        "source_updates": job["source_updates"],
    }
    ck: dict = {"update": k}
    t = time.perf_counter()
    ck["state_file"] = Mo.save_state(out / f"state_{k:05d}.safetensors", model, meta)
    ck["save_s"] = time.perf_counter() - t
    t = time.perf_counter()
    r = Mo.roundtrip_copy(model, x, scale, c)
    yr, ynr = r["y"].reshape(x.shape[0], -1).numpy(), r["y_n"].reshape(x.shape[0], -1).numpy()
    raw4, norm4 = U.sums4(yr, x.reshape(x.shape[0], -1).numpy()), U.sums4(ynr, x_n_flat)
    pr, pn = U.pooled(raw4), U.pooled(norm4)
    ck["train_eval"] = {
        "mode": "eval-mode deep copy, every tile alone, wire path (64 c bytes)",
        "payload_bytes": r["payload_bytes"],
        "raw": {
            q: pr[q]
            for q in ("sse", "target_energy", "pred_energy", "dot", "n", "mse", "rse", "cosine", "norm_ratio")
        },
        "normalized_mse_direct": pn["mse"],
        "finite": bool(U.finite_rows(yr).all()),
    }
    ck["bridge"] = Mo.bridge_stats(model, r["z"])
    ck["train_eval_s"] = time.perf_counter() - t
    sums = {"TRAIN.raw": raw4, "TRAIN.norm": norm4, "TRAIN.finite": U.finite_rows(yr).astype(np.uint8)}
    ck["sums_file"] = K.save_tensors(
        out / f"checkpoint_{k:05d}_train_sums.safetensors",
        {q: np.ascontiguousarray(v) for q, v in sums.items()},
        {"format": "ulr_20261015_v1.sums/v1", "arm": job["arm"], "update": str(k)},
    )
    ck["training_model_unchanged"] = Mo.fingerprint(model)["state_sha256"] == fp0
    ck["global_rng_unchanged"] = Mo.rng_sha256() == rng0
    ck["mode_kept"] = model.training == mode0
    if not (ck["training_model_unchanged"] and ck["global_rng_unchanged"] and ck["mode_kept"]):
        raise K.ULRError(f"checkpoint {k}: the evaluation changed the training model, its mode or the RNG")
    return ck


def run(job: dict, rec: dict, out: Path, curve, ckpts) -> None:
    import torch
    import uid_common as C
    import uid_r1_measure as M
    import ulr_model as Mo
    from torch import nn

    cfg, c = job["model"], int(job["c"])
    init = K.load_checked(job["bridge_init"], job["bridge_init_sha256"])
    model = Mo.build_arm(job["source"], init, c, cfg)
    rec["start"] = {
        "backbone_sha256": Mo.backbone_sha256(model),
        "source_state_sha256": job["source"]["state_sha256"],
        "bridge_equals_persisted_init": Mo.bridge_equals(model, init, c),
        "state_sha256": Mo.state_sha256(model),
        "source_updates_inherited": job["source_updates"],
    }
    rec["model"] = {
        "backbone_internal_shape": [16, 4, 4],
        "wire_latent_shape": [c, 4, 4],
        "wire_latent_numel": 16 * c,
        "payload_bytes": 64 * c,
        "parameters_by_group": {g: v["numel"] for g, v in Mo.grad_stats(model).items()},
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "buffer_elements": sum(b.numel() for b in model.buffers()),
    }
    x = K.load_checked(job["train_tiles"], job["train_tiles_sha256"], "torch")["x"].contiguous()
    if x.dtype != torch.float32 or tuple(x.shape) != (job["n_train_tiles"], 1, 32, 32):
        raise K.ULRError(
            f"TRAIN tiles {x.dtype} {tuple(x.shape)} are not fp32 [{job['n_train_tiles']}, 1, 32, 32]"
        )
    scale = float(job["scale_global"])
    x_n = Mo.normalize(x, scale).contiguous().requires_grad_(False)
    x_n_flat = x_n.reshape(x.shape[0], -1).numpy()
    rec["inputs"] = {
        "train_tiles_sha256": job["train_tiles_sha256"],
        "normalized_sha256": C.tensor_sha256(x_n, "x_n"),
        "scale_global": scale,
    }
    if rec["inputs"]["normalized_sha256"] != job["expected_normalized_sha256"]:
        raise K.ULRError("the normalized TRAIN tensor differs from the one UCX / UOA trained on")
    n_up, batch = int(job["max_updates"]), int(job["batch_size"])
    sched = Mo.schedule(x.shape[0], n_up, batch, int(job["schedule_seed"]))
    rec["schedule"] = {"sha256": Mo.schedule_sha256(sched), "seed": job["schedule_seed"]}
    if rec["schedule"]["sha256"] != job["schedule_sha256"]:
        raise K.ULRError("the recomputed sampling schedule differs from the registered schedule")
    o = job["optimizer"]
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(o["lr"]),
        betas=tuple(float(b) for b in o["betas"]),
        eps=float(o["eps"]),
        weight_decay=float(o["weight_decay"]),
    )
    rec["trainable"] = M.trainable_coverage(model, opt)
    if (
        rec["trainable"]["in_optimizer"] != rec["model"]["parameters_total"]
        or rec["trainable"]["tensors_missing_from_optimizer"]
    ):
        raise K.ULRError(f"the optimizer does not cover every parameter: {rec['trainable']}")
    marks = {int(v) for v in job["checkpoints"]}
    rec["checkpoints"] = []
    model.train()
    if 0 in marks:
        ck = checkpoint(model, 0, job, out, x, x_n_flat, scale)
        write_line(ckpts, ck)
        rec["checkpoints"].append(
            {q: ck[q] for q in ("update", "state_file", "sums_file", "train_eval", "train_eval_s", "save_s")}
        )
    theta0 = Mo.snapshot(model)
    rec["rng_sha256_before_training"] = M.state_fingerprint(torch.nn.Module())["rng_sha256"]
    soft_at, every = float(job["soft_stop_monotonic"]), int(job["curve_every"])
    exposures = [0] * x.shape[0]
    update_s, measure_s, ck_s, done = [], [], [], 0
    for k in range(1, n_up + 1):
        if time.monotonic() >= soft_at:
            raise SoftStop(f"soft stop before update {k}: {done} update(s) done")
        idx = sched[k - 1]
        fixture_hook(job, k)
        batch_x = x_n[idx]
        t0 = time.perf_counter()
        loss, y_n = Mo.train_update(model, opt, batch_x, nn.functional)
        t1 = time.perf_counter()
        point = None
        if k == 1 or k % every == 0 or k == n_up:
            point = {
                "update": k,
                "pre_update_train_loss": float(loss.detach()),
                "batch_indices": idx,
                "pre_update_batch": C.metrics(y_n.detach(), batch_x),
                "grad": Mo.grad_stats(model),
            }
            prev = Mo.snapshot(model)
        t2 = time.perf_counter()
        opt.step()
        t3 = time.perf_counter()
        update_s.append((t1 - t0) + (t3 - t2))
        for i in idx:
            exposures[i] += 1
        done = k
        if point is not None:
            point["param_change_this_update"] = Mo.param_change(model, prev)
            point["param_change_from_start"] = Mo.param_change(model, theta0)
            point["monotonic_s"] = time.monotonic()
            point["measurement_s"] = time.perf_counter() - t3 + (t2 - t1)
            measure_s.append(point["measurement_s"])
            write_line(curve, point)
        if k in marks:
            t4 = time.perf_counter()
            ck = checkpoint(model, k, job, out, x, x_n_flat, scale)
            ck["monotonic_s"] = time.monotonic()
            ck_s.append(time.perf_counter() - t4)
            write_line(ckpts, ck)
            rec["checkpoints"].append(
                {
                    q: ck[q]
                    for q in ("update", "state_file", "sums_file", "train_eval", "train_eval_s", "save_s")
                }
            )
    rec["updates_done"] = done
    rec["exposures"] = {
        "sum": sum(exposures),
        "histogram": {str(v): exposures.count(v) for v in sorted(set(exposures))},
        "passes_over_train": sum(exposures) / x.shape[0],
        "optimizer_steps": done,
        "adam_state_steps": sorted({int(s["step"]) for s in opt.state.values()}),
        "heldout_tiles_in_any_batch": 0,
    }
    bn = {n: b for n, b in model.named_buffers() if n.endswith("num_batches_tracked")}
    rec["batchnorm"] = {
        "num_batches_tracked": sorted({int(b) for b in bn.values()}),
        "layers": len(bn),
        "running_stats_finite": all(
            bool(torch.isfinite(b).all()) for n, b in model.named_buffers() if n.endswith(("_mean", "_var"))
        ),
    }
    rec["update_time_s"] = {"compute_only": K.stats(update_s), "per_update": update_s}
    rec["measurement_time_s"] = K.stats(measure_s)
    rec["checkpoint_time_s"] = K.stats(ck_s)
    rec["model_state_sha256_final"] = Mo.state_sha256(model)
    rec["param_change_total"] = Mo.param_change(model, theta0)
    rec["rng_sha256_after_training"] = M.state_fingerprint(torch.nn.Module())["rng_sha256"]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
