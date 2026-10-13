"""UID-20261012-v2 r1 child: one arm in one fresh process.

``python -B uid_r1_child.py <job.json>``. The job file (written by the supervisor) names the arm, the frozen config,
the arm's output directory and the soft stop on the supervisor's ``time.monotonic()`` clock (system-wide on Windows).
The CPU configuration is applied, and observed, before the model, the input or any other tensor is created. The child
writes ``child_record.json`` and ``curve.jsonl``; a process that is killed writes nothing more, and its arm is closed
by the supervisor from the start record.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

R1_DIR = Path(__file__).resolve().parent
FROZEN_DIR = R1_DIR.parent / "uid_20261012_v2"


class SoftStop(Exception):
    """The soft stop of this arm arrived before training finished."""


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    out = Path(job["out_dir"])
    start_watchdog(job)  # first: also covers imports and setup; exits at once if the supervisor is gone
    sys.path[:0] = [str(Path(p)) for p in job["sys_path"]]  # frozen package first, then r1
    import signal

    if hasattr(signal, "SIGBREAK"):  # Ctrl+Break in a shared console: an interrupt, not an abrupt exit
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    import torch
    import uid_common as C  # frozen package module
    import uid_r1_control as K
    import uid_r1_measure as M

    rec: dict = {"arm": job["arm"], "mode": job["mode"], "pid": os.getpid(), "status": "EXECUTION_ERROR"}
    curve = (out / "curve.jsonl").open("x", encoding="utf-8")
    sampler = M.RSSSampler(job["poll_interval_s"]).start()
    job["_sampler"] = sampler
    t_begin = time.monotonic()
    try:
        if job["mode"] == "scientific" and job.get("fixture_hooks"):
            raise RuntimeError("fixture hooks are refused in scientific mode")
        rec["env"] = {k: os.environ.get(k) for k in ENV_KEYS}
        if rec["env"]["CUDA_VISIBLE_DEVICES"] != "-1" or rec["env"]["PYTHONPATH"] is not None:
            raise RuntimeError(f"child environment not as set by the supervisor: {rec['env']}")
        cpu = job["cpu"]
        try:
            rec["cpu"] = C.configure_cpu(cpu["intra_op_threads"], cpu["inter_op_threads"])
        except Exception as exc:  # report, never claim
            rec["cpu"] = {"error": f"{type(exc).__name__}: {exc}"}
            raise
        rec["cpu"]["requested"] = cpu
        rec["cpu"]["applied_as_requested"] = (
            rec["cpu"]["intra_op_threads"] == cpu["intra_op_threads"]
            and rec["cpu"]["inter_op_threads"] == cpu["inter_op_threads"]
            and rec["cpu"]["deterministic_algorithms"] is True
            and rec["cpu"]["cuda_available"] is False
        )
        if not rec["cpu"]["applied_as_requested"]:
            raise RuntimeError(f"CPU configuration not applied as requested: {rec['cpu']}")
        if rec["cpu"]["torch_version"] != job["expected_torch_version"]:
            raise RuntimeError(f"torch {rec['cpu']['torch_version']} is not {job['expected_torch_version']}")
        cfg, raw_sha, canon_sha = load_job_config(job, C, K)
        rec["config_sha256"] = {"raw": raw_sha, "canonical": canon_sha}
        run_arm(job, cfg, rec, curve, torch, C, M)
    except SoftStop as exc:
        rec["status"] = "RESOURCE_STOP"
        rec["stop_reason"] = str(exc)
    except KeyboardInterrupt:
        rec["status"] = "INCOMPLETE"
        # a stop request reaches the main thread at its next Python bytecode; a main thread blocked in one long
        # native call is not woken and is ended by the supervisor's kill instead (no child record then)
        rec["stop_reason"] = "interrupted (stop request or Ctrl+C -> KeyboardInterrupt)"
    except Exception as exc:
        rec["status"] = "EXECUTION_ERROR"
        rec["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        curve.close()
        rec["rss"] = sampler.stop()
        rec["child_monotonic_s"] = {"begin": t_begin, "end": time.monotonic()}
        K.create_exclusive(out / "child_record.json", K.json_safe(rec))
    return K.CHILD_EXIT[rec["status"]]


FIXTURE_PROTOCOL_ID = "UID-R1-ENGINEERING-FIXTURE"
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH")
MODULES = ("uid_common", "uformer_ae", "uformer_blocks", "uid_r1_control", "uid_r1_measure")


def imported_modules(job: dict, K) -> dict:
    """Files and SHA-256 of the diagnostic modules actually imported; in scientific mode each must have its approved
    hash, and ``cg_fedllm`` (the package, with uncommitted SSBD work) must not be imported at all."""
    files = {n: str(Path(sys.modules[n].__file__).resolve()) for n in MODULES if n in sys.modules}
    files["uid_r1_child"] = str(Path(__file__).resolve())
    sha = {n: K.sha256_file(p) for n, p in files.items()}
    expected = job.get("expected_module_sha256")
    if expected is not None:
        bad = {n: sha[n] for n in sha if expected.get(n) != sha[n]}
        if bad:
            raise RuntimeError(f"imported module(s) without the approved SHA-256: {bad}")
    if "cg_fedllm" in sys.modules:
        raise RuntimeError("cg_fedllm must not be imported by the diagnostic")
    return {"files": files, "sha256": sha, "checked_against_approved": expected is not None}


def load_job_config(job: dict, C, K) -> tuple[dict, str, str]:
    """Scientific: the frozen loader and the frozen hashes only. Engineering: a fixture config only, which can never
    be the registered one (own protocol id, other bytes, other seeds)."""
    if job["mode"] == "scientific":
        cfg, raw, canon = C.load_config(job["config_path"])
        if (raw, canon) != (K.FROZEN_CONFIG_RAW_SHA256, K.FROZEN_CONFIG_CANONICAL_SHA256):
            raise RuntimeError("scientific job with a config other than the frozen one")
        if job["model_arm"] != job["arm"]:
            raise RuntimeError("scientific arm label and model arm differ")
        return cfg, raw, canon
    if job["mode"] != "engineering":
        raise RuntimeError(f"unknown mode {job['mode']!r}")
    data = Path(job["config_path"]).read_bytes()
    cfg = json.loads(data.decode("utf-8"))
    raw, canon = K.hashlib.sha256(data).hexdigest(), K.sha256_canonical(cfg)
    if cfg.get("protocol_id") != FIXTURE_PROTOCOL_ID or raw == K.FROZEN_CONFIG_RAW_SHA256:
        raise RuntimeError("engineering jobs accept only fixture configs")
    if int(cfg["input"]["data_seed"]) == 20261012 or int(cfg["model_init_seed"]) == 1:
        raise RuntimeError("fixture configs must not use the registered seeds")
    return cfg, raw, canon


def start_watchdog(job: dict) -> None:
    """Exit (no record) if the supervisor is gone or the hard deadline passes; turn a stop request into
    KeyboardInterrupt. The supervisor is identified by PID and creation time, so a reused PID is not mistaken for it."""
    import _thread
    import threading

    import psutil

    def gone(why: str) -> None:
        sys.stderr.write(f"uid_r1_child: {why}; exiting without a child record\n")
        sys.stderr.flush()
        os._exit(137)  # the supervisor (or, if it is dead, the start record) closes the arm

    try:
        parent = psutil.Process(int(job["supervisor_pid"]))
        same = abs(parent.create_time() - float(job["supervisor_create_time"])) < 0.01
    except psutil.Error:
        same = False
    if not same:
        gone("supervisor not running at child start")
    hard_at, poll, stop_file = (
        float(job["hard_stop_monotonic"]),
        float(job["poll_interval_s"]),
        Path(job["stop_file"]),
    )
    hooks = job.get("fixture_hooks") or {}
    ignore_stop = job["mode"] == "engineering" and bool(hooks.get("ignore_stop_request"))

    def run() -> None:
        interrupted = False
        while True:
            time.sleep(poll)
            if time.monotonic() >= hard_at:
                gone("hard deadline passed")
            if not parent.is_running():
                gone("supervisor ended")
            if not interrupted and not ignore_stop and stop_file.exists():
                interrupted = True
                _thread.interrupt_main()

    threading.Thread(target=run, name="watchdog", daemon=True).start()


def fixture_hook(job: dict, where: str, k: int = 0) -> None:
    """Engineering-only behaviours; ``run_arm`` refuses any hook in scientific mode before the watchdog starts."""
    h = job.get("fixture_hooks") or {}
    if not h:
        return
    if where == "update":
        if h.get("sleep_each_update_s"):
            time.sleep(float(h["sleep_each_update_s"]))
        if h.get("hang_at_update") == k:
            time.sleep(10**6)
        if h.get("raise_at_update") == k:
            raise RuntimeError(f"fixture error at update {k}")
        if h.get("alloc_mib_at_update", [None])[0] == k:
            import torch

            job["_ballast"] = torch.ones(int(h["alloc_mib_at_update"][1]) * 2**18)  # touched float32 pages
            time.sleep(10**6)
    elif where == "final_eval" and h.get("hang_in_final_eval"):
        time.sleep(10**6)


def write_point(curve, line: dict, K) -> None:
    curve.write(json.dumps(K.json_safe(line), sort_keys=True, allow_nan=False) + "\n")
    curve.flush()
    os.fsync(curve.fileno())


def run_arm(job: dict, cfg: dict, rec: dict, curve, torch, C, M) -> None:
    """Frozen settings throughout: ``C.build_model`` / ``C.make_input`` / ``C.reconstruct`` / ``C.metrics`` and the
    arm's frozen optimizer and update count; only the measurement and stop mechanics are new."""
    import uid_r1_control as K
    from torch import nn

    arm, soft_at, every = job["model_arm"], float(job["soft_stop_monotonic"]), int(job["curve_every"])
    acfg = cfg[arm.lower().replace("-", "_")]
    if int(acfg["max_updates"]) != int(job["max_updates"]):
        raise RuntimeError(f"job max_updates {job['max_updates']} != config {acfg['max_updates']}")
    model = C.build_model(cfg, arm)
    rec["modules"] = imported_modules(job, K)  # after build_model, which imports uformer_ae / uformer_blocks
    x = C.make_input(cfg).requires_grad_(False)
    o = acfg["optimizer"]
    if o["name"] != "Adam":
        raise RuntimeError(f"unsupported optimizer {o['name']!r}")
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(o["lr"]),
        betas=tuple(float(b) for b in o["betas"]),
        eps=float(o["eps"]),
        weight_decay=float(o["weight_decay"]),
    )
    rec["input"] = C.describe_input(x)
    rec["baselines"] = C.baselines(x)
    rec["model_state_sha256_init"] = C.state_sha256(model)
    rec["trainable"] = M.trainable_coverage(model, opt)
    theta0 = M.snapshot(model)
    # step 0 on disposable copies: the training model, its BN buffers and the RNG state are left untouched
    fp_before = M.state_fingerprint(model)
    step0 = {}
    for mode in ("eval", "train"):
        with M.isolated(model, mode) as twin, torch.no_grad():
            step0[mode] = C.metrics(C.reconstruct(twin, x), x)
    fp_after = M.state_fingerprint(model)
    rec["step0"] = {"post_init": step0, "isolation_verified": fp_before == fp_after}
    if fp_before != fp_after:
        raise RuntimeError("step-0 measurement changed the training model or the RNG state")
    model.train()
    max_updates, done, prev = int(acfg["max_updates"]), 0, None
    for k in range(1, max_updates + 1):
        if time.monotonic() >= soft_at:
            raise SoftStop(f"soft stop before update {k}: {done} update(s) done")
        fixture_hook(job, "update", k)
        opt.zero_grad(set_to_none=True)
        x_hat = C.reconstruct(model, x)  # train-mode forward of update k (pre-update for update k)
        loss = nn.functional.mse_loss(x_hat, x)
        loss.backward()
        point = None
        if k == 1 or k % every == 0 or k == max_updates:
            point = {
                "update": k,
                "pre_update_train_loss": float(loss.detach()),
                "pre_update_train": C.metrics(x_hat, x),
                "grad": M.grad_stats(model),
            }
            prev = M.snapshot(model)
        opt.step()
        done = k
        if point is not None:
            point["param_change_this_update"] = M.param_change(model, prev)
            point["param_change_from_init"] = M.param_change(model, theta0)
            point["monotonic_s"] = time.monotonic()
            point["rss_bytes_at_point"] = job["_sampler"].poll()
            write_point(curve, point, K)
    rec["updates_done"] = done
    rec["model_state_sha256_final"] = C.state_sha256(model)
    rec["param_change_total"] = M.param_change(model, theta0)
    fixture_hook(job, "final_eval")
    # final: two independent copies of the same final state; each starts from the same CPU RNG state
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    final, outputs, loaded, loaded_before = {}, {}, [], []
    for mode in ("eval", "train"):
        twin = C.build_model(cfg, arm)
        twin.load_state_dict(final_state, strict=True)
        loaded_before.append(C.state_sha256(twin))
        twin.train(mode == "train")
        torch.set_rng_state(rng)
        with torch.no_grad():
            outputs[mode] = C.reconstruct(twin, x)
        final[f"{mode}_mode"] = C.metrics(outputs[mode], x)
        loaded.append(C.state_sha256(twin))
    final["copies_loaded_identical_state"] = len(set(loaded_before)) == 1
    final["train_copy_state_changed_by_forward"] = loaded[1] != loaded_before[1]
    final["train_minus_eval_output_sse"] = float(
        (outputs["train"].double() - outputs["eval"].double()).square().sum()
    )
    rec["final"] = final
    rec["verdict_basis"] = "eval_mode"
    rec["status"] = (
        "COMPLETED_PASS" if C.passes(final["eval_mode"], cfg["criteria"]) else "COMPLETED_FIT_FAIL"
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
