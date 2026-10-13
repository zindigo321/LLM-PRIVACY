"""UCOV-20261013-v1 child: one tile arm in one fresh process (adapted from the URP-20261013-v1 ``urp_child.py``).

``python -B ucov_child.py <job.json>``. The job file, written by the supervisor, names:
- the arm, the resolved config and the arm's output directory;
- the soft stop, on the supervisor's ``time.monotonic()`` clock.

Setup comes first: the CPU configuration is applied and observed before the model, the input or any other tensor is
created.

The child writes ``child_record.json``, ``curve.jsonl``, ``init_state.safetensors`` and ``final_state.safetensors``. A
process that is killed writes nothing more, and the supervisor closes its arm from the start record.

Changes from URP (``review/child_diff.patch``):
- **Input.** The arm's own registered tile, from ``ucov_control.load_tile``.
- **Model.** A fresh frozen U-FIT Uformer and a fresh Adam per arm. Its initial state must equal the U-FIT record's.
- **Status.** The terminal status of a completed arm is the PRIMARY policy UPP-20261013-v1, from
  ``upp_classify.primary_metrics`` on the final eval-mode original-space sums and the arm's two baselines:
  COMPLETED_PASS or COMPLETED_FIT_FAIL.
  - The child never reads the UID ``criteria``.
  - The historical strict and EP-v1 references are computed after the run, by the analyzer only.
- **Unchanged from URC/URP:** measurements, the training loop, step-0 isolation, the final train/eval procedure and
  the state files.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

UCOV_DIR = Path(__file__).resolve().parent


class SoftStop(Exception):
    """The soft stop of this arm arrived before training finished."""


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    out = Path(job["out_dir"])
    start_watchdog(job)  # first: also covers imports and setup; exits at once if the supervisor is gone
    sys.path[:0] = [str(Path(p)) for p in job["sys_path"]]  # UCOV package, then URP, UPP, frozen UID and r1
    import signal

    if hasattr(signal, "SIGBREAK"):  # Ctrl+Break in a shared console: an interrupt, not an abrupt exit
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    import torch
    import ucov_control as U
    import uid_common as C  # frozen UID module, unchanged
    import uid_r1_control as K  # r1 module, unchanged
    import uid_r1_measure as M  # r1 module, unchanged

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
        cfg, raw_sha, canon_sha = load_job_config(job, K, U)
        rec["config_sha256"] = {"raw": raw_sha, "canonical": canon_sha}
        run_arm(job, cfg, rec, curve, torch, C, M, U)
    except SoftStop as exc:
        rec["status"] = "RESOURCE_STOP"
        rec["stop_reason"] = str(exc)
    except KeyboardInterrupt:
        rec["status"] = "INCOMPLETE"
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


FIXTURE_PROTOCOL_ID = "UCOV-V1-ENGINEERING-FIXTURE"
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH")
MODULES = (
    "uid_common",
    "uformer_ae",
    "uformer_blocks",
    "uid_r1_control",
    "uid_r1_measure",
    "urp_control",
    "upp_classify",
    "ucov_control",
)
FORBIDDEN_MODULES = ("cg_fedllm", "resnet3_s1", "urc_control", "ep_classify", "urp_child", "urp_analyze")


def imported_modules(job: dict, K) -> dict:
    """Files and SHA-256 of the diagnostic modules actually imported; in scientific mode each must have its approved
    hash. ``cg_fedllm``, the URC ResNet modules, the EP classifier and the closed URP child/analyzer must not be
    imported, and every imported repository module must be listed."""
    files = {n: str(Path(sys.modules[n].__file__).resolve()) for n in MODULES if n in sys.modules}
    files["ucov_child"] = str(Path(__file__).resolve())
    sha = {n: K.sha256_file(p) for n, p in files.items()}
    expected = job.get("expected_module_sha256")
    if expected is not None:
        bad = {n: sha.get(n) for n in set(sha) | set(expected) if expected.get(n) != sha.get(n)}
        if bad:
            raise RuntimeError(f"imported module(s) without the approved SHA-256: {bad}")
    present = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    if present:
        raise RuntimeError(f"module(s) {present} must not be imported by the UCOV diagnostic")
    repo = UCOV_DIR.parents[1]
    local = sorted(
        n
        for n, m in list(sys.modules.items())
        if n not in ("__main__", "__mp_main__")
        and isinstance(getattr(m, "__file__", None), str)
        and Path(m.__file__).is_absolute()
        and Path(m.__file__).resolve().is_relative_to(repo)
    )
    unlisted = [n for n in local if n not in MODULES]
    if unlisted:
        raise RuntimeError(f"repository module(s) imported but not listed as runtime modules: {unlisted}")
    return {"files": files, "sha256": sha, "checked_against_approved": expected is not None, "local": local}


def load_job_config(job: dict, K, U) -> tuple[dict, str, str]:
    """Scientific: the frozen resolved config, which must equal a fresh resolution of the pinned sources and have the
    hash named in the job. Engineering: a fixture config only, which can never be the registered one."""
    data = Path(job["config_path"]).read_bytes()
    cfg = json.loads(data.decode("utf-8"))
    raw, canon = K.hashlib.sha256(data).hexdigest(), K.sha256_canonical(cfg)
    if job["mode"] == "scientific":
        if Path(job["config_path"]).resolve() != (U.UCOV_DIR / U.RESOLVED_NAME).resolve():
            raise RuntimeError("scientific job with a config other than the frozen resolved_config.json")
        if raw != job.get("expected_config_sha256") or cfg != U.load_resolved():
            raise RuntimeError("resolved_config.json is not the frozen, freshly re-resolved configuration")
        if job["model_arm"] != job["arm"] or job["arm"] not in U.ARMS:
            raise RuntimeError(
                f"scientific arm must be one of {U.ARMS}, got {job['arm']!r} / {job['model_arm']!r}"
            )
        return cfg, raw, canon
    if job["mode"] != "engineering":
        raise RuntimeError(f"unknown mode {job['mode']!r}")
    if cfg.get("protocol_id") != FIXTURE_PROTOCOL_ID or raw == job.get("expected_config_sha256"):
        raise RuntimeError("engineering jobs accept only fixture configs")
    if int(cfg["model_init_seed"]) == 1:
        raise RuntimeError("fixture configs must not use the registered model seed")
    registered = {t["patch_file_sha256"] for t in U.load_ucov_config()["inputs"]["tiles"].values()}
    if any(t["patch_file_sha256"] in registered for t in cfg["inputs"]["tiles"].values()):
        raise RuntimeError("fixture configs must not use a registered tile")
    return cfg, raw, canon


def precision_checks(model, x, torch) -> dict:
    """CPU fp32 for the input and every parameter and buffer of the model (the shared precision condition)."""
    float_tensors = [t for t in (*model.parameters(), *model.buffers()) if t.is_floating_point()]
    return {
        "input_dtype_cpu": (str(x.dtype), str(x.device)) == ("torch.float32", "cpu"),
        "model_float_tensors_fp32_cpu": all(
            t.dtype == torch.float32 and t.device.type == "cpu" for t in float_tensors
        ),
        "default_dtype": str(torch.get_default_dtype()),
    }


def spaces(y_n, x, x_n, scale: float, C, R) -> dict:
    """One output in both spaces: original-space sums of ``denormalize(y_n)`` against the tile ``x`` (the verdict
    basis) and normalized-space sums of ``y_n`` against ``x_n`` (the training coordinates)."""
    return {"original": C.metrics(R.denormalize(y_n, scale), x), "normalized": C.metrics(y_n, x_n)}


def start_watchdog(job: dict) -> None:
    """Exit (no record) if the supervisor is gone or the hard deadline passes; turn a stop request into
    KeyboardInterrupt. The supervisor is identified by PID and creation time, so a reused PID is not mistaken for it."""
    import _thread
    import threading

    import psutil

    def gone(why: str) -> None:
        sys.stderr.write(f"ucov_child: {why}; exiting without a child record\n")
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
    """Engineering-only behaviours; ``main`` refuses any hook in scientific mode before the arm starts."""
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


def run_arm(job: dict, cfg: dict, rec: dict, curve, torch, C, M, U) -> None:
    """Frozen settings throughout: ``C.reconstruct`` / ``C.metrics`` / ``C.baselines`` / ``C.describe_input`` (frozen
    UID), the resolved optimizer and update count, and the frozen ``uid_common.build_model(cfg, "U-FIT")`` through
    the frozen URP binding. The model sees only ``x_n`` (normalized space); the loss is normalized-space MSE, as in the
    production loop. The terminal status of a completed arm is the primary policy UPP-20261013-v1."""
    import uid_r1_control as K
    import upp_classify as P
    from torch import nn

    R = U.urp()
    arm = job["arm"]
    soft_at, every, out = float(job["soft_stop_monotonic"]), int(job["curve_every"]), Path(job["out_dir"])
    if int(cfg["max_updates"]) != int(job["max_updates"]):
        raise RuntimeError(f"job max_updates {job['max_updates']} != config {cfg['max_updates']}")
    if not cfg["inputs_expected"][arm]["usable"]:
        raise RuntimeError(
            f"{arm}: a degenerate registered tile is never trained (the supervisor must not start it)"
        )
    policy = U.load_primary_policy(
        cfg["primary_policy"]
    )  # pinned file, pinned classifier, expected thresholds
    rec["primary_policy"] = {
        "policy_id": policy["policy_id"],
        "policy_sha256": policy["policy_sha256"],
        "thresholds": policy["thresholds"],
    }
    model = U.build_model(cfg)  # fresh model for this arm
    rec["modules"] = imported_modules(job, K)  # after build_model, which imports uformer_ae / uformer_blocks
    rec["model"] = {"config": model.cfg.to_dict(), "parameters": model.num_parameters()}
    rec["model"]["buffer_elements"] = sum(b.numel() for b in model.buffers())
    rec["model_state_sha256_init"] = C.state_sha256(model)
    if (
        job["mode"] == "scientific"
        and rec["model_state_sha256_init"] != cfg["reference"]["model_state_sha256_init"]
    ):
        raise RuntimeError(f"initial state {rec['model_state_sha256_init']} is not the U-FIT initial state")
    scale = float(cfg["inputs"]["normalization"]["scale_global"])
    x, x_n, patch_sha = U.load_tile(cfg, arm)
    x, x_n = x.requires_grad_(False), x_n.requires_grad_(False)
    rec["input"] = {
        "patch_file_sha256": patch_sha,
        "slice": cfg["inputs_expected"][arm]["slice"],
        "original": C.describe_input(x),
        "normalized": C.describe_input(x_n),
    }
    rec["baselines"] = {"original": C.baselines(x), "normalized": C.baselines(x_n)}
    rec["input_checks"] = U.input_checks(rec, cfg["inputs_expected"][arm], float(x_n.abs().max()))
    rec["precision"] = precision_checks(model, x_n, torch)
    if not (rec["precision"]["input_dtype_cpu"] and rec["precision"]["model_float_tensors_fp32_cpu"]):
        raise RuntimeError(f"precision condition (CPU fp32) not met: {rec['precision']}")
    if job["mode"] == "scientific" and not rec["input_checks"]["all_equal"]:
        raise RuntimeError(f"registered input check failed: {rec['input_checks']}")
    o = cfg["optimizer"]
    if o["name"] != "Adam":
        raise RuntimeError(f"unsupported optimizer {o['name']!r}")
    opt = torch.optim.Adam(  # fresh optimizer for this arm
        model.parameters(),
        lr=float(o["lr"]),
        betas=tuple(float(b) for b in o["betas"]),
        eps=float(o["eps"]),
        weight_decay=float(o["weight_decay"]),
    )
    rec["optimizer_defaults"] = {
        k: v for k, v in opt.defaults.items() if isinstance(v, (int, float, tuple, bool))
    }
    meta = {"protocol_id": cfg["protocol_id"], "arm": arm, "variant": cfg["variant"]}
    rec["init_state_file"] = R.save_state(out / "init_state.safetensors", model, {**meta, "point": "init"})
    if rec["init_state_file"]["state_sha256"] != rec["model_state_sha256_init"]:
        raise RuntimeError("initial state file hash differs from the initial model state")
    rec["trainable"] = M.trainable_coverage(model, opt)
    theta0 = M.snapshot(model)
    fp_before = M.state_fingerprint(model)  # step 0 on disposable copies: model, BN buffers and RNG untouched
    step0 = {}
    for mode in ("eval", "train"):
        with M.isolated(model, mode) as twin, torch.no_grad():
            step0[mode] = spaces(C.reconstruct(twin, x_n), x, x_n, scale, C, R)
    fp_after = M.state_fingerprint(model)
    rec["step0"] = {"post_init": step0, "isolation_verified": fp_before == fp_after}
    if fp_before != fp_after:
        raise RuntimeError("step-0 measurement changed the training model or the RNG state")
    model.train()
    max_updates, done, prev = int(cfg["max_updates"]), 0, None
    for k in range(1, max_updates + 1):
        if time.monotonic() >= soft_at:
            raise SoftStop(f"soft stop before update {k}: {done} update(s) done")
        fixture_hook(job, "update", k)
        opt.zero_grad(set_to_none=True)
        y_n = C.reconstruct(
            model, x_n
        )  # train-mode forward of update k (pre-update for update k), normalized
        loss = nn.functional.mse_loss(y_n, x_n)
        loss.backward()
        point = None
        if k == 1 or k % every == 0 or k == max_updates:
            point = {
                "update": k,
                "pre_update_train_loss": float(loss.detach()),
                "pre_update_train": C.metrics(y_n, x_n),  # normalized space, the loss coordinates
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
    rec["final_state_file"] = R.save_state(out / "final_state.safetensors", model, {**meta, "point": "final"})
    if rec["final_state_file"]["state_sha256"] != rec["model_state_sha256_final"]:
        raise RuntimeError("final state file hash differs from the final model state")
    fixture_hook(job, "final_eval")
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    final, outputs, loaded, loaded_before = {}, {}, [], []
    for mode in ("eval", "train"):  # two independent copies of the final state, from the same CPU RNG state
        twin = U.build_model(cfg)
        twin.load_state_dict(final_state, strict=True)
        loaded_before.append(C.state_sha256(twin))
        twin.train(mode == "train")
        torch.set_rng_state(rng)
        with torch.no_grad():
            outputs[mode] = C.reconstruct(twin, x_n)
        both = spaces(outputs[mode], x, x_n, scale, C, R)
        final[f"{mode}_mode"] = both["original"]  # the verdict basis (eval mode), original space
        final[f"{mode}_mode_normalized"] = both["normalized"]  # reported only
        loaded.append(C.state_sha256(twin))
    rec["modules_at_end"] = imported_modules(job, K)  # nothing unlisted was imported during the run
    final["copies_loaded_identical_state"] = len(set(loaded_before)) == 1
    final["train_copy_state_changed_by_forward"] = loaded[1] != loaded_before[1]
    final["train_minus_eval_output_sse_normalized"] = float(
        (outputs["train"].double() - outputs["eval"].double()).square().sum()
    )
    rec["final"] = final
    rec["verdict_basis"] = "eval_mode, original space, primary policy UPP-20261013-v1"
    b = rec["baselines"]["original"]
    verdict = P.primary_metrics(
        final["eval_mode"], b["zero_prediction"]["mse"], b["shared_scalar_mean"]["mse"], policy
    )
    rec["primary_verdict"] = verdict
    # COMPLETED_PASS / COMPLETED_FIT_FAIL now mean the PRIMARY policy (NONINFORMATIVE is a non-pass, class F)
    rec["status"] = "COMPLETED_PASS" if verdict["outcome"] == "PASS" else "COMPLETED_FIT_FAIL"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
