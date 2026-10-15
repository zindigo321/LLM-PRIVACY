"""ULR-20261015-v1 supervisor: one invocation of five registered steps, each in a fresh child process.

Scientific use (the frozen package):
    python -B ulr_supervise.py --scientific --frozen-ulr-manifest-sha256 <sha256>
Read-only:
    python -B ulr_supervise.py --status
Engineering (lifecycle checks only; fixture configs, never the registered data or run area):
    python -B ulr_supervise.py --fixture <fixture_config.json>

Steps, in order: INIT, C16-CONTROL, C4-REDUCED, C1-REDUCED (the three neural starts), EVAL.
- **Ledger.** Each step has a start record before its child is spawned and an end record after it ends, in a
  two-root write-once ledger.
  - A start without an end blocks every later invocation.
  - A step that does not complete ends the invocation, and later steps are NOT_RUN.
  - There is no retry and no option selecting a step, seed, arm, width or budget.
- **Fixed before held-out data.** The start record of EVAL carries the SHA-256 of every fitted or shared quantity:
  the bridge initializations, every arm state, the codec state and the source. Every earlier child is refused the
  held-out file and every UCX / UOA evaluation output.
- **Limits** run on one monotonic clock:
  - the per-step wall limits;
  - the invocation aggregate (9,000 s minus the pilots and the verification reserve);
  - the working set of each child process tree against min(4 GiB, 50% of the available RAM just before that child's
    launch).

  A neural step is refused below the registered floor.
The supervisor imports no torch and no numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

T_START = time.monotonic()
PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))

import psutil  # noqa: E402
import ulr_common as K  # noqa: E402
import ulr_spec as S  # noqa: E402

CONFIG_NAME = "ulr_config.json"
MANIFEST_NAME = "manifest_ulr.json"
CHILD_SCRIPT = {"init": "ulr_init_child.py", "train": "ulr_train_child.py", "eval": "ulr_eval_child.py"}
RECORD_NAME = {"init": "init_record.json", "train": "train_record.json", "eval": "eval_record.json"}


def dir_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def tree_memory(proc: psutil.Process) -> tuple[int, int]:
    rss = private = 0
    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0, 0
    for p in procs:
        try:
            mi = p.memory_info()
        except psutil.NoSuchProcess:
            continue
        rss += mi.rss
        private += getattr(mi, "private", 0) or 0
    return rss, private


def kill_tree(proc: psutil.Process, wait_s: float) -> None:
    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.NoSuchProcess:
        procs = [proc]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=wait_s)


def run_child(phase: str, job: dict, plan: dict, out: Path, limit_s: float, stop: list) -> dict[str, Any]:
    avail = psutil.virtual_memory().available
    rss_limit = min(int(plan["rss_cap_bytes"]), int(avail * plan["rss_available_fraction"]))
    t_spawn = time.monotonic()
    kill_at = min(t_spawn + limit_s, plan["t0"] + plan["aggregate_s"]) - plan["end_margin_s"]
    stop_file = out / f"STOP_REQUEST_{phase}"
    hooks = (
        (plan.get("fixture_hooks") or {}).get(job.get("arm", phase))
        if plan["mode"] == "engineering"
        else None
    )
    job = {
        **job,
        "fixture_hooks": hooks,
        "phase": phase,
        "mode": plan["mode"],
        "protocol_id": plan["protocol_id"],
        "out_dir": str(out),
        "cpu": plan["cpu"],
        "torch_version": plan["torch_version"],
        "hard_stop_monotonic": kill_at + plan["kill_wait_s"],
        "soft_stop_monotonic": kill_at - plan["save_reserve_s"],
        "poll_interval_s": plan["poll_interval_s"],
        "supervisor_pid": os.getpid(),
        "supervisor_create_time": psutil.Process(os.getpid()).create_time(),
        "stop_file": str(stop_file),
    }
    job_path = out / f"{phase}_job.json"
    K.write_json(job_path, job)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(K.CHILD_ENV)
    cmd = [plan["python_executable"], "-B", str(PKG / CHILD_SCRIPT[phase]), str(job_path)]
    guard = {
        "rss_limit_bytes": rss_limit,
        "available_before_bytes": avail,
        "samples": 0,
        "sampled_max_rss_bytes": 0,
        "sampled_max_private_bytes_reported_only": 0,
        "action": None,
    }
    with (out / f"{phase}.stdout.txt").open("xb") as so, (out / f"{phase}.stderr.txt").open("xb") as se:
        popen = subprocess.Popen(
            cmd,
            cwd=str(PKG),
            env=env,
            stdout=so,
            stderr=se,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        try:
            proc = psutil.Process(popen.pid)
        except psutil.NoSuchProcess:
            proc = None
        stop_sent = None
        while popen.poll() is None and proc is not None:
            try:
                rss, private = tree_memory(proc)
                guard["samples"] += 1
                guard["sampled_max_rss_bytes"] = max(guard["sampled_max_rss_bytes"], rss)
                guard["sampled_max_private_bytes_reported_only"] = max(
                    guard["sampled_max_private_bytes_reported_only"], private
                )
                now = time.monotonic()
                if rss > rss_limit:
                    guard["action"] = "killed: RSS limit"
                    kill_tree(proc, plan["kill_wait_s"])
                    break
                if now >= kill_at:
                    guard["action"] = "killed: hard deadline"
                    kill_tree(proc, plan["kill_wait_s"])
                    break
                if stop and stop_sent is None:
                    stop_file.write_text("stop\n", encoding="utf-8")
                    stop_sent, guard["action"] = now, "stop requested (interrupt)"
                if stop_sent is not None and now >= stop_sent + 2 * plan["kill_wait_s"]:
                    guard["action"] = "killed: interrupt grace expired"
                    kill_tree(proc, plan["kill_wait_s"])
                    break
                time.sleep(plan["poll_interval_s"])
            except KeyboardInterrupt:
                stop.append(True)
        popen.wait()
    t_end = time.monotonic()
    child = None
    try:
        p = out / RECORD_NAME[phase]
        child = K.load_json(p) if p.is_file() else None
    except (OSError, ValueError):
        child = None
    killed = (guard["action"] or "").startswith("killed")
    if child is not None and not killed and K.EXIT.get(child.get("status")) == popen.returncode:
        status = child["status"]
    elif guard["action"] in ("killed: RSS limit", "killed: hard deadline"):
        status = "RESOURCE_STOP"
    elif stop:
        status = "INCOMPLETE"
    else:
        status = "EXECUTION_ERROR"
    if status == "COMPLETED" and guard["action"] is not None:
        status = "INCOMPLETE"
    return {
        "phase": phase,
        "status": status,
        "child_exit_code": popen.returncode,
        "child_status": None if child is None else child.get("status"),
        "guard": guard,
        "wall_s": t_end - t_spawn,
        "limit_s": limit_s,
        "stderr_bytes": (out / f"{phase}.stderr.txt").stat().st_size,
        "_record": child,
    }


def state_ref(f: dict) -> dict[str, Any]:
    return {k: f[k] for k in ("path", "file_sha256", "state_sha256")}


def step_jobs(cfg: dict, got: dict) -> dict[str, Any]:
    u, o = cfg["ucx"], cfg["uoa"]
    forbid = [u["heldout_tiles"]["path"], *u["eval_outputs"], *o["eval_outputs"]]
    common = {
        "train_tiles": u["train_tiles"]["path"],
        "train_tiles_sha256": u["train_tiles"]["sha256"],
        "n_train_tiles": len(cfg["split"]["keys"]["TRAIN"]) * S.TILES_PER_SNAPSHOT,
        "scale_global": u["scale_global"],
        "model": cfg["model"],
        "source": o["source_state"],
    }
    jobs: dict[str, Any] = {
        "INIT": {
            **common,
            "diag_schedule_seed": cfg["diag_schedule_seed"],
            "diag_batches": cfg["diag_batches"],
            "batch_size": cfg["batch_size"],
            "forbidden_paths": forbid,
        }
    }
    if "INIT" in got:
        bi = got["INIT"]["bridge_init_file"]
        for arm in S.NEURAL_STEPS:
            jobs[arm] = {
                **common,
                "arm": arm,
                "c": S.ARMS[arm],
                "bridge_init": bi["path"],
                "bridge_init_sha256": bi["sha256"],
                "optimizer": cfg["optimizer"],
                "max_updates": cfg["max_updates"],
                "batch_size": cfg["batch_size"],
                "schedule_seed": cfg["schedule_seed"],
                "schedule_sha256": cfg["schedule_sha256"],
                "curve_every": cfg["curve_every"],
                "checkpoints": cfg["checkpoints"],
                "expected_normalized_sha256": u["normalized_sha256"],
                "source_updates": S.SOURCE_UPDATES,
                "forbidden_paths": forbid + [u["pca_decoder"]["path"]],
            }
    if set(S.NEURAL_STEPS) <= set(got):
        arms = {}
        for arm in S.NEURAL_STEPS:
            cks = {c["update"]: c for c in got[arm]["checkpoints"]}
            if (
                got[arm].get("updates_done") != cfg["max_updates"]
                or 0 not in cks
                or cfg["max_updates"] not in cks
            ):
                raise K.ULRError(f"{arm}: no step-zero or final state")
            arms[arm] = {
                "c": S.ARMS[arm],
                "states": {
                    "STEP0": state_ref(cks[0]["state_file"]),
                    "FINAL": state_ref(cks[cfg["max_updates"]]["state_file"]),
                },
            }
        jobs["EVAL"] = {
            **common,
            "heldout_tiles": u["heldout_tiles"]["path"],
            "heldout_tiles_sha256": u["heldout_tiles"]["sha256"],
            "train_mean_tile": u["train_mean_tile"]["path"],
            "train_mean_tile_sha256": u["train_mean_tile"]["sha256"],
            "pca_decoder": u["pca_decoder"]["path"],
            "pca_decoder_sha256": u["pca_decoder"]["sha256"],
            "int8_shared_step": u["int8_shared_step"],
            "scalar4_levels": o["scalar4_levels"]["path"],
            "scalar4_levels_sha256": o["scalar4_levels"]["sha256"],
            "source_scales": o["source_scales"]["path"],
            "source_scales_sha256": o["source_scales"]["sha256"],
            "arms": arms,
            "cohort_keys": cfg["split"]["keys"],
            "wrong_map": cfg["wrong_map"],
            "forbidden_paths": [*u["eval_outputs"], *o["eval_outputs"]],
        }
    return jobs


def fitted_quantities(got: dict, cfg: dict) -> dict[str, Any]:
    return {
        "bridge_init_sha256": got["INIT"]["bridge_init_file"]["sha256"],
        "arm_states": {
            arm: {str(c["update"]): c["state_file"]["file_sha256"] for c in got[arm]["checkpoints"]}
            for arm in S.NEURAL_STEPS
        },
        "source_state_file_sha256": cfg["uoa"]["source_state"]["file_sha256"],
        "pca_decoder_sha256": cfg["ucx"]["pca_decoder"]["sha256"],
        "scalar4_levels_sha256": cfg["uoa"]["scalar4_levels"]["sha256"],
        "source_scales_sha256": cfg["uoa"]["source_scales"]["sha256"],
        "statement": "fixed before the evaluation; no held-out tile was opened by any earlier step",
    }


def step_limit(step: str, plan: dict) -> float:
    lim = plan["limits"]
    return {"INIT": lim["init_s"], "EVAL": lim["eval_child_s"]}.get(step, lim["neural_arm_s"])


def invoke(cfg: dict, plan: dict, identity: dict) -> dict[str, Any]:
    run_root = Path(plan["run_root"])
    ledger = K.Ledger(run_root / "ledger", plan["ledger_mirror"])
    status = K.ledger_status(ledger.read())
    if status["invocations"] or any(v != "NOT_STARTED" for v in status["steps"].values()):
        raise K.ULRError(f"the ledger already records an invocation: {status}")
    if (run_root / "steps").exists():
        raise K.ULRError(f"{run_root / 'steps'} exists: earlier step output without a usable ledger state")
    plan["t0"] = T_START
    ledger.write(
        "invocation_001.json",
        {
            "protocol_id": plan["protocol_id"],
            "mode": plan["mode"],
            "identity": identity,
            "plan": plan,
            "wall_clock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "supervisor_pid": os.getpid(),
            "python": sys.executable,
            "counters_allowed": {
                "invocations": 1,
                "neural_starts": S.LIMITS["max_neural_starts"],
                "channel_pca_fits": 1,
            },
        },
    )
    out = {s: run_root / "steps" / s for s in S.STEPS}
    got: dict[str, Any] = {}
    ends: dict[str, Any] = {}
    stop: list = []
    failed = None
    counters = {"neural_starts": 0, "channel_pca_fits": 0}
    for step in S.STEPS:
        if failed is not None:
            ends[step] = {"step": step, "status": "NOT_RUN", "reason": f"{failed} did not complete"}
            continue
        try:
            job = step_jobs(cfg, got)[step]
        except Exception as exc:
            ends[step] = {
                "step": step,
                "status": "NOT_RUN",
                "reason": f"job not built: {type(exc).__name__}: {exc}",
            }
            failed = step
            continue
        limit_s = step_limit(step, plan)
        start = {
            "step": step,
            "monotonic_s": time.monotonic(),
            "limit_s": limit_s,
            "wall_clock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if step == "EVAL":
            start["fitted_quantities_frozen"] = fitted_quantities(got, cfg)
        if step in S.NEURAL_STEPS:
            avail = psutil.virtual_memory().available
            lim = min(int(plan["rss_cap_bytes"]), int(avail * plan["rss_available_fraction"]))
            start["memory_limit_bytes"] = lim
            if lim < int(plan["min_rss_limit_bytes"]):
                ends[step] = {"step": step, "status": "NOT_RUN", "reason": f"memory limit {lim} < floor"}
                failed = step
                continue
            counters["neural_starts"] += 1
            start["neural_start_number"] = counters["neural_starts"]
            if counters["neural_starts"] > S.LIMITS["max_neural_starts"]:
                raise K.ULRError("neural start counter exceeded")
        if step == "INIT":
            counters["channel_pca_fits"] += 1
        left = T_START + plan["aggregate_s"] - time.monotonic()
        if min(limit_s, left) <= plan["end_margin_s"] + plan["save_reserve_s"]:
            ends[step] = {"step": step, "status": "NOT_RUN", "reason": "no step or aggregate budget left"}
            failed = step
            continue
        ledger.write(f"start_{step}.json", start)
        out[step].mkdir(parents=True, exist_ok=False)
        try:
            end = run_child(S.STEP_PHASE[step], job, plan, out[step], limit_s, stop)
        except KeyboardInterrupt:
            stop.append(True)
            end = {
                "phase": S.STEP_PHASE[step],
                "status": "INCOMPLETE",
                "reason": "supervisor interrupted",
                "_record": None,
            }
        except Exception as exc:
            end = {
                "phase": S.STEP_PHASE[step],
                "status": "EXECUTION_ERROR",
                "reason": f"{type(exc).__name__}: {exc}",
                "_record": None,
            }
        rec = end.pop("_record")
        end.update(
            step=step,
            evidence_bytes_run_root=dir_bytes(run_root),
            aggregate_wall_s=time.monotonic() - T_START,
        )
        ledger.write(f"end_{step}.json", end)
        ends[step] = end
        if end["status"] == "COMPLETED" and rec is not None:
            got[step] = rec
        else:
            failed = step
    summary = {
        "aggregate_wall_s": time.monotonic() - T_START,
        "steps": {s: ends[s]["status"] for s in S.STEPS},
        "counters": counters,
        "evidence_bytes_run_root": dir_bytes(run_root),
        "evidence_within_cap": dir_bytes(run_root) <= plan["limits"]["evidence_cap_bytes"],
        "interrupted": bool(stop),
    }
    ledger.write("invocation_001_end.json", summary)
    summary["ledger_after"] = K.ledger_status(ledger.read())
    return summary


def registered_config_problems(cfg: dict) -> list[str]:
    expected = {
        "protocol_id": S.PROTOCOL_ID,
        "model": S.MODEL,
        "optimizer": S.OPTIMIZER,
        "max_updates": S.MAX_UPDATES,
        "checkpoints": list(S.CHECKPOINTS),
        "batch_size": S.BATCH_SIZE,
        "schedule_seed": S.SCHEDULE_SEED,
        "curve_every": S.CURVE_EVERY,
        "diag_schedule_seed": S.DIAG_SCHEDULE_SEED,
        "diag_batches": S.DIAG_BATCHES,
        "cpu": S.CPU_THREADS,
        "torch_version": S.TORCH_VERSION,
        "limits": S.LIMITS,
        "mechanics": S.MECHANICS,
        "arms": S.ARMS,
    }
    problems = [
        f"{k}: {cfg.get(k)!r} is not the registered value"
        for k, v in expected.items()
        if json.loads(json.dumps(cfg.get(k))) != json.loads(json.dumps(v))
    ]
    if cfg.get("execution", {}).get("run_root") != "D:/cgfed-runs/ulr_20261015_v1":
        problems.append("the run root is not the registered one")
    pol = cfg.get("primary_policy") or {}
    vdir = PKG / "vendor" / "uoa_20261014_v1" / "vendor" / "upp_20261014_v1"
    if pol.get("policy_sha256") != K.sha256_file(vdir / "upp_policy.json") or pol.get(
        "classifier_sha256"
    ) != K.sha256_file(vdir / "upp_20261014_classify.py"):
        problems.append("the vendored primary policy or classifier differs from its pinned SHA-256")
    return problems


def plan_from(cfg: dict, mode: str) -> dict[str, Any]:
    ex, lim = cfg["execution"], cfg["limits"]
    return {
        "mode": mode,
        "protocol_id": cfg["protocol_id"],
        "cpu": cfg["cpu"],
        "torch_version": cfg["torch_version"],
        "limits": lim,
        "aggregate_s": lim["aggregate_s"]
        - cfg["preflight"]["pilot_wall_s_total"]
        - lim["verification_reserve_s"],
        "rss_cap_bytes": lim["rss_cap_bytes"],
        "rss_available_fraction": lim["rss_available_fraction"],
        "min_rss_limit_bytes": cfg["preflight"]["min_rss_limit_bytes"],
        "run_root": ex["run_root"],
        "ledger_mirror": ex["ledger_mirror"],
        "python_executable": ex["python_executable"],
        **cfg["mechanics"],
    }


def verify_package(sha: str) -> tuple[dict, dict]:
    mpath = PKG / MANIFEST_NAME
    actual = K.sha256_file(mpath) if mpath.is_file() else None
    if actual != sha:
        raise K.ULRError(f"{MANIFEST_NAME} SHA-256 {actual} is not the frozen {sha}")
    manifest = K.load_json(mpath)
    bad = [
        e["path"]
        for e in manifest["files"]
        if not (PKG / e["path"]).is_file() or K.sha256_file(PKG / e["path"]) != e["sha256"]
    ]
    bad += [
        e["path"]
        for e in manifest["external_bound_files"]
        if not Path(e["path"]).is_file() or K.sha256_file(e["path"]) != e["sha256"]
    ]
    if bad:
        raise K.ULRError(f"package identity check failed: {bad}")
    cfg = K.load_json(PKG / CONFIG_NAME)
    problems = registered_config_problems(cfg)
    if problems:
        raise K.ULRError(f"the config is not the registered configuration: {problems}")
    for name, p in cfg["preflight"]["pilots"].items():
        rec = K.load_json(PKG / p["record"])
        if rec.get("gate", {}).get("decision") != "GO" or K.sha256_file(PKG / p["record"]) != p["sha256"]:
            raise K.ULRError(f"the resource pilot record {name} is not the pinned GO record")
    return {
        "ulr_manifest_sha256": actual,
        "ulr_config_sha256": K.sha256_file(PKG / CONFIG_NAME),
        "pilots": cfg["preflight"]["pilots"],
    }, cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--scientific", action="store_true")
    mode.add_argument("--fixture")
    ap.add_argument("--frozen-ulr-manifest-sha256")
    args = ap.parse_args()
    if args.status:
        ex = K.load_json(PKG / CONFIG_NAME)["execution"]
        s = K.ledger_status(K.Ledger(Path(ex["run_root"]) / "ledger", ex["ledger_mirror"]).read())
        s.update(run_root_exists=Path(ex["run_root"]).exists(), protocol_id=S.PROTOCOL_ID)
        print(json.dumps(s, indent=2, sort_keys=True))
        return 0
    if os.environ.get("PYTHONPATH"):
        raise K.ULRError("PYTHONPATH is set; ULR runs only without it")
    if args.fixture:
        cfg = K.load_json(args.fixture)
        if cfg.get("protocol_id") != S.FIXTURE_PROTOCOL_ID:
            raise K.ULRError("fixture mode accepts only fixture configs")
        if "cgfed-runs" in cfg["execution"]["run_root"] or "cgfed-runs" in json.dumps(
            [cfg["ucx"], cfg["uoa"]]
        ):
            raise K.ULRError("fixture configs must not use the run area or the registered artifacts")
        plan = plan_from(cfg, "engineering")
        plan["fixture_hooks"] = cfg.get("fixture_hooks") or {}
        summary = invoke(cfg, plan, {"fixture_config_sha256": K.sha256_file(args.fixture)})
    else:
        if not args.frozen_ulr_manifest_sha256:
            raise K.ULRError("--scientific requires --frozen-ulr-manifest-sha256")
        identity, cfg = verify_package(args.frozen_ulr_manifest_sha256)
        if Path(sys.executable).resolve() != Path(cfg["execution"]["python_executable"]).resolve():
            raise K.ULRError(f"interpreter {sys.executable} is not {cfg['execution']['python_executable']}")
        summary = invoke(cfg, plan_from(cfg, "scientific"), identity)
    print(json.dumps(K.json_safe(summary), indent=2, sort_keys=True))
    return 0 if all(v == "COMPLETED" for v in summary["steps"].values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except K.ULRError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
