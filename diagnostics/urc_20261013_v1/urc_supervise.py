"""URC-20261013-v1 supervisor: one invocation, the single arm R-FIT, one fresh child process (adapted from r1).

Scientific use (only after the URC package is approved):
    python -B urc_supervise.py --scientific --approved-urc-manifest-sha256 <sha256>
Read-only:
    python -B urc_supervise.py --status

The plan comes only from ``urc_control.registered_plan``, which refuses anything but one R-FIT start; there is no
option that selects an arm, so U-CHECK or U-FIT cannot be launched from here.

The supervisor imports no torch. Before each arm it writes the start record (both ledger roots) and only then spawns
the child. It enforces, on one ``time.monotonic()`` clock: the arm limit from the arm's start record, the aggregate
limit from the invocation record, and the RSS cap (sampled every ``poll_interval_s``). A soft stop is given to the
child; at the hard deadline, or when the RSS cap is exceeded, the child is killed. Ctrl+C asks the child to stop
(stop file -> KeyboardInterrupt in the child) and kills it after ``interrupt_grace_s``. Every arm that was started gets
an end record; if the supervisor itself dies, the start record without an end record marks the arm INCOMPLETE and
every later invocation is refused.
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

import psutil

# the aggregate clock starts here, at import, before package verification, ledger reads and any arm
T_START = time.monotonic()
URC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(URC_DIR))

import urc_control as U  # noqa: E402

K = U.K  # the r1 ledger / start accounting / deadline module, unchanged and hash-pinned


def dir_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def kill_tree(proc: psutil.Process, wait_s: float) -> None:
    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.NoSuchProcess:  # the child exited between the poll and the kill
        procs = [proc]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=wait_s)


def read_child_record(path: Path) -> dict[str, Any] | None:
    try:
        return K.load_json(path) if path.is_file() else None
    except (OSError, ValueError):
        return None


def run_one(
    plan: dict,
    arm_spec: dict,
    t0: float,
    t_start: float,
    config_path: Path,
    sys_path: list[str],
    hooks: dict | None,
    rss_limit: int,
    stop_requested: list,
) -> dict[str, Any]:
    """Run one arm whose start record is already written; returns the end-record fields."""
    arm, run_root = arm_spec["arm"], Path(plan["run_root"])
    out = run_root / "arms" / arm
    out.mkdir(parents=True, exist_ok=False)
    soft_at, kill_at = K.arm_deadlines(
        t0,
        t_start,
        arm_spec["max_seconds"],
        plan["aggregate_seconds"],
        plan["end_margin_s"],
        plan["eval_reserve_s"],
    )
    stop_file = out / "STOP_REQUEST"
    job = {
        "arm": arm,
        "model_arm": arm_spec.get("model_arm", arm),
        "max_updates": arm_spec["max_updates"],
        "expected_torch_version": plan["torch_version"],
        "mode": plan["mode"],
        "config_path": str(config_path),
        "out_dir": str(out),
        "sys_path": sys_path,
        "cpu": plan["cpu"],
        "curve_every": plan["curve_every"],
        "soft_stop_monotonic": soft_at,
        "hard_stop_monotonic": kill_at + plan["kill_wait_s"],
        "poll_interval_s": plan["poll_interval_s"],
        "supervisor_pid": os.getpid(),
        "supervisor_create_time": psutil.Process(os.getpid()).create_time(),
        "stop_file": str(stop_file),
        "fixture_hooks": hooks,
        "expected_module_sha256": plan.get("expected_module_sha256"),
        "expected_config_sha256": plan.get("expected_config_sha256"),
    }
    K.create_exclusive(out / "job.json", job)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    cmd = [plan["python_executable"], "-B", str(URC_DIR / "urc_child.py"), str(out / "job.json")]
    with (out / "child.stdout.txt").open("xb") as so, (out / "child.stderr.txt").open("xb") as se:
        popen = subprocess.Popen(
            cmd,
            cwd=str(URC_DIR),
            env=env,
            stdout=so,
            stderr=se,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        try:
            proc = psutil.Process(popen.pid)
        except psutil.NoSuchProcess:  # the child already exited; its exit code and record decide the status
            proc = None
        guard = {
            "rss_limit_bytes": rss_limit,
            "poll_interval_s": plan["poll_interval_s"],
            "samples": 0,
            "sampled_max_rss_bytes": 0,
            "action": None,
            "action_monotonic_s": None,
        }
        interrupt_at = None
        while True:
            try:
                if not supervise_step(
                    popen, proc, guard, plan, kill_at, rss_limit, stop_file, stop_requested, interrupt_at
                ):
                    break
                if stop_requested and interrupt_at is None:
                    interrupt_at = guard["action_monotonic_s"]
                time.sleep(plan["poll_interval_s"])
            except KeyboardInterrupt:  # Ctrl+C / Ctrl+Break: ask the child to stop, keep supervising
                stop_requested.append(True)
        popen.wait()
    t_end = time.monotonic()
    rc = popen.returncode
    child = read_child_record(out / "child_record.json")
    killed = guard["action"] is not None and guard["action"].startswith("killed")
    if child is not None and not killed and K.CHILD_EXIT.get(child.get("status")) == rc:
        status = child["status"]
    elif guard["action"] in ("killed: RSS limit", "killed: hard deadline"):
        status = "RESOURCE_STOP"
    elif stop_requested:
        status = "INCOMPLETE"
    else:
        status = "EXECUTION_ERROR"
    return {
        "arm": arm,
        "status": status,
        "child_exit_code": rc,
        "child_status": None if child is None else child.get("status"),
        "child_record_present": child is not None,
        "guard": guard,
        "monotonic_s": {
            "invocation_t0": t0,
            "start": t_start,
            "end": t_end,
            "soft_stop": soft_at,
            "kill_at": kill_at,
        },
        "arm_wall_s": t_end - t_start,
        "aggregate_wall_s": t_end - t0,
        "updates_done": None if child is None else child.get("updates_done"),
    }


def supervise_step(
    popen, proc, guard, plan, kill_at, rss_limit, stop_file, stop_requested, interrupt_at
) -> bool:
    """One poll of the child. Returns False when supervision of this child is over (exited or killed)."""
    if popen.poll() is not None or proc is None:
        return False
    now = time.monotonic()
    try:
        rss = sum(p.memory_info().rss for p in [proc, *proc.children(recursive=True)])
        guard["samples"] += 1
        guard["sampled_max_rss_bytes"] = max(guard["sampled_max_rss_bytes"], rss)
    except psutil.NoSuchProcess:
        rss = 0
    if rss > rss_limit:
        guard.update(action="killed: RSS limit", action_monotonic_s=now)
        kill_tree(proc, plan["kill_wait_s"])
        return False
    if now >= kill_at:
        guard.update(action="killed: hard deadline", action_monotonic_s=now)
        kill_tree(proc, plan["kill_wait_s"])
        return False
    if stop_requested and interrupt_at is None:
        stop_file.write_text("stop\n", encoding="utf-8")
        guard.update(action="stop requested (interrupt)", action_monotonic_s=now)
    if interrupt_at is not None and now >= interrupt_at + plan["interrupt_grace_s"]:
        guard.update(action="killed: interrupt grace expired", action_monotonic_s=now)
        kill_tree(proc, plan["kill_wait_s"])
        return False
    return True


def invoke(
    plan: dict, config_path: Path, sys_path: list[str], identity: dict, hooks: dict | None = None
) -> dict[str, Any]:
    """The whole invocation. Refuses (raises ControlError) before writing anything if the ledger forbids it."""
    import signal

    if hasattr(signal, "SIGBREAK"):  # Ctrl+Break would otherwise end the process without the end records
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    U.check_plan(plan)  # one R-FIT start in scientific mode; no UID arm in any mode; r1 timing invariants
    run_root = Path(plan["run_root"])
    ledger = K.Ledger(run_root / "ledger", plan["ledger_mirror"])
    K.check_can_open(ledger.read(), plan)  # an unfinished prior start or a used invocation refuses here
    if (run_root / "arms").exists():
        raise K.ControlError(f"{run_root / 'arms'} exists: earlier arm output without a usable ledger state")
    t0 = (
        T_START  # supervisor import: setup, verification and ledger reads are charged to the aggregate budget
    )
    avail = psutil.virtual_memory().available
    rss_limit = min(int(plan["rss_cap_bytes"]), int(avail * plan["rss_available_fraction"]))
    ledger.write(
        "invocation_001.json",
        {
            "protocol_id": U.PROTOCOL_ID,
            "mode": plan["mode"],
            "identity": identity,
            "plan": plan,
            "available_ram_bytes_at_launch": avail,
            "rss_limit_bytes": rss_limit,
            "monotonic_t0": t0,
            "aggregate_clock_start": "import of urc_supervise (before package verification)",
            "monotonic_at_record": time.monotonic(),
            "wall_clock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "supervisor_pid": os.getpid(),
            "python": sys.executable,
        },
    )
    stop_requested: list = []
    results = []
    try:
        for spec in plan["arms"]:
            records = ledger.read()
            try:
                K.check_can_start(records, plan, spec["arm"])
            except K.ControlError as exc:
                results.append({"arm": spec["arm"], "status": "NOT_RUN", "reason": str(exc)})
                continue
            if (
                time.monotonic() + plan["eval_reserve_s"] + plan["end_margin_s"]
                >= t0 + plan["aggregate_seconds"]
            ):
                results.append(
                    {"arm": spec["arm"], "status": "NOT_RUN", "reason": "aggregate budget exhausted"}
                )
                continue
            t_start = time.monotonic()  # the arm limit runs from this, the time stored in the start record
            ledger.write(
                f"start_{spec['arm']}.json",
                {
                    "arm": spec["arm"],
                    "monotonic_s": t_start,
                    "max_updates": spec["max_updates"],
                    "max_seconds": spec["max_seconds"],
                },
            )
            end = None
            try:
                end = run_one(
                    plan, spec, t0, t_start, config_path, sys_path, hooks, rss_limit, stop_requested
                )
            except KeyboardInterrupt:
                stop_requested.append(True)
                end = {"arm": spec["arm"], "status": "INCOMPLETE", "reason": "supervisor interrupted"}
            except Exception as exc:
                end = {
                    "arm": spec["arm"],
                    "status": "EXECUTION_ERROR",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            end["evidence_bytes_run_root"] = dir_bytes(run_root)
            ledger.write(f"end_{spec['arm']}.json", end)
            results.append(end)
            if stop_requested:
                break
    except KeyboardInterrupt:
        stop_requested.append(True)
    summary = {
        "t0": t0,
        "aggregate_wall_s": time.monotonic() - t0,
        "arms": results,
        "evidence_bytes_run_root": dir_bytes(run_root),
        "evidence_within_cap": dir_bytes(run_root) <= plan["evidence_cap_bytes"],
        "interrupted": bool(stop_requested),
    }
    ledger.write("invocation_001_end.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true", help="read-only: print the scientific ledger state")
    mode.add_argument("--scientific", action="store_true", help="run the registered R-FIT invocation")
    ap.add_argument("--approved-urc-manifest-sha256")
    args = ap.parse_args()
    if args.status:
        execution = U.load_urc_config()["execution"]
        primary = Path(execution["run_root"]) / "ledger"
        s = K.ledger_status(primary, execution["ledger_mirror"], list(U.REGISTERED_ARMS))
        s["ledger_primary"], s["ledger_mirror"] = str(primary), execution["ledger_mirror"]
        s["run_root_exists"] = Path(execution["run_root"]).exists()
        s["protocol_id"] = U.PROTOCOL_ID
        print(json.dumps(s, indent=2, sort_keys=True))
        return 0
    if not args.approved_urc_manifest_sha256:
        raise K.ControlError("--scientific requires --approved-urc-manifest-sha256")
    if os.environ.get("PYTHONPATH"):
        raise K.ControlError("PYTHONPATH is set; the scientific invocation runs only without it")
    identity, resolved = U.verify_package(args.approved_urc_manifest_sha256)
    if Path(sys.executable).resolve() != Path(resolved["execution"]["python_executable"]).resolve():
        raise K.ControlError(
            f"interpreter {sys.executable} is not {resolved['execution']['python_executable']}"
        )
    plan = U.registered_plan(resolved)
    plan["expected_module_sha256"] = U.expected_module_hashes(K.load_json(URC_DIR / U.MANIFEST_NAME))
    plan["expected_config_sha256"] = identity["resolved_config_sha256"]
    sys_path = [str(URC_DIR), str(U.FROZEN_DIR), str(U.R1_DIR)]
    summary = invoke(plan, URC_DIR / U.RESOLVED_NAME, sys_path, identity)
    print(json.dumps(K.json_safe(summary), indent=2, sort_keys=True))
    return (
        0
        if all(a["status"] in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL", "NOT_RUN") for a in summary["arms"])
        else 1
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except K.ControlError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
