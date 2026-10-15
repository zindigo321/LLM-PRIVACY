"""ULR-20261015-v1 resource pilot runner: one synthetic pilot of one arm, with a gate fixed in this file before any
measurement.

``python -B preflight/run_resource_pilot.py <arm> full`` writes ``preflight/pilot_<arm>_record.json`` (refused if it
exists: one pilot per arm). ``... <arm> fixture <out_dir>`` is an engineering test of this runner with a small model,
written to ``out_dir``; it is never evidence about the budget.

The child runs with PYTHONPATH removed and the registered child environment. It is killed at 120 s of wall clock or
when its working set exceeds min(4 GiB, 50% of the available RAM before its launch).

**Gate (GO only if every condition holds):**
1. the child exited 0 within 120 s, every loss was finite, and every payload had 64 c bytes;
2. the projected arm is <= 1,800 - 20 - 60 = 1,720 s. The projection is 1.15 x (6,000 x the p90 update + 61 curve
   points x the slowest curve point + 4 checkpoints x (evaluation + 2 saves));
3. the projected EVAL is <= 1,500 - 20 = 1,480 s. The projection is 1.15 x 1,944 x (6 arm states x (send + 3
   receives) + the source (float + 3 decodes));
4. 1.25 x the peak working set is <= the memory limit at launch. That value becomes a floor of the neural steps.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

PRE = Path(__file__).resolve().parent
PKG = PRE.parent
sys.path.insert(0, str(PKG))
import ulr_common as K  # noqa: E402
import ulr_spec as S  # noqa: E402

PY = "D:/conda_envs/cgfedllm/python.exe"
HEADROOM, MEM_FACTOR = 1.15, 1.25


def gate(child: dict | None, exit_code: int | None, wall_s: float, rss_limit: int, peak: int, c: int) -> dict:
    out: dict = {"conditions": {}}
    cnd = out["conditions"]
    cnd["exit_0_within_wall"] = exit_code == 0 and wall_s <= S.LIMITS["pilot_s"] and child is not None
    if not cnd["exit_0_within_wall"]:
        out["decision"] = "NO-GO"
        return out
    cnd["losses_finite"] = bool(child["steps"]["all_losses_finite"]) and child["steps"]["completed"] == 25
    cnd["payload_bytes"] = child["payload_bytes"] == [S.wire_bytes(c)]
    t_up, t_curve = child["steps"]["measured_s"]["p90_nearest_rank"], child["curve_point_s"]["max"]
    t_ck = child["checkpoint_eval_1440_s"] + 2 * child["state_save_s"]
    arm = HEADROOM * (S.MAX_UPDATES * t_up + 61 * t_curve + len(S.CHECKPOINTS) * t_ck)
    arm_budget = S.LIMITS["neural_arm_s"] - S.MECHANICS["end_margin_s"] - S.MECHANICS["save_reserve_s"]
    ev = (
        HEADROOM
        * 1944
        * (
            6 * (child["send_per_tile_s"] + 3 * child["receive_per_tile_s"])
            + 2 * child["source_float_per_tile_s"]
        )
    )
    ev_budget = S.LIMITS["eval_child_s"] - S.MECHANICS["end_margin_s"]
    floor = math.ceil(MEM_FACTOR * peak)
    out["projection"] = {"update_p90_s": t_up, "curve_point_max_s": t_curve, "checkpoint_s": t_ck, "neural_arm_projected_s": arm,
                         "neural_arm_budget_s": arm_budget, "eval_projected_s": ev, "eval_budget_s": ev_budget, "peak_working_set_bytes": peak,
                         "memory_floor_bytes": floor, "memory_limit_at_launch_bytes": rss_limit}  # fmt: skip
    cnd["neural_arm_fits"], cnd["eval_fits"], cnd["memory_fits"] = (
        arm <= arm_budget,
        ev <= ev_budget,
        floor <= rss_limit,
    )
    out["decision"] = "GO" if all(cnd.values()) else "NO-GO"
    return out


def main(arm: str, mode: str, out_dir: str | None = None) -> int:
    if arm not in S.ARMS or mode not in ("full", "fixture"):
        raise SystemExit(
            "usage: run_resource_pilot.py C16-CONTROL|C4-REDUCED|C1-REDUCED full|fixture [out_dir]"
        )
    if mode == "full":
        out, child_json = PRE / f"pilot_{arm}_record.json", PRE / f"pilot_{arm}_child.json"
    else:
        if not out_dir or Path(out_dir).resolve().is_relative_to(PRE):
            raise SystemExit("fixture mode needs an out_dir outside preflight/")
        out, child_json = (
            Path(out_dir) / f"fixture_pilot_{arm}_record.json",
            Path(out_dir) / f"fixture_pilot_{arm}_child.json",
        )
    if out.exists() or child_json.exists():
        raise SystemExit(f"{out} or {child_json} exists: one pilot per arm, never repeated")
    if mode == "full":
        with open(PRE / f"pilot_{arm}_started.txt", "x", encoding="utf-8") as fh:
            fh.write(f"started {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(K.CHILD_ENV)
    avail = psutil.virtual_memory().available
    rss_limit = min(int(S.LIMITS["rss_cap_bytes"]), int(avail * S.LIMITS["rss_available_fraction"]))
    cmd = [PY, "-B", str(PRE / "resource_pilot.py"), arm, mode, str(child_json)]
    t0 = time.monotonic()
    so, se = (
        (out.parent / f"{out.stem}.stdout.txt").open("xb"),
        (out.parent / f"{out.stem}.stderr.txt").open("xb"),
    )
    p = subprocess.Popen(cmd, cwd=str(PKG), env=env, stdout=so, stderr=se)
    proc, peak, action = psutil.Process(p.pid), 0, None
    while p.poll() is None:
        try:
            peak = max(peak, proc.memory_info().rss)
        except psutil.NoSuchProcess:
            break
        if peak > rss_limit:
            action = "killed: RSS limit"
            proc.kill()
            break
        if time.monotonic() - t0 > S.LIMITS["pilot_s"]:
            action = "killed: wall limit"
            proc.kill()
            break
        time.sleep(0.25)
    p.wait()
    so.close()
    se.close()
    wall = time.monotonic() - t0
    child = json.loads(child_json.read_text("utf-8")) if child_json.is_file() else None
    os_peak = (child or {}).get("memory", {}).get("os_peak_wset_bytes") or 0
    rec = {"protocol_id": S.PROTOCOL_ID, "arm": arm, "mode": mode, "command": cmd, "exit_code": p.returncode, "wall_s": wall, "guard_action": action,
           "available_before_bytes": avail, "rss_limit_bytes": rss_limit, "sampled_peak_rss_bytes": peak,
           "child_record_sha256": K.sha256_file(child_json) if child_json.is_file() else None, "child": child,
           "gate_rule": "see the module docstring of run_resource_pilot.py (fixed before measurement)"}  # fmt: skip
    rec["gate"] = gate(child, p.returncode, wall, rss_limit, max(peak, os_peak), S.ARMS[arm])
    K.write_json(out, rec)
    print(
        json.dumps(
            {
                "decision": rec["gate"]["decision"],
                "wall_s": wall,
                "projection": rec["gate"].get("projection"),
            },
            indent=1,
        )
    )
    return 0 if rec["gate"]["decision"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
