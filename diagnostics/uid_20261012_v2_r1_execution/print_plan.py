"""Read-only: print the registered plan, deadlines and module hashes the r1 supervisor would use.

Not part of the approved package. Verifies the package through the package's own ``verify_package`` and writes nothing.
"""

import json
import sys
from pathlib import Path

R1 = Path(__file__).resolve().parents[1] / "uid_20261012_v2_r1"
sys.path.insert(0, str(R1))

import psutil  # noqa: E402
import uid_r1_control as K  # noqa: E402

APPROVED = "9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044"

ident, cfg, execution = K.verify_package(APPROVED)
plan = K.registered_plan(cfg, execution)
K.validate_plan(plan)
keys = (
    "arms", "prerequisite", "max_training_starts", "max_invocations", "aggregate_seconds", "rss_cap_bytes",
    "rss_available_fraction", "evidence_cap_bytes", "curve_every", "cpu", "end_margin_s", "eval_reserve_s",
    "poll_interval_s", "interrupt_grace_s", "kill_wait_s", "python_executable", "torch_version", "run_root",
)
print("identity", json.dumps(ident, sort_keys=True))
print("plan", json.dumps({k: plan[k] for k in keys}, sort_keys=True))
for a in plan["arms"]:
    soft, kill = K.arm_deadlines(0.0, 0.0, a["max_seconds"], plan["aggregate_seconds"], plan["end_margin_s"],
                                 plan["eval_reserve_s"])
    print(a["arm"], "soft_stop_after_s", soft, "kill_at_s", kill, "(from its start record, aggregate not binding)")
avail = psutil.virtual_memory().available
print("available_bytes_now", avail, "rss_limit_now", min(plan["rss_cap_bytes"], int(avail * plan["rss_available_fraction"])))
print("module_hashes", json.dumps(K.expected_module_hashes(K.load_json(R1 / "manifest_r1.json")), sort_keys=True))
