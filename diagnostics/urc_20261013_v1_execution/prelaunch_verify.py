"""Pre-launch verification of URC-20261013-v1 (read-only; writes nothing). Run from this directory:
``python -B prelaunch_verify.py``. Recomputes the manifest hash and every entry, re-runs the package identity and the
plan, exercises the child's scientific config-loading branch with the job the supervisor would write, and records
the interpreter, environment, memory and disk state.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

APPROVED = "b058adf35bf326516f17cb3d773222840b0d1f6605a0f6e4a535564e94588cb6"
URC = Path(__file__).resolve().parents[1] / "urc_20261013_v1"
sys.path.insert(0, str(URC))


def main() -> int:
    raw = (URC / "manifest_urc.json").read_bytes()
    m = json.loads(raw)
    bad = [
        e["path"]
        for e in m["files"]
        if (lambda b: (hashlib.sha256(b).hexdigest(), len(b)) != (e["sha256"], e["bytes"]))(
            (URC / e["path"]).resolve().read_bytes()
        )
    ]
    listed = {(URC / e["path"]).resolve() for e in m["files"]}
    out = {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_is_approved": hashlib.sha256(raw).hexdigest() == APPROVED,
        "entries": len(m["files"]),
        "entries_mismatching": bad,
        "package_files_not_in_manifest": sorted(
            p.relative_to(URC).as_posix() for p in URC.rglob("*") if p.is_file() and p.resolve() not in listed
        ),
    }
    import urc_child as CH
    import urc_control as U

    U.check_pinned()
    ident, resolved = U.verify_package(APPROVED)
    plan = U.registered_plan(resolved)
    out["identity"] = ident
    out["plan"] = {k: plan[k] for k in ("arms", "max_training_starts", "max_invocations", "aggregate_seconds")}
    out["plan"]["run_root"], out["plan"]["ledger_mirror"] = plan["run_root"], plan["ledger_mirror"]
    out["plan"]["python_executable"], out["plan"]["cpu"] = plan["python_executable"], plan["cpu"]
    out["expected_module_sha256"] = U.expected_module_hashes(m)
    job = {
        "mode": "scientific",
        "config_path": str(URC / U.RESOLVED_NAME),
        "expected_config_sha256": ident["resolved_config_sha256"],
        "arm": U.ARM,
        "model_arm": U.ARM,
    }
    cfg, raw_sha, _ = CH.load_job_config(job, U.K, U)
    out["child_scientific_config_branch"] = {"accepted": True, "raw_sha256": raw_sha, "arm": cfg["arm"]}
    import psutil

    vm = psutil.virtual_memory()
    out["interpreter"] = {"sys.executable": sys.executable, "matches_plan": Path(sys.executable).resolve()
                          == Path(plan["python_executable"]).resolve(), "version": sys.version.split()[0]}
    out["environment"] = {k: os.environ.get(k) for k in
                          ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "PYTHONDONTWRITEBYTECODE")}
    out["memory"] = {"available_bytes": vm.available, "rss_limit_at_launch_would_be": min(
        plan["rss_cap_bytes"], int(vm.available * plan["rss_available_fraction"]))}
    out["disk_free_bytes"] = {"D:": shutil.disk_usage("D:/").free}
    out["torch_imported"] = "torch" in sys.modules
    print(json.dumps(out, indent=2, sort_keys=True))
    ok = out["manifest_is_approved"] and not bad and out["interpreter"]["matches_plan"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
