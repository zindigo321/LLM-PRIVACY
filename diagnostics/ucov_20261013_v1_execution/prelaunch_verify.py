"""Pre-launch verification of UCOV-20261013-v1 (read-only on the package, the run root and the ledger mirror).

``python -B prelaunch_verify.py`` from this directory, with PYTHONPATH unset. It writes only
``prelaunch_verify.out.txt`` here (exclusive, LF bytes).
1. Manifest bytes = the frozen digest; every entry and every external file (SHA-256 and size); unlisted files.
2. ``ucov_control.check_pinned`` / ``verify_package`` / ``registered_plan`` (the supervisor's own functions).
3. The child's scientific config branch (``ucov_child.load_job_config``) with the job fields the supervisor writes.
4. Import resolution on the child's and the supervisor's sys.path, without importing; ``cg_fedllm`` located only.
5. Dependency versions against the manifest; torch imported last, for its version and CUDA availability.
6. Interpreter, environment, ledger status, run root and mirror, memory, disk, bytecode before and after.
"""

import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path

BASE_PATH = list(sys.path[1:])
FROZEN = "3bc2ab302f6cd77b63b9bb8b77894041fd97cbdd63d5fa7f380a4dae080fe824"
HERE = Path(__file__).resolve().parent
UCOV = HERE.parent / "ucov_20261013_v1"
ENV_KEYS = ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "PYTHONDONTWRITEBYTECODE")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ok_file(path: Path, digest: str, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size and sha(path.read_bytes()) == digest


def bytecode() -> list[str]:
    roots = [
        UCOV,
        *(
            HERE.parent / d
            for d in (
                "upp_20261013_v1",
                "urp_20261013_v1",
                "uid_20261012_v2",
                "uid_20261012_v2_r1",
                "exploratory_progression_20261012",
            )
        ),
    ]
    return sorted(
        str(p) for r in roots for p in r.rglob("*") if p.name == "__pycache__" or p.suffix == ".pyc"
    )


def resolve_on(path_list: list[str], names: list[str], expected: dict[str, str]) -> dict:
    out = {}
    for name in names:
        spec = importlib.machinery.PathFinder.find_spec(name, path_list)
        origin = None if spec is None else spec.origin
        digest = sha(Path(origin).read_bytes()) if origin and Path(origin).is_file() else None
        out[name] = {"origin": origin, "sha256": digest, "approved": digest == expected.get(name)}
    return out


def verify(out: dict) -> None:
    raw = (UCOV / "manifest_ucov.json").read_bytes()
    m = json.loads(raw)
    bad = [
        e["path"] for e in m["files"] if not ok_file((UCOV / e["path"]).resolve(), e["sha256"], e["bytes"])
    ]
    ext = [
        e["path"] for e in m["external_bound_files"] if not ok_file(Path(e["path"]), e["sha256"], e["bytes"])
    ]
    listed = {(UCOV / e["path"]).resolve() for e in m["files"]}
    unlisted = sorted(
        p.relative_to(UCOV).as_posix() for p in UCOV.rglob("*") if p.is_file() and p.resolve() not in listed
    )
    out["manifest"] = {
        "sha256": sha(raw),
        "bytes": len(raw),
        "is_frozen": sha(raw) == FROZEN,
        "entries": len(m["files"]),
        "entries_mismatching": bad,
        "external": len(m["external_bound_files"]),
        "external_mismatching": ext,
        "unlisted": unlisted,
    }
    sys.path.insert(0, str(UCOV))
    import ucov_child as CH
    import ucov_control as U

    K = U.K
    U.check_pinned()
    ident, resolved = U.verify_package(FROZEN)
    plan = U.registered_plan(resolved)
    modules = U.expected_module_hashes(m)
    out["identity"] = ident
    out["plan"] = {
        k: plan[k]
        for k in (
            "mode",
            "arms",
            "prerequisite",
            "max_training_starts",
            "max_invocations",
            "aggregate_seconds",
            "rss_cap_bytes",
            "rss_available_fraction",
            "evidence_cap_bytes",
            "curve_every",
            "cpu",
            "run_root",
            "ledger_mirror",
            "python_executable",
            "primary_policy_id",
            "primary_policy_sha256",
            *U.MECHANICS_KEYS,
        )
    }
    soft, kill = K.arm_deadlines(
        0.0, 0.0, 1800.0, plan["aggregate_seconds"], plan["end_margin_s"], plan["eval_reserve_s"]
    )
    out["plan"]["first_arm_deadlines_s"] = {"soft_stop": soft, "kill": kill}
    out["primary_policy"] = resolved["primary_policy"]
    out["secondary_references"] = resolved["secondary_references"]
    out["expected_module_sha256"] = modules
    job = {
        "mode": "scientific",
        "config_path": str(U.UCOV_DIR / U.RESOLVED_NAME),
        "expected_config_sha256": ident["resolved_config_sha256"],
        "arm": "U-TILE-1",
        "model_arm": "U-TILE-1",
    }
    cfg, raw_sha, canon_sha = CH.load_job_config(job, K, U)
    out["child_scientific_config_branch"] = {
        "accepted": True,
        "raw_sha256": raw_sha,
        "canonical_sha256": canon_sha,
    }
    child_path = [
        str(UCOV),
        str(U.URP_DIR),
        str(U.UPP_DIR),
        str(U.FROZEN_DIR),
        str(U.R1_DIR),
        str(UCOV),
        *BASE_PATH,
    ]
    sup_path = [
        str(UCOV),
        str(UCOV),
        *BASE_PATH,
        str(U.R1_DIR),
        str(U.FROZEN_DIR),
        str(U.URP_DIR),
        str(U.UPP_DIR),
    ]
    child = resolve_on(child_path, [*CH.MODULES, "ucov_child"], modules)
    sup = resolve_on(sup_path, ["ucov_control", "uid_r1_control", "upp_classify", "urp_control"], modules)
    cg = importlib.machinery.PathFinder.find_spec("cg_fedllm", child_path)
    out["imports"] = {
        "child": child,
        "child_all_approved": all(v["approved"] for v in child.values()),
        "supervisor": sup,
        "supervisor_all_approved": all(v["approved"] for v in sup.values()),
        "cg_fedllm_would_resolve_to (forbidden in the child; not imported)": None
        if cg is None
        else cg.origin,
    }
    out["torch_imported_by_package_checks"] = "torch" in sys.modules
    want = {k: m["dependencies"][k] for k in ("torch", "safetensors", "psutil", "numpy", "python")}
    have = {d: importlib.metadata.version(d) for d in ("torch", "safetensors", "psutil", "numpy")}
    have["python"] = sys.version.split()[0]
    import psutil
    import torch

    out["dependencies"] = {
        "manifest": want,
        "installed": have,
        "all_equal": have == want,
        "torch.__version__": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    ex = resolved["execution"]
    status = K.ledger_status(Path(ex["run_root"]) / "ledger", ex["ledger_mirror"], list(U.ARMS))
    status["run_root_exists"] = Path(ex["run_root"]).exists()
    out["ledger_status"] = status
    vm = psutil.virtual_memory()
    out["interpreter"] = {
        "sys.executable": sys.executable,
        "version": sys.version.split()[0],
        "matches_plan": Path(sys.executable).resolve() == Path(plan["python_executable"]).resolve(),
        "dont_write_bytecode": sys.dont_write_bytecode,
    }
    out["environment"] = {k: os.environ.get(k) for k in ENV_KEYS}
    out["memory"] = {
        "total_bytes": vm.total,
        "available_bytes": vm.available,
        "rss_limit_at_launch_would_be": min(
            plan["rss_cap_bytes"], int(vm.available * plan["rss_available_fraction"])
        ),
    }
    out["disk_free_bytes"] = {"D:": shutil.disk_usage("D:/").free}
    mf = out["manifest"]
    out["checks"] = {
        "manifest_frozen": mf["is_frozen"],
        "entries_all_match": not mf["entries_mismatching"] and mf["entries"] == 65,
        "external_all_match": not mf["external_mismatching"] and mf["external"] == 5,
        "only_manifest_unlisted": mf["unlisted"] == ["manifest_ucov.json"],
        "policy_is_upp_v1": resolved["primary_policy"]["policy_id"] == "UPP-20261013-v1"
        and resolved["primary_policy"]["thresholds"]
        == {"max_rse": 0.5, "min_cosine": 0.65, "baseline_factor": 0.8},
        "plan_four_arms_in_order": [a["arm"] for a in plan["arms"]] == list(U.ARMS)
        and plan["max_training_starts"] == 4,
        "child_imports_approved": out["imports"]["child_all_approved"],
        "supervisor_imports_approved": out["imports"]["supervisor_all_approved"],
        "dependencies_equal": out["dependencies"]["all_equal"],
        "cuda_unavailable": out["dependencies"]["cuda_available"] is False,
        "interpreter_matches_plan": out["interpreter"]["matches_plan"],
        "pythonpath_unset": out["environment"]["PYTHONPATH"] is None,
        "ledger_empty": status["invocations"] == 0 and status["starts"] == 0 and not status["records"],
        "no_run_root_no_mirror": not status["run_root_exists"] and not any(status["roots_exist"].values()),
        "torch_free_package_checks": out["torch_imported_by_package_checks"] is False,
    }


def main() -> int:
    out_path = HERE / "prelaunch_verify.out.txt"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; never overwritten")
    out: dict = {"recorded_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "bytecode_before": bytecode()}
    try:
        verify(out)
    except Exception as exc:  # recorded, never hidden: any failure blocks the launch
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["bytecode_after"] = bytecode()
    checks = out.setdefault("checks", {})
    checks["no_bytecode"] = not out["bytecode_before"] and not out["bytecode_after"]
    out["all_ok"] = "error" not in out and len(checks) == 16 and all(checks.values())
    data = (json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with out_path.open("xb") as fh:
        fh.write(data)
    print(json.dumps({k: out.get(k) for k in ("all_ok", "checks", "error")}, indent=2))
    print(f"prelaunch_verify.out.txt {len(data)} bytes sha256={sha(data)}")
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
