"""Pre-launch verification of URP-20261013-v1 (read-only on the package, the run root and the ledger mirror).

``python -B prelaunch_verify.py`` from this directory, with PYTHONPATH unset. It writes only
``prelaunch_verify.out.txt`` in this directory (exclusive create, LF bytes) and refuses if that file exists.

1. Manifest: SHA-256 of the file bytes against the approved value; every entry (SHA-256 and size); every external
   bound file (SHA-256 and size); package files outside the manifest.
2. Package identity: ``urp_control.check_pinned`` and ``urp_control.verify_package`` (every entry, the pinned files,
   the U-FIT reference record, and a fresh resolution equal to ``resolved_config.json``), and the registered plan.
3. The child's scientific config branch (``urp_child.load_job_config``) with the job fields the supervisor writes.
4. Import resolution on the child's and the supervisor's sys.path, without importing: each runtime module must
   resolve to the manifest-listed file with the approved SHA-256. ``cg_fedllm`` is located only, never imported.
5. Dependency versions against the manifest; torch is imported only after the package checks, to read
   ``torch.__version__`` and ``torch.cuda.is_available()`` (both checked by the child after its start record).
6. Interpreter, environment, read-only ledger status, run root and mirror existence, memory and disk.
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

BASE_PATH = list(sys.path[1:])  # the interpreter's own path, without this script's directory
APPROVED = "1df7af08c8b5897ad23fac88ab94e4ad36212a47cd542bb584e85ca07df91bd1"
HERE = Path(__file__).resolve().parent
URP = HERE.parent / "urp_20261013_v1"
EXPECTED_UNLISTED = ["CONTINUATION_CHECKPOINT.md", "PROPOSED_LAUNCH.md", "manifest_urp.json"]
ENV_KEYS = ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "PYTHONDONTWRITEBYTECODE")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_file(path: Path, digest: str, size: int) -> dict:
    if not path.is_file():
        return {"path": str(path), "present": False, "match": False}
    data = path.read_bytes()
    return {"path": str(path), "present": True, "match": (sha(data), len(data)) == (digest, size)}


def resolve_on(path_list: list[str], names: list[str], expected: dict[str, str]) -> dict:
    """Where each top-level module name resolves on ``path_list`` (PathFinder only: nothing is imported)."""
    out = {}
    for name in names:
        spec = importlib.machinery.PathFinder.find_spec(name, path_list)
        origin = None if spec is None else spec.origin
        digest = sha(Path(origin).read_bytes()) if origin and Path(origin).is_file() else None
        out[name] = {"origin": origin, "sha256": digest, "approved": digest == expected.get(name)}
    return out


def bytecode_files() -> list[str]:
    """Any ``__pycache__`` or ``.pyc`` under the package and the closed directories it imports from."""
    roots = [URP, URP.parent / "uid_20261012_v2", URP.parent / "uid_20261012_v2_r1",
             URP.parent / "exploratory_progression_20261012"]
    return sorted(str(p) for r in roots for p in r.rglob("*") if p.name == "__pycache__" or p.suffix == ".pyc")


def main() -> int:
    out_path = HERE / "prelaunch_verify.out.txt"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; never overwritten")
    out: dict = {"recorded_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "bytecode_before": bytecode_files()}
    try:
        verify(out)
    except Exception as exc:  # recorded, never hidden: any failure blocks the launch
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["bytecode_after"] = bytecode_files()
    checks = out.setdefault("checks", {})
    checks["no_bytecode_written"] = not out["bytecode_before"] and not out["bytecode_after"]
    out["all_ok"] = "error" not in out and len(checks) == 15 and all(checks.values())
    data = (json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with out_path.open("xb") as fh:
        fh.write(data)
    print(data.decode("utf-8"), end="")
    print(f"prelaunch_verify.out.txt {len(data)} bytes sha256={sha(data)}")
    return 0 if out["all_ok"] else 1


def verify(out: dict) -> None:
    # 1. manifest bytes, entries, external bindings, unlisted files
    raw = (URP / "manifest_urp.json").read_bytes()
    m = json.loads(raw)
    rows = [check_file((URP / e["path"]).resolve(), e["sha256"], e["bytes"]) for e in m["files"]]
    ext = [check_file(Path(e["path"]), e["sha256"], e["bytes"]) for e in m["external_bound_files"]]
    listed = {(URP / e["path"]).resolve() for e in m["files"]}
    unlisted = sorted(
        p.relative_to(URP).as_posix() for p in URP.rglob("*") if p.is_file() and p.resolve() not in listed
    )
    out["manifest"] = {
        "path": str(URP / "manifest_urp.json"),
        "bytes": len(raw),
        "sha256": sha(raw),
        "is_approved": sha(raw) == APPROVED,
        "protocol_id": m.get("protocol_id"),
        "entries": len(rows),
        "entries_matching": sum(r["match"] for r in rows),
        "entries_mismatching": [r["path"] for r in rows if not r["match"]],
        "external_bound_files": len(ext),
        "external_matching": sum(r["match"] for r in ext),
        "external_mismatching": [r["path"] for r in ext if not r["match"]],
        "package_files_not_in_manifest": unlisted,
        "unlisted_as_expected": unlisted == EXPECTED_UNLISTED,
    }

    # 2. package identity and the registered plan (the supervisor's own functions; torch-free)
    sys.path.insert(0, str(URP))
    import urp_child as CH
    import urp_control as U

    K = U.K
    U.check_pinned()
    ident, resolved = U.verify_package(APPROVED)
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
            *U.MECHANICS_KEYS,
        )
    }
    soft, kill = K.arm_deadlines(0.0, 0.0, plan["arms"][0]["max_seconds"], plan["aggregate_seconds"],
                                 plan["end_margin_s"], plan["eval_reserve_s"])
    out["plan"]["deadlines_from_arm_start_s"] = {"soft_stop": soft, "kill": kill, "kill_wait": plan["kill_wait_s"]}
    out["resolved"] = {
        k: resolved[k] for k in ("arm", "variant", "model_init_seed", "optimizer", "max_updates", "max_seconds",
                                 "cpu", "criteria", "limits", "mechanics")
    }
    out["resolved"]["u_fit_model"] = resolved["u_fit"]["model"]
    out["resolved"]["input"] = resolved["input"]
    out["resolved"]["input_expected"] = resolved["input_expected"]
    out["resolved"]["reference_model_state_sha256_init"] = resolved["reference"]["model_state_sha256_init"]
    out["expected_module_sha256"] = modules

    # 3. the child's scientific config branch, with the fields the supervisor puts in job.json
    job = {
        "mode": "scientific",
        "config_path": str(U.URP_DIR / U.RESOLVED_NAME),
        "expected_config_sha256": ident["resolved_config_sha256"],
        "arm": U.ARM,
        "model_arm": U.ARM,
    }
    cfg, raw_sha, canon_sha = CH.load_job_config(job, K, U)
    out["child_scientific_config_branch"] = {
        "accepted": True,
        "raw_sha256": raw_sha,
        "canonical_sha256": canon_sha,
        "arm": cfg["arm"],
    }

    # 4. import resolution, nothing imported: child sys.path = job sys_path + script dir + interpreter path
    child_path = [str(URP), str(U.FROZEN_DIR), str(U.R1_DIR), str(URP), *BASE_PATH]
    supervisor_path = [str(URP), str(URP), *BASE_PATH, str(U.R1_DIR), str(U.FROZEN_DIR)]
    child = resolve_on(child_path, [*CH.MODULES, "urp_child"], modules)
    sup = resolve_on(supervisor_path, ["urp_control", "uid_r1_control"], modules)
    cg = importlib.machinery.PathFinder.find_spec("cg_fedllm", child_path)
    out["imports"] = {
        "child": child,
        "child_all_approved": all(v["approved"] for v in child.values()),
        "supervisor": sup,
        "supervisor_all_approved": all(v["approved"] for v in sup.values()),
        "child_forbidden_modules": list(CH.FORBIDDEN_MODULES),
        "cg_fedllm_would_resolve_to (forbidden in the child; not imported)": None if cg is None else cg.origin,
    }
    out["torch_imported_by_package_checks"] = "torch" in sys.modules

    # 5. dependencies (distribution metadata), then torch itself
    want = {k: m["dependencies"][k] for k in ("torch", "safetensors", "psutil", "numpy", "python")}
    have = {d: importlib.metadata.version(d) for d in ("torch", "safetensors", "psutil", "numpy")}
    have["python"] = sys.version.split()[0]
    import torch

    out["dependencies"] = {
        "manifest": want,
        "installed": have,
        "all_equal": have == want,
        "torch.__version__": torch.__version__,
        "torch_version_equals_plan": torch.__version__ == plan["torch_version"],
        "torch_origin": torch.__file__,
        "cuda_available_with_env": torch.cuda.is_available(),
    }

    # 6. interpreter, environment, ledger, memory, disk
    import psutil

    ex = resolved["execution"]
    status = K.ledger_status(Path(ex["run_root"]) / "ledger", ex["ledger_mirror"], [U.ARM])
    status["run_root_exists"] = Path(ex["run_root"]).exists()
    out["ledger_status"] = status
    vm = psutil.virtual_memory()
    out["interpreter"] = {
        "sys.executable": sys.executable,
        "matches_plan": Path(sys.executable).resolve() == Path(plan["python_executable"]).resolve(),
        "version": sys.version.split()[0],
        "dont_write_bytecode": sys.dont_write_bytecode,
    }
    out["environment"] = {k: os.environ.get(k) for k in ENV_KEYS}
    out["memory"] = {
        "total_bytes": vm.total,
        "available_bytes": vm.available,
        "rss_limit_at_launch_would_be": min(plan["rss_cap_bytes"], int(vm.available * plan["rss_available_fraction"])),
    }
    out["disk_free_bytes"] = {"D:": shutil.disk_usage("D:/").free}
    out["cpu_logical"] = psutil.cpu_count(logical=True)
    out["cpu_physical"] = psutil.cpu_count(logical=False)

    mf = out["manifest"]
    checks = {
        "manifest_approved": mf["is_approved"],
        "entries_all_match": not mf["entries_mismatching"] and mf["entries"] == 61,
        "external_all_match": not mf["external_mismatching"] and mf["external_bound_files"] == 4,
        "unlisted_as_expected": mf["unlisted_as_expected"],
        "child_imports_approved": out["imports"]["child_all_approved"],
        "supervisor_imports_approved": out["imports"]["supervisor_all_approved"],
        "dependencies_equal": out["dependencies"]["all_equal"],
        "torch_version_equals_plan": out["dependencies"]["torch_version_equals_plan"],
        "cuda_unavailable": out["dependencies"]["cuda_available_with_env"] is False,
        "interpreter_matches_plan": out["interpreter"]["matches_plan"],
        "pythonpath_unset": out["environment"]["PYTHONPATH"] is None,
        "ledger_empty": status["invocations"] == 0 and status["starts"] == 0 and not status["records"],
        "no_run_root_no_mirror": not status["run_root_exists"] and not any(status["roots_exist"].values()),
        "torch_free_package_checks": out["torch_imported_by_package_checks"] is False,
    }
    out["checks"] = checks


if __name__ == "__main__":
    sys.exit(main())
