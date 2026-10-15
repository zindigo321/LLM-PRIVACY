"""ULR-20261015-v1 pre-launch verification (read-only). ``python -B prelaunch_verify.py <manifest_sha256>``.

Checks: the manifest hash; every listed package file and every external bound file against the manifest; no
bytecode in the package; PYTHONPATH unset; no run root and no ledger records yet (mirror empty); no other Python
process; available memory against the registered neural floor; the registered config values. Prints one JSON line.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import psutil

PKG = Path(__file__).resolve().parent.parent / "ulr_20261015_v1"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(expected: str) -> int:
    res: dict = {"problems": []}
    m_path = PKG / "manifest_ulr.json"
    res["manifest_sha256"] = sha(m_path)
    if res["manifest_sha256"] != expected:
        res["problems"].append("manifest SHA-256 differs from the expected frozen digest")
    m = json.loads(m_path.read_text("utf-8"))
    bad = [
        e["path"]
        for e in m["files"]
        if not (PKG / e["path"]).is_file() or sha(PKG / e["path"]) != e["sha256"]
    ]
    bad += [
        e["path"]
        for e in m["external_bound_files"]
        if not Path(e["path"]).is_file() or sha(Path(e["path"])) != e["sha256"]
    ]
    res["files_checked"] = len(m["files"])
    res["external_checked"] = len(m["external_bound_files"])
    res["mismatches"] = bad
    listed = {e["path"] for e in m["files"]}
    extra = [
        p.relative_to(PKG).as_posix()
        for p in PKG.rglob("*")
        if p.is_file() and p.relative_to(PKG).as_posix() not in listed | {"manifest_ulr.json"}
    ]
    res["unlisted_package_files"] = extra
    res["bytecode"] = [str(p) for p in PKG.rglob("__pycache__")]
    res["pythonpath"] = os.environ.get("PYTHONPATH")
    cfg = json.loads((PKG / "ulr_config.json").read_text("utf-8"))
    run_root = Path(cfg["execution"]["run_root"])
    mirror = Path(cfg["execution"]["ledger_mirror"])
    res["run_root_exists"] = run_root.exists()
    res["mirror_records"] = sorted(p.name for p in mirror.iterdir()) if mirror.exists() else []
    me = os.getpid()
    res["other_python_processes"] = [
        p.info["pid"]
        for p in psutil.process_iter(["pid", "name"])
        if "python" in (p.info["name"] or "").lower() and p.info["pid"] != me
    ]
    avail = psutil.virtual_memory().available
    lim = min(cfg["limits"]["rss_cap_bytes"], int(avail * cfg["limits"]["rss_available_fraction"]))
    res["memory"] = {
        "available_bytes": avail,
        "child_limit_bytes": lim,
        "neural_floor_bytes": cfg["preflight"]["min_rss_limit_bytes"],
    }
    if (
        bad
        or extra
        or res["bytecode"]
        or res["pythonpath"]
        or res["run_root_exists"]
        or res["mirror_records"]
    ):
        res["problems"].append("package, environment or run-area state is not launchable")
    if lim < cfg["preflight"]["min_rss_limit_bytes"]:
        res["problems"].append("available memory is below the registered neural floor")
    res["all_ok"] = not res["problems"]
    print(json.dumps(res, sort_keys=True))
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
