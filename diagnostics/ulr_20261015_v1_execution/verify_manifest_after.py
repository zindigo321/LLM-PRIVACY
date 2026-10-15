"""ULR-20261015-v1: re-hash every file listed in the frozen manifest after the run (read-only).

``python -B verify_manifest_after.py``; prints one JSON line (also kept as ``verify_manifest_after.out.txt``).
"""

import hashlib
import json
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "ulr_20261015_v1"
m_bytes = (PKG / "manifest_ulr.json").read_bytes()
m = json.loads(m_bytes)
bad = [e["path"] for e in m["files"] if hashlib.sha256((PKG / e["path"]).read_bytes()).hexdigest() != e["sha256"]]
ext = [e["path"] for e in m["external_bound_files"] if hashlib.sha256(Path(e["path"]).read_bytes()).hexdigest() != e["sha256"]]
listed = {e["path"] for e in m["files"]} | {"manifest_ulr.json"}
new = sorted(
    p.relative_to(PKG).as_posix()
    for p in PKG.rglob("*")
    if p.is_file() and p.relative_to(PKG).as_posix() not in listed
)
print(json.dumps({
    "manifest_sha256": hashlib.sha256(m_bytes).hexdigest(),
    "files_unchanged": f"{len(m['files']) - len(bad)}/{len(m['files'])}",
    "external_unchanged": f"{len(m['external_bound_files']) - len(ext)}/{len(m['external_bound_files'])}",
    "changed": bad + ext,
    "files_not_in_manifest": new,
}, sort_keys=True))
