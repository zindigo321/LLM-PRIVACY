"""ULR-20261015-v1: content fingerprint of the protected work that this study must not change (read-only).

``python -B fingerprint_protected.py <out.json>`` writes, for every protected git checkout, its HEAD, branch and
``git status`` entries with the SHA-256 and size of every dirty or untracked file (a nested worktree listed as an
untracked directory is fingerprinted through its own ``git status``), and for every protected run root the SHA-256 and
size of every file. ``python -B fingerprint_protected.py --compare <before.json> <after.json>`` compares two records.
Standard library only; nothing is written except the output file (created exclusively).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

CHECKOUTS = {
    "original_repository": "D:/projects-cd/cg-fedllm-reproduction",
    "ucx_worktree": "D:/projects-cd/cg-fedllm-ucx",
    "uho_worktree": "D:/projects-cd/cg-fedllm-uho",
    "publish_ucov_worktree": "D:/projects-cd/cg-fedllm-publish-ucov",
    "publish_uformer_worktree": "D:/projects-cd/cg-fedllm-publish-uformer",
    "uoa_worktree": "D:/projects-cd/cg-fedllm-uoa",
}
# every file under these trees, including files ignored by .gitignore (for example manifest-bound *.log evidence)
TREES = (
    "D:/projects-cd/cg-fedllm-reproduction/diagnostics",
    "D:/projects-cd/cg-fedllm-reproduction/.claude",
    "D:/projects-cd/cg-fedllm-ucx/diagnostics",
    "D:/projects-cd/cg-fedllm-uho/diagnostics",
    "D:/projects-cd/cg-fedllm-uoa/diagnostics",
)
RUN_ROOTS = (
    "D:/cgfed-runs/uoa_20261014_v1",
    "D:/cgfed-runs/ucx_20261014_v1",
    "D:/cgfed-runs/uho_20261014_v1",
    "D:/cgfed-runs/ucov_20261013_v1",
    "D:/cgfed-runs/urp_20261013_v1",
    "D:/cgfed-runs/urc_20261013_v1",
    "D:/cgfed-runs/uid_20261012_v2_r1",
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo: str, *args: str) -> str:
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True)
    return r.stdout.decode("utf-8", "surrogateescape")


def files_under(root: Path) -> dict[str, dict]:
    return {
        p.relative_to(root).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def checkout(repo: str) -> dict:
    out = {
        "head": git(repo, "rev-parse", "HEAD").strip(),
        "branch": git(repo, "branch", "--show-current").strip(),
    }
    raw = git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    entries = [e for e in raw.split("\0") if e]
    out["status_entries"] = entries
    files, nested = {}, {}
    for e in entries:
        code, path = e[:2], e[3:]
        p = Path(repo) / path
        if p.is_file():
            files[path] = {"status": code, "sha256": sha256_file(p), "bytes": p.stat().st_size}
        elif p.is_dir() and (p / ".git").exists():
            nested[path] = checkout(str(p))
        elif p.is_dir():
            files.update({f"{path}{k}": {"status": code, **v} for k, v in files_under(p).items()})
        else:
            files[path] = {"status": code, "sha256": None, "bytes": None}
    out["dirty_files"] = files
    out["nested_worktrees"] = nested
    return out


def main(out_path: str) -> int:
    rec = {
        "schema": "ulr-20261015-v1.protected-fingerprint/v1",
        "checkouts": {k: checkout(v) for k, v in CHECKOUTS.items() if Path(v).is_dir()},
        "trees": {t: files_under(Path(t)) for t in TREES if Path(t).is_dir()},
        "run_roots": {r: files_under(Path(r)) for r in RUN_ROOTS if Path(r).is_dir()},
    }
    data = (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode("utf-8")
    with open(out_path, "xb") as fh:
        fh.write(data)
    summary = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "checkouts": {
            k: (v["head"], len(v["status_entries"]), len(v["dirty_files"]))
            for k, v in rec["checkouts"].items()
        },
        "trees": {k: len(v) for k, v in rec["trees"].items()},
        "run_roots": {k: len(v) for k, v in rec["run_roots"].items()},
    }
    print(json.dumps(summary, indent=1))
    return 0


def compare(a_path: str, b_path: str) -> int:
    a = json.loads(Path(a_path).read_text("utf-8"))
    b = json.loads(Path(b_path).read_text("utf-8"))
    diffs = []
    for k in sorted(set(a["checkouts"]) | set(b["checkouts"])):
        if a["checkouts"].get(k) != b["checkouts"].get(k):
            diffs.append(f"checkout {k} differs")
    for group in ("trees", "run_roots"):
        for k in sorted(set(a[group]) | set(b[group])):
            fa, fb = a[group].get(k, {}), b[group].get(k, {})
            for f in sorted(set(fa) | set(fb)):
                if fa.get(f) != fb.get(f):
                    diffs.append(
                        f"{group} {k}: {f} {'changed' if f in fa and f in fb else 'added or removed'}"
                    )
    print(json.dumps({"identical": not diffs, "differences": diffs}, indent=1))
    return 0 if not diffs else 1


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    sys.exit(main(sys.argv[1]))
