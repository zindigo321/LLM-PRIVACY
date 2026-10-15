"""UOA-20261014-v1 shared helpers: import paths, the write-once two-root ledger, and re-exports of the audited UCX
helpers (hashing, exclusive writes, the split-enforcing open guard, the watchdog, memory and environment checks).

Standard library only at module level. ``setup_paths`` puts the package and its vendored, byte-identical dependencies
first on ``sys.path``; nothing is imported from another worktree, and ``cg_fedllm`` is never imported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PKG = Path(__file__).resolve().parent
VENDOR = PKG / "vendor"
VENDOR_DIRS = ("ucx_20261014_v1", "uid_20261012_v2", "uid_20261012_v2_r1", "upp_20261014_v1")


def setup_paths() -> None:
    for p in reversed([PKG, *(VENDOR / d for d in VENDOR_DIRS)]):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


setup_paths()

import ucx_common as _K  # noqa: E402  (vendored, byte-identical; see SOURCE_INVENTORY.md)

sha256_bytes = _K.sha256_bytes
sha256_file = _K.sha256_file
json_safe = _K.json_safe
canonical_bytes = _K.canonical_bytes
pretty_bytes = _K.pretty_bytes
write_exclusive = _K.write_exclusive
write_json = _K.write_json
load_json = _K.load_json
stats = _K.stats
install_open_guard = _K.install_open_guard
start_watchdog = _K.start_watchdog
peak_memory = _K.peak_memory
CHILD_ENV = _K.CHILD_ENV
EXIT = _K.EXIT


class UOAError(RuntimeError):
    """A registered precondition does not hold; the step refuses (nothing scientific is substituted)."""


def check_env(rec: dict[str, Any]) -> None:
    try:
        _K.check_env(rec)
    except _K.UCXError as exc:
        raise UOAError(str(exc)) from exc


def check_imports(rec: dict[str, Any]) -> None:
    """Every imported module that lives in a repository worktree must come from this package (or its vendor
    directory); ``cg_fedllm`` must not be imported at all."""
    if "cg_fedllm" in sys.modules:
        raise UOAError("cg_fedllm must not be imported by a UOA process")
    bad, rows = [], {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not isinstance(f, str) or not Path(f).is_absolute():
            continue
        p = Path(f).resolve()
        if "projects-cd" in p.as_posix():
            if not p.is_relative_to(PKG):
                bad.append(f"{name}: {p}")
            elif name not in ("__main__", "__mp_main__"):
                rows[name] = {"path": p.relative_to(PKG).as_posix(), "sha256": sha256_file(p)}
    rec["modules"] = dict(sorted(rows.items()))
    if bad:
        raise UOAError(f"modules imported from outside this package: {bad}")


LEDGER_NAMES = ("invocation_001.json", "invocation_001_end.json") + tuple(
    f"{kind}_{step}.json"
    for step in ("DIAG", "STAGE-A", "U-RELU-LONG", "U-LEAKY-LONG", "CALIB", "EVAL")
    for kind in ("start", "end")
)


class Ledger:
    """Two roots holding byte-identical, write-once records (a primary under the run root, a mirror in the package).
    Any disagreement, unknown entry or unreadable record refuses; deleting the run root does not reset the mirror."""

    def __init__(self, primary: str | Path, mirror: str | Path) -> None:
        self.roots = (Path(primary), Path(mirror))
        if self.roots[0].resolve() == self.roots[1].resolve():
            raise UOAError("ledger primary and mirror must be different directories")

    def read(self) -> dict[str, Any]:
        listings = []
        for root in self.roots:
            files: dict[str, bytes] = {}
            if root.exists():
                for p in root.iterdir():
                    if not p.is_file() or p.name not in LEDGER_NAMES:
                        raise UOAError(f"unexpected entry in ledger root {root}: {p.name}")
                    files[p.name] = p.read_bytes()
            listings.append(files)
        if listings[0] != listings[1]:
            raise UOAError(f"ledger roots disagree: {sorted(listings[0])} vs {sorted(listings[1])}")
        try:
            return {n: json.loads(d.decode("utf-8")) for n, d in listings[0].items()}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UOAError(f"unreadable ledger record: {exc}") from exc

    def write(self, name: str, obj: Any) -> str:
        """Primary first, then mirror. A failure between the two leaves the roots disagreeing, which blocks."""
        if name not in LEDGER_NAMES:
            raise UOAError(f"invalid ledger record name {name!r}")
        data = canonical_bytes(obj)
        for root in self.roots:
            write_exclusive(root / name, data)
        return sha256_bytes(data)


def ledger_status(records: dict[str, Any]) -> dict[str, Any]:
    steps = ("DIAG", "STAGE-A", "U-RELU-LONG", "U-LEAKY-LONG", "CALIB", "EVAL")
    state = {}
    for s in steps:
        start, end = records.get(f"start_{s}.json"), records.get(f"end_{s}.json")
        if end is not None and start is None:
            raise UOAError(f"end record without start record for {s}")
        state[s] = (
            "NOT_STARTED" if start is None else ("STARTED_UNFINISHED" if end is None else end["status"])
        )
    return {
        "invocations": int("invocation_001.json" in records),
        "invocation_closed": "invocation_001_end.json" in records,
        "steps": state,
        "neural_starts": sum(int(f"start_{s}.json" in records) for s in ("U-RELU-LONG", "U-LEAKY-LONG")),
        "pca_fits": int("start_STAGE-A.json" in records),
        "records": sorted(records),
    }


def load_checked(path: str | Path, sha: str, kind: str = "numpy") -> dict[str, Any]:
    """A safetensors file, after its bytes match the recorded SHA-256."""
    data = Path(path).read_bytes()
    if sha256_bytes(data) != sha:
        raise UOAError(f"{Path(path).name} differs from its recorded SHA-256")
    if kind == "torch":
        from safetensors.torch import load
    else:
        from safetensors.numpy import load
    return load(data)


def save_tensors(
    path: str | Path, tensors: dict[str, Any], meta: dict[str, str], kind: str = "numpy"
) -> dict:
    """Write a safetensors file exclusively and read it back bit for bit."""
    if kind == "torch":
        import torch
        from safetensors.torch import load, save

        data = save(tensors, metadata=meta)
        back = load(data)
        same = set(back) == set(tensors) and all(
            back[k].dtype == tensors[k].dtype and torch.equal(back[k], tensors[k]) for k in tensors
        )
    else:
        import numpy as np
        from safetensors.numpy import load, save

        data = save(tensors, metadata=meta)
        back = load(data)
        same = set(back) == set(tensors) and all(
            back[k].dtype == tensors[k].dtype and np.array_equal(back[k], tensors[k]) for k in tensors
        )
    if not same:
        raise UOAError(f"{Path(path).name} does not round-trip bit for bit")
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": write_exclusive(path, data),
        "tensors": len(tensors),
    }
