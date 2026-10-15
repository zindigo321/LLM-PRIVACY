"""ULR-20261015-v1 shared helpers: import paths, the write-once two-root ledger, the import check, and re-exports of
the audited UOA / UCX helpers (hashing, exclusive writes, open guard, watchdog, memory and environment checks).

Standard library only at module level. ``setup_paths`` puts this package, then the byte-identical vendored UOA
package (``vendor/uoa_20261014_v1``) and its own vendored dependencies first on ``sys.path``. Nothing is imported
from another worktree, and ``cg_fedllm`` is never imported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PKG = Path(__file__).resolve().parent
UOA = PKG / "vendor" / "uoa_20261014_v1"
UOA_VENDOR = ("ucx_20261014_v1", "uid_20261012_v2", "uid_20261012_v2_r1", "upp_20261014_v1")


def setup_paths() -> None:
    for p in reversed([PKG, UOA, *(UOA / "vendor" / d for d in UOA_VENDOR)]):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


setup_paths()

import uoa_common as _U  # noqa: E402  (vendored UOA helper, byte-identical; it adds its own vendor paths)

setup_paths()  # keep this package first after the vendored module adjusted the path

sha256_bytes = _U.sha256_bytes
sha256_file = _U.sha256_file
json_safe = _U.json_safe
canonical_bytes = _U.canonical_bytes
pretty_bytes = _U.pretty_bytes
write_exclusive = _U.write_exclusive
write_json = _U.write_json
load_json = _U.load_json
stats = _U.stats
install_open_guard = _U.install_open_guard
start_watchdog = _U.start_watchdog
peak_memory = _U.peak_memory
load_checked = _U.load_checked
save_tensors = _U.save_tensors
CHILD_ENV = _U.CHILD_ENV
EXIT = _U.EXIT


class ULRError(RuntimeError):
    """A registered precondition does not hold; the step refuses (nothing scientific is substituted)."""


def check_env(rec: dict[str, Any]) -> None:
    try:
        _U.check_env(rec)
    except _U.UOAError as exc:
        raise ULRError(str(exc)) from exc


def check_imports(rec: dict[str, Any]) -> None:
    """Every imported module that lives under a repository worktree must come from this package (its vendor tree
    included); ``cg_fedllm`` must not be imported at all."""
    if "cg_fedllm" in sys.modules:
        raise ULRError("cg_fedllm must not be imported by a ULR process")
    bad, rows = [], {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not isinstance(f, str) or not Path(f).is_absolute() or name in ("__main__", "__mp_main__"):
            continue
        p = Path(f).resolve()
        if "projects-cd" in p.as_posix():
            if not p.is_relative_to(PKG):
                bad.append(f"{name}: {p}")
            else:
                rows[name] = {"path": p.relative_to(PKG).as_posix(), "sha256": sha256_file(p)}
    rec["modules"] = dict(sorted(rows.items()))
    if bad:
        raise ULRError(f"modules imported from outside this package: {bad}")


STEP_NAMES = ("INIT", "C16-CONTROL", "C4-REDUCED", "C1-REDUCED", "EVAL")
LEDGER_NAMES = ("invocation_001.json", "invocation_001_end.json") + tuple(
    f"{kind}_{step}.json" for step in STEP_NAMES for kind in ("start", "end")
)


class Ledger:
    """Two roots holding byte-identical, write-once records (a primary under the run root, a mirror in the package).
    Any disagreement, unknown entry or unreadable record refuses; deleting the run root does not reset the mirror."""

    def __init__(self, primary: str | Path, mirror: str | Path) -> None:
        self.roots = (Path(primary), Path(mirror))
        if self.roots[0].resolve() == self.roots[1].resolve():
            raise ULRError("ledger primary and mirror must be different directories")

    def read(self) -> dict[str, Any]:
        listings = []
        for root in self.roots:
            files: dict[str, bytes] = {}
            if root.exists():
                for p in root.iterdir():
                    if not p.is_file() or p.name not in LEDGER_NAMES:
                        raise ULRError(f"unexpected entry in ledger root {root}: {p.name}")
                    files[p.name] = p.read_bytes()
            listings.append(files)
        if listings[0] != listings[1]:
            raise ULRError(f"ledger roots disagree: {sorted(listings[0])} vs {sorted(listings[1])}")
        try:
            return {n: json.loads(d.decode("utf-8")) for n, d in listings[0].items()}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ULRError(f"unreadable ledger record: {exc}") from exc

    def write(self, name: str, obj: Any) -> str:
        if name not in LEDGER_NAMES:
            raise ULRError(f"invalid ledger record name {name!r}")
        data = canonical_bytes(obj)
        for root in self.roots:
            write_exclusive(root / name, data)
        return sha256_bytes(data)


def ledger_status(records: dict[str, Any]) -> dict[str, Any]:
    state = {}
    for s in STEP_NAMES:
        start, end = records.get(f"start_{s}.json"), records.get(f"end_{s}.json")
        if end is not None and start is None:
            raise ULRError(f"end record without start record for {s}")
        state[s] = (
            "NOT_STARTED" if start is None else ("STARTED_UNFINISHED" if end is None else end["status"])
        )
    return {
        "invocations": int("invocation_001.json" in records),
        "invocation_closed": "invocation_001_end.json" in records,
        "steps": state,
        "neural_starts": sum(int(f"start_{s}.json" in records) for s in STEP_NAMES[1:4]),
        "channel_pca_fits": int("start_INIT.json" in records),
        "records": sorted(records),
    }
