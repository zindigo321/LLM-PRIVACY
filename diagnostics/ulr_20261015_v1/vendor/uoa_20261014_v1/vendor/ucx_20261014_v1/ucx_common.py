"""UCX-20261014-v1 shared helpers: hashing, write-once records, split-enforcing open guard, watchdog, child checks.

Standard library only at module level (psutil is imported lazily by the watchdog), so the supervisor, the analyzer and
every child can import it before torch or numpy. Records are written once (O_CREAT | O_EXCL) and fsynced.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

UCX_DIR = Path(__file__).resolve().parent
DIAG_ROOT = UCX_DIR.parent
REPO = DIAG_ROOT.parent
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONPATH",
    "OPENBLAS_NUM_THREADS",
)
CHILD_ENV = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
    "OPENBLAS_NUM_THREADS": "1",
}
EXIT = {"COMPLETED": 0, "RESOURCE_STOP": 3, "INCOMPLETE": 130, "EXECUTION_ERROR": 1}


class UCXError(RuntimeError):
    """A registered precondition does not hold; the step refuses (nothing scientific is substituted)."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(obj: Any) -> Any:
    """Non-finite floats become "nan" / "inf" / "-inf" so records stay strict JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return "nan" if obj != obj else ("inf" if obj > 0 else "-inf")
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def canonical_bytes(obj: Any) -> bytes:
    text = json.dumps(
        json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return (text + "\n").encode("utf-8")


def pretty_bytes(obj: Any) -> bytes:
    return (json.dumps(json_safe(obj), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_exclusive(path: str | Path, data: bytes) -> str:
    """Create ``path`` (never overwrite), write ``data``, fsync; returns the SHA-256 of the bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha256_bytes(data)


def write_json(path: str | Path, obj: Any) -> str:
    return write_exclusive(path, pretty_bytes(obj))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stats(values: list[float]) -> dict[str, Any]:
    s = sorted(values)
    n = len(s)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "min": s[0],
        "median": s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]),
        "p90_nearest_rank": s[max(0, math.ceil(0.9 * n) - 1)],
        "max": s[-1],
        "mean": math.fsum(s) / n,
        "sum": math.fsum(s),
    }


def install_open_guard(forbidden: list[str], attempts: list[str]) -> None:
    """An audit hook that refuses (PermissionError) any ``open`` of a forbidden path and records the attempt. Only a
    path whose base name is a forbidden base name is resolved, so ordinary imports stay cheap."""
    deny = {os.path.normcase(os.path.realpath(p)) for p in forbidden}
    names = {os.path.normcase(os.path.basename(p)) for p in deny}

    def hook(event: str, args: tuple) -> None:
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        try:
            raw = os.fsdecode(args[0])
            if os.path.normcase(os.path.basename(raw)) not in names:
                return
            p = os.path.normcase(os.path.realpath(raw))
        except (TypeError, ValueError, OSError):
            return
        if p in deny:
            attempts.append(p)
            raise PermissionError(f"UCX split enforcement: this process may not open {p}")

    sys.addaudithook(hook)


LEDGER_NAMES = ("invocation_001.json", "invocation_001_end.json") + tuple(
    f"{kind}_{step}.json"
    for step in ("BUILD-TRAIN", "PCA-FIT", "U-QUARTER", "BUILD-HELDOUT", "EVAL")
    for kind in ("start", "end")
)


class Ledger:
    """Two roots holding byte-identical, write-once records (a primary under the run root, a mirror outside it). Any
    disagreement, unknown entry or unreadable record refuses; deleting the run root does not reset the mirror."""

    def __init__(self, primary: str | Path, mirror: str | Path) -> None:
        self.roots = (Path(primary), Path(mirror))
        if self.roots[0].resolve() == self.roots[1].resolve():
            raise UCXError("ledger primary and mirror must be different directories")

    def read(self) -> dict[str, Any]:
        listings = []
        for root in self.roots:
            files: dict[str, bytes] = {}
            if root.exists():
                for p in root.iterdir():
                    if not p.is_file() or p.name not in LEDGER_NAMES:
                        raise UCXError(f"unexpected entry in ledger root {root}: {p.name}")
                    files[p.name] = p.read_bytes()
            listings.append(files)
        if listings[0] != listings[1]:
            raise UCXError(f"ledger roots disagree: {sorted(listings[0])} vs {sorted(listings[1])}")
        try:
            return {n: json.loads(d.decode("utf-8")) for n, d in listings[0].items()}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UCXError(f"unreadable ledger record: {exc}") from exc

    def write(self, name: str, obj: Any) -> str:
        """Primary first, then mirror. A failure between the two leaves the roots disagreeing, which blocks."""
        if name not in LEDGER_NAMES:
            raise UCXError(f"invalid ledger record name {name!r}")
        data = canonical_bytes(obj)
        for root in self.roots:
            write_exclusive(root / name, data)
        return sha256_bytes(data)


def ledger_status(records: dict[str, Any]) -> dict[str, Any]:
    steps = ("BUILD-TRAIN", "PCA-FIT", "U-QUARTER", "BUILD-HELDOUT", "EVAL")
    state = {}
    for s in steps:
        start, end = records.get(f"start_{s}.json"), records.get(f"end_{s}.json")
        if end is not None and start is None:
            raise UCXError(f"end record without start record for {s}")
        state[s] = (
            "NOT_STARTED" if start is None else ("STARTED_UNFINISHED" if end is None else end["status"])
        )
    return {
        "invocations": int("invocation_001.json" in records),
        "invocation_closed": "invocation_001_end.json" in records,
        "steps": state,
        "neural_starts": int("start_U-QUARTER.json" in records),
        "pca_fits": int("start_PCA-FIT.json" in records),
        "records": sorted(records),
    }


def start_watchdog(job: dict[str, Any], who: str) -> None:
    """Exit (no record) if the supervisor is gone or the hard deadline passes; turn a stop request into
    KeyboardInterrupt. The supervisor is identified by PID and creation time, so a reused PID is not mistaken for it."""
    import _thread
    import threading
    import time

    import psutil

    def gone(why: str) -> None:
        sys.stderr.write(f"{who}: {why}; exiting without a record\n")
        sys.stderr.flush()
        os._exit(137)

    try:
        parent = psutil.Process(int(job["supervisor_pid"]))
        same = abs(parent.create_time() - float(job["supervisor_create_time"])) < 0.01
    except psutil.Error:
        same = False
    if not same:
        gone("supervisor not running at child start")
    hard_at, poll, stop_file = (
        float(job["hard_stop_monotonic"]),
        float(job["poll_interval_s"]),
        Path(job["stop_file"]),
    )

    def run() -> None:
        interrupted = False
        while True:
            time.sleep(poll)
            if time.monotonic() >= hard_at:
                gone("hard deadline passed")
            if not parent.is_running():
                gone("supervisor ended")
            if not interrupted and stop_file.exists():
                interrupted = True
                _thread.interrupt_main()

    threading.Thread(target=run, name="watchdog", daemon=True).start()


def peak_memory() -> dict[str, Any]:
    import psutil

    mi = psutil.Process(os.getpid()).memory_info()
    return {
        "os_peak_wset_bytes": getattr(mi, "peak_wset", None),
        "rss_end_bytes": mi.rss,
        "private_end_bytes": getattr(mi, "private", None),
    }


def check_env(rec: dict[str, Any]) -> None:
    rec["env"] = {k: os.environ.get(k) for k in ENV_KEYS}
    expected = {k: v for k, v in CHILD_ENV.items() if k in ENV_KEYS}
    if rec["env"]["PYTHONPATH"] is not None or any(rec["env"][k] != v for k, v in expected.items()):
        raise UCXError(f"child environment not as set by the supervisor: {rec['env']}")
