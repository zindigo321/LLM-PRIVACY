"""UID-20261012-v2 r1: start accounting, deadlines, plans and package checks (imports no torch).

Engineering revision r1 of the frozen UID-20261012-v2 package (execution machinery only). Every start is a record
created with O_CREAT | O_EXCL and fsync in two ledger roots, a primary under the run root and a mirror outside it, which
must stay byte-identical: a record is written once, a crash mid-write leaves a record that blocks re-entry, and changing
or deleting the run root does not reset the count. An arm starts only after every earlier arm has an end record and its
prerequisite ended COMPLETED_PASS; a start without an end record is INCOMPLETE and is never re-entered automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

R1_DIR = Path(__file__).resolve().parent
DIAG_ROOT = R1_DIR.parent
FROZEN_DIR = DIAG_ROOT / "uid_20261012_v2"
REGISTERED_PROTOCOL_ID = "UID-20261012-v2"
REVISION_ID = "UID-20261012-v2-r1"
TERMINAL = ("COMPLETED_PASS", "COMPLETED_FIT_FAIL", "RESOURCE_STOP", "EXECUTION_ERROR", "INCOMPLETE")
INVOCATION_RE = re.compile(r"^invocation_(\d{1,4})\.json$")


class ControlError(RuntimeError):
    """A ledger, plan or package precondition does not hold; no start may be made."""


def canonical_bytes(obj: Any) -> bytes:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def sha256_canonical(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def create_exclusive(path: Path, obj: Any) -> str:
    """Create ``path`` (never overwrite), write canonical JSON, fsync; returns the SHA-256 of the written bytes."""
    data = canonical_bytes(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(data).hexdigest()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_safe(obj: Any) -> Any:
    """Non-finite floats become "nan" / "inf" / "-inf" (the frozen ``uid_common.json_safe`` rule)."""
    if isinstance(obj, float) and obj != obj:
        return "nan"
    if isinstance(obj, float) and obj in (float("inf"), float("-inf")):
        return "inf" if obj > 0 else "-inf"
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


# child exit codes; the supervisor accepts a child record only when its status matches the exit code
CHILD_EXIT = {
    "COMPLETED_PASS": 0,
    "COMPLETED_FIT_FAIL": 0,
    "RESOURCE_STOP": 3,
    "INCOMPLETE": 130,
    "EXECUTION_ERROR": 1,
}


# ---- ledger ---------------------------------------------------------------------------------------------
LEDGER_NAME_RE = re.compile(r"^(invocation_\d{3}(_end)?|start_[A-Za-z0-9-]+|end_[A-Za-z0-9-]+)\.json$")


class Ledger:
    """Two roots holding byte-identical, write-once records; any disagreement or unreadable record refuses."""

    def __init__(self, primary: str | Path, mirror: str | Path) -> None:
        self.roots = (Path(primary), Path(mirror))
        if self.roots[0].resolve() == self.roots[1].resolve():
            raise ControlError("ledger primary and mirror must be different directories")

    def read(self) -> dict[str, Any]:
        listings = []
        for root in self.roots:
            files: dict[str, bytes] = {}
            if root.exists():
                for p in root.iterdir():
                    if not p.is_file() or not LEDGER_NAME_RE.match(p.name):
                        raise ControlError(f"unexpected entry in ledger root {root}: {p.name}")
                    files[p.name] = p.read_bytes()
            listings.append(files)
        if listings[0] != listings[1]:
            raise ControlError(f"ledger roots disagree: {sorted(listings[0])} vs {sorted(listings[1])}")
        records = {}
        for name, data in listings[0].items():
            try:
                records[name] = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ControlError(f"unreadable ledger record {name}: {exc}") from exc
        return records

    def write(self, name: str, obj: Any) -> str:
        """Primary first, then mirror. A failure between the two leaves the roots disagreeing, which blocks."""
        if not LEDGER_NAME_RE.match(name):
            raise ControlError(f"invalid ledger record name {name!r}")
        digests = [create_exclusive(root / name, obj) for root in self.roots]
        return digests[0]


def summarize(records: dict[str, Any], arms: list[str]) -> dict[str, Any]:
    starts = sorted(n for n in records if n.startswith("start_"))
    opened = sorted(n for n in records if INVOCATION_RE.match(n))
    state = {}
    for arm in arms:
        start, end = records.get(f"start_{arm}.json"), records.get(f"end_{arm}.json")
        if end is not None and start is None:
            raise ControlError(f"end record without start record for {arm}")
        state[arm] = (
            "NOT_STARTED" if start is None else ("STARTED_UNFINISHED" if end is None else end["status"])
        )
    for name in starts:
        arm = name[len("start_") : -len(".json")]
        if arm not in state:
            state[arm] = (
                "STARTED_UNFINISHED"
                if f"end_{arm}.json" not in records
                else records[f"end_{arm}.json"]["status"]
            )
    return {"invocations": len(opened), "starts": len(starts), "arms": state}


def check_can_open(records: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    s = summarize(records, [a["arm"] for a in plan["arms"]])
    unfinished = [a for a, v in s["arms"].items() if v == "STARTED_UNFINISHED"]
    if unfinished:
        raise ControlError(f"INCOMPLETE prior start(s) {unfinished}: never re-entered automatically")
    if s["invocations"] >= plan["max_invocations"]:
        raise ControlError(
            f"{s['invocations']} invocation(s) already recorded; max {plan['max_invocations']}"
        )
    return s


def check_can_start(records: dict[str, Any], plan: dict[str, Any], arm: str) -> None:
    s = summarize(records, [a["arm"] for a in plan["arms"]])
    if f"start_{arm}.json" in records:
        raise ControlError(f"{arm} already has a start record; a second start is refused")
    if [a for a, v in s["arms"].items() if v == "STARTED_UNFINISHED"]:
        raise ControlError("an unfinished start exists; refusing a new start")
    if s["starts"] >= plan["max_training_starts"]:
        raise ControlError(f"{s['starts']} start(s) recorded; max {plan['max_training_starts']}")
    pre = plan["prerequisite"].get(arm)
    if pre is not None and s["arms"].get(pre) != "COMPLETED_PASS":
        raise ControlError(f"{arm} requires {pre} COMPLETED_PASS, found {s['arms'].get(pre)}")


def validate_plan(plan: dict[str, Any]) -> None:
    """Timing invariants. The kill is issued ``end_margin_s`` before a limit and may wait ``kill_wait_s`` for the
    process to exit, so the end record fits inside the limit only if the margin exceeds that wait."""
    if not plan["end_margin_s"] > plan["kill_wait_s"] > 0:
        raise ControlError("end_margin_s must exceed kill_wait_s (> 0)")
    if not (plan["eval_reserve_s"] > 0 and plan["poll_interval_s"] > 0 and plan["interrupt_grace_s"] > 0):
        raise ControlError("eval_reserve_s, poll_interval_s and interrupt_grace_s must be positive")
    for a in plan["arms"]:
        if a["max_seconds"] <= plan["end_margin_s"] + plan["eval_reserve_s"]:
            raise ControlError(f"{a['arm']}: max_seconds leaves no time to train")


def arm_deadlines(
    t0: float, t_start: float, arm_limit: float, aggregate_limit: float, margin: float, reserve: float
) -> tuple[float, float]:
    """(soft stop, hard kill) on one clock. The kill leaves ``margin`` for the end record inside both the arm limit
    (from the start record) and the aggregate limit (from the invocation record); training stops ``reserve`` earlier
    so that the final evaluation and the arm's records fit before the kill."""
    kill_at = min(t_start + arm_limit, t0 + aggregate_limit) - margin
    return kill_at - reserve, kill_at


# ---- package identity and the registered plan ---------------------------------------------------------
FROZEN_MANIFEST_SHA256 = "a80bebd414cad4aaf55c48cc5ca665ff658cf1388d687056db0c2ccc3cf286ca"
FROZEN_CONFIG_RAW_SHA256 = "d7ae76052d50776705f386655b0b501d403104d7d4ca118865a2b6ff3b71b0bd"
FROZEN_CONFIG_CANONICAL_SHA256 = "bb0111fab6310ab5430d931b781fc3c97cdb928da773d0cf8e71ea532fb4e5b3"
REGISTERED_ARMS = ("U-CHECK", "U-FIT")
REGISTERED_PREREQUISITE = {"U-FIT": "U-CHECK"}  # protocol.md: U-FIT only after U-CHECK passes
REGISTERED_CURVE_EVERY = 100  # protocol.md: step 0, step 1, every 100 updates and the final state
REGISTERED_EVIDENCE_CAP_BYTES = 512 * 2**20  # protocol.md: new retained evidence at most 512 MiB
REGISTERED_RSS_AVAILABLE_FRACTION = 0.5  # protocol.md: RSS limit min(4 GiB, 50% of available RAM at launch)
EXECUTION_KEYS = (
    "run_root",
    "ledger_mirror",
    "python_executable",
    "torch_version",
    "end_margin_s",
    "eval_reserve_s",
    "poll_interval_s",
    "interrupt_grace_s",
    "kill_wait_s",
)


# modules the scientific child may import, by import name -> path relative to R1_DIR (as listed in manifest_r1.json)
MODULE_PATHS = {
    "uid_common": "../uid_20261012_v2/uid_common.py",
    "uformer_ae": "../uid_20261012_v2/uformer_ae.py",
    "uformer_blocks": "../uid_20261012_v2/uformer_blocks.py",
    "uid_r1_control": "uid_r1_control.py",
    "uid_r1_measure": "uid_r1_measure.py",
    "uid_r1_child": "uid_r1_child.py",
}


def expected_module_hashes(r1_manifest: dict[str, Any]) -> dict[str, str]:
    """Approved SHA-256 of every module the child imports, taken from the (already verified) r1 manifest."""
    by_path = {e["path"]: e["sha256"] for e in r1_manifest["files"]}
    missing = [p for p in MODULE_PATHS.values() if p not in by_path]
    if missing:
        raise ControlError(f"manifest_r1.json does not list runtime module(s) {missing}")
    return {name: by_path[path] for name, path in MODULE_PATHS.items()}


def ledger_status(primary: str | Path, mirror: str | Path, arms: list[str]) -> dict[str, Any]:
    """Read-only view of a ledger (no directory is created); absent roots read as an empty ledger."""
    records = Ledger(primary, mirror).read()
    s = summarize(records, arms)
    s["records"] = sorted(records)
    s["roots_exist"] = {"primary": Path(primary).exists(), "mirror": Path(mirror).exists()}
    return s


def verify_entries(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    bad = []
    for entry in manifest["files"]:
        p = root / entry["path"]
        actual = sha256_file(p) if p.is_file() else None
        if actual != entry["sha256"]:
            bad.append({"path": entry["path"], "expected": entry["sha256"], "actual": actual})
    return bad


def verify_package(approved_r1_manifest_sha256: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Identity of the frozen package and of revision r1, bound to the approved r1 manifest hash."""
    frozen_manifest, r1_manifest = FROZEN_DIR / "manifest.json", R1_DIR / "manifest_r1.json"
    cfg_raw = (FROZEN_DIR / "config.json").read_bytes()
    cfg = json.loads(cfg_raw.decode("utf-8"))
    ident = {
        "frozen_manifest_sha256": sha256_file(frozen_manifest),
        "r1_manifest_sha256": sha256_file(r1_manifest) if r1_manifest.is_file() else None,
        "config_raw_sha256": hashlib.sha256(cfg_raw).hexdigest(),
        "config_canonical_sha256": sha256_canonical(cfg),
    }
    problems = []
    if ident["frozen_manifest_sha256"] != FROZEN_MANIFEST_SHA256:
        problems.append("frozen manifest.json changed")
    if ident["r1_manifest_sha256"] != approved_r1_manifest_sha256:
        problems.append("manifest_r1.json does not have the approved SHA-256")
    if ident["config_raw_sha256"] != FROZEN_CONFIG_RAW_SHA256:
        problems.append("config.json raw bytes changed")
    if ident["config_canonical_sha256"] != FROZEN_CONFIG_CANONICAL_SHA256:
        problems.append("config.json canonical content changed")
    problems += [f"frozen file mismatch: {b}" for b in verify_entries(load_json(frozen_manifest), FROZEN_DIR)]
    if ident["r1_manifest_sha256"] is not None:
        problems += [f"r1 file mismatch: {b}" for b in verify_entries(load_json(r1_manifest), R1_DIR)]
    if problems:
        raise ControlError(f"package identity check failed: {problems}")
    execution = load_json(R1_DIR / "execution_r1.json")
    return ident, cfg, execution


def registered_plan(cfg: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    """Limits from the frozen config and protocol; only execution mechanics come from execution_r1.json."""
    if cfg.get("protocol_id") != REGISTERED_PROTOCOL_ID:
        raise ControlError(f"config protocol_id {cfg.get('protocol_id')!r} is not {REGISTERED_PROTOCOL_ID}")
    g = cfg["global_limits"]
    arms = []
    for arm in REGISTERED_ARMS:
        a = cfg[arm.lower().replace("-", "_")]
        arms.append(
            {"arm": arm, "max_updates": int(a["max_updates"]), "max_seconds": float(a["max_seconds"])}
        )
    plan = {
        "mode": "scientific",
        "arms": arms,
        "prerequisite": dict(REGISTERED_PREREQUISITE),
        "max_training_starts": int(g["max_training_starts"]),
        "max_invocations": 1,
        "aggregate_seconds": float(g["max_scientific_seconds"]),
        "rss_cap_bytes": int(g["max_rss_bytes"]),
        "rss_available_fraction": REGISTERED_RSS_AVAILABLE_FRACTION,
        "evidence_cap_bytes": REGISTERED_EVIDENCE_CAP_BYTES,
        "curve_every": REGISTERED_CURVE_EVERY,
        "cpu": {k: int(cfg["cpu_config"][k]) for k in ("intra_op_threads", "inter_op_threads")},
    }
    missing = [k for k in EXECUTION_KEYS if k not in execution]
    if missing:
        raise ControlError(f"execution_r1.json lacks {missing}")
    plan.update({k: execution[k] for k in EXECUTION_KEYS})
    return plan


def fixture_plan(
    fixture_cfg: dict[str, Any], tmp_root: Path, arms: list[dict[str, Any]], **over: Any
) -> dict[str, Any]:
    """An engineering plan: fixture config, fixture limits, ledgers and run root under ``tmp_root`` only."""
    if fixture_cfg.get("protocol_id") == REGISTERED_PROTOCOL_ID:
        raise ControlError("fixture plans cannot use the registered config")
    execution = load_json(R1_DIR / "execution_r1.json")
    plan = {
        "mode": "engineering",
        "arms": arms,
        "prerequisite": {},
        "max_training_starts": 2,
        "max_invocations": 1,
        "aggregate_seconds": 600.0,
        "rss_cap_bytes": 4 * 2**30,
        "rss_available_fraction": 0.5,
        "evidence_cap_bytes": 64 * 2**20,
        "curve_every": 2,
        "cpu": {"intra_op_threads": 2, "inter_op_threads": 1},
        "run_root": str(tmp_root / "run"),
        "ledger_mirror": str(tmp_root / "mirror"),
        "python_executable": execution["python_executable"],
        "torch_version": execution["torch_version"],
        "end_margin_s": 1.0,
        "eval_reserve_s": 2.0,
        "poll_interval_s": 0.1,
        "interrupt_grace_s": 8.0,
        "kill_wait_s": 0.5,
    }
    plan.update(over)
    return plan
