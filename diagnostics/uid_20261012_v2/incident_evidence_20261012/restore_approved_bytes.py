"""Narrowly scoped restoration of the two approved UID-20261012-v2 source files.

Reverses exactly the two post-freeze str_replace edits recorded in the 2026-10-12 session transcript (the only edits
made to these files after the manifest hashes were computed). Candidate bytes are hashed in memory; nothing is written
unless both candidates match the frozen manifest. Every file in the diagnostic tree is hashed before and after, so the
log shows which paths the restoration changed. Writes restoration_log.json next to this script.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

DIAG = Path(r"D:\projects-cd\cg-fedllm-reproduction\diagnostics\uid_20261012_v2")
INC = DIAG / "incident_evidence_20261012"

EDITS = [
    {
        "path": "uformer_blocks.py",
        "current_sha256": "9c68e47f375df7d000960f214452238eb8d717b2fb2f67b87f2feb072b8b8d6b",
        "expected_sha256": "3333f1862288e720d83aadbe8f171898f5f6943d2c0f58c9bd0132aba60b0747",
        # recorded forward edit: approved text -> edited text; {NL} is the file's own newline sequence
        "approved": "Adaptations; none changes the computation of the configured path "
        "(``checks/test_upstream_equivalence.py`` compares{NL}outputs and gradients with the pinned file):",
        "edited": "Adaptations; none is meant to change the computation of the configured path "
        "(``checks/check_upstream_equivalence.py``{NL}compares outputs and gradients with the pinned file):",
        # filename-only reversal: hashed for the record, never written
        "filename_only": "Adaptations; none is meant to change the computation of the configured path "
        "(``checks/test_upstream_equivalence.py``{NL}compares outputs and gradients with the pinned file):",
    },
    {
        "path": "uformer_ae.py",
        "current_sha256": "4f14df26f3173c8a473b73890b1a871bc35ede9dba87711761482e81a176a01c",
        "expected_sha256": "a3ad6e3f2b6e6f009158276f5534e2c68bf42781df18a1c1d9fcd0f5867c8606",
        "approved": "package code; ``checks/test_resnet_sampling_equivalence.py`` compares them with the package.",
        "edited": "package code; ``checks/check_resnet_sampling.py`` compares them with the package.",
        "filename_only": None,
    },
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_hashes() -> dict:
    return {p.relative_to(DIAG).as_posix(): sha(p.read_bytes()) for p in sorted(DIAG.rglob("*")) if p.is_file()}


def main() -> int:
    log = {"before_tree": tree_hashes(), "files": []}
    plans = []
    for e in EDITS:
        p = DIAG / e["path"]
        data = p.read_bytes()
        nl = "\r\n" if b"\r\n" in data else "\n"

        def enc(s: str, nl: str = nl) -> bytes:
            return s.replace("{NL}", nl).encode("utf-8")

        edited, approved = enc(e["edited"]), enc(e["approved"])
        rec = {
            "path": e["path"],
            "before_sha256": sha(data),
            "before_bytes": len(data),
            "unchanged_since_preservation": sha(data) == e["current_sha256"],
            "newline": "CRLF" if nl == "\r\n" else "LF",
            "crlf_count": data.count(b"\r\n"),
            "edited_occurrences": data.count(edited),
            "approved_occurrences_before": data.count(approved),
        }
        candidate = data.replace(edited, approved, 1) if rec["edited_occurrences"] == 1 else None
        rec["candidate_sha256"] = sha(candidate) if candidate is not None else None
        rec["candidate_bytes"] = len(candidate) if candidate is not None else None
        rec["expected_sha256"] = e["expected_sha256"]
        rec["candidate_matches_expected"] = rec["candidate_sha256"] == e["expected_sha256"]
        if e["filename_only"] and rec["edited_occurrences"] == 1:
            fo = data.replace(edited, enc(e["filename_only"]), 1)
            rec["filename_only_reversal_sha256"] = sha(fo)
            rec["filename_only_reversal_bytes"] = len(fo)
            rec["filename_only_matches_expected"] = sha(fo) == e["expected_sha256"]
        log["files"].append(rec)
        plans.append((p, candidate, rec))
    ok = all(r["unchanged_since_preservation"] and r["candidate_matches_expected"] for _, _, r in plans)
    log["all_candidates_match"] = ok
    log["written"] = False
    if ok:
        for p, candidate, rec in plans:
            tmp = p.with_name(p.name + ".restore.tmp")
            with open(tmp, "xb") as fh:
                fh.write(candidate)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, p)
            after = p.read_bytes()
            rec["after_sha256"] = sha(after)
            rec["after_bytes"] = len(after)
        log["written"] = True
        log["after_tree"] = tree_hashes()
        keys = set(log["before_tree"]) | set(log["after_tree"])
        log["changed_paths"] = sorted(k for k in keys if log["before_tree"].get(k) != log["after_tree"].get(k))
    (INC / "restoration_log.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {k: v for k, v in log.items() if k not in ("before_tree", "after_tree")}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
