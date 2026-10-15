"""Verify published ULR evidence bytes, without executing scientific code."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-tree", help="Git tree/ref to check; INDEX checks staged blobs.")
    args = parser.parse_args()
    archive = Path(__file__).resolve().parent
    repo = archive.parents[1]
    index = json.loads((archive / "archive_index.json").read_text(encoding="utf-8"))
    entries = index["published_evidence"]
    data_by_path = {}
    if args.git_tree:
        specs = [
            (":" if args.git_tree == "INDEX" else args.git_tree + ":") + e["repository_path"] for e in entries
        ]
        result = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=repo,
            input=("\n".join(specs) + "\n").encode(),
            capture_output=True,
            check=True,
        )
        pos = 0
        for entry in entries:
            end = result.stdout.index(b"\n", pos)
            fields = result.stdout[pos:end].split()
            if len(fields) != 3 or fields[1] != b"blob":
                raise RuntimeError(f"Missing/non-blob Git entry: {entry['repository_path']}")
            size = int(fields[2])
            start = end + 1
            data_by_path[entry["repository_path"]] = result.stdout[start : start + size]
            if result.stdout[start + size : start + size + 1] != b"\n":
                raise RuntimeError("Invalid git cat-file framing")
            pos = start + size + 1
        if pos != len(result.stdout):
            raise RuntimeError("Unexpected extra git cat-file output")
    else:
        for entry in entries:
            path = (repo / entry["repository_path"]).resolve()
            path.relative_to(repo.resolve())
            data_by_path[entry["repository_path"]] = path.read_bytes()

    failures = []
    for entry in entries:
        data = data_by_path[entry["repository_path"]]
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            failures.append(entry["repository_path"])
    pkg = "diagnostics/ulr_20261015_v1/"
    manifest_data = data_by_path[pkg + "manifest_ulr.json"]
    if hashlib.sha256(manifest_data).hexdigest() != index["frozen_manifest_sha256"]:
        failures.append("frozen_manifest_sha256")
    manifest = json.loads(manifest_data)
    for entry in manifest["files"]:
        key = pkg + entry["path"]
        data = data_by_path.get(key, b"")
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            failures.append("manifest:" + key)
    if len(entries) != index["published_evidence_file_count"]:
        failures.append("published_evidence_file_count")
    if sum(len(data_by_path[e["repository_path"]]) for e in entries) != index["published_evidence_bytes"]:
        failures.append("published_evidence_bytes")
    print(
        json.dumps(
            {
                "all_ok": not failures,
                "mode": args.git_tree or "working_tree",
                "published_files_checked": len(entries),
                "manifest_entries_checked": len(manifest["files"]),
                "failures": failures,
                "scientific_execution": False,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
