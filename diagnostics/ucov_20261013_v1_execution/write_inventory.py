"""Write ``EVIDENCE_INVENTORY_SHA256.txt`` LAST: SHA-256 and size of every file in this execution directory (this
inventory excluded), the policy directory, the UCOV package (with its ledger mirror) and the UCOV run root.
LF bytes, exclusive create. ``python -B write_inventory.py``.
"""

import hashlib
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOTS = [
    HERE,
    HERE.parent / "upp_20261013_v1",
    HERE.parent / "ucov_20261013_v1",
    Path("D:/cgfed-runs/ucov_20261013_v1"),
]


def main() -> int:
    out = HERE / "EVIDENCE_INVENTORY_SHA256.txt"
    lines = [
        f"recorded: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}  (execution evidence, policy, package with ledger mirror, "
        "run root; this file excluded)"
    ]
    for root in ROOTS:
        for p in sorted(q for q in root.rglob("*") if q.is_file() and q.resolve() != out.resolve()):
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()} {p.stat().st_size:>8}  {p.resolve()}")
    data = ("\n".join(lines) + "\n").encode("utf-8")
    with out.open("xb") as fh:
        fh.write(data)
    print(
        f"{out.name}: {len(lines) - 1} entries, {len(data)} bytes, sha256={hashlib.sha256(data).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
