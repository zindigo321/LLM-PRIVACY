"""ULR-20261015-v1 evidence inventory (written last). ``python -B write_inventory.py``.

Writes ``EVIDENCE_INVENTORY_SHA256.txt``: SHA-256, size and absolute path of every file of the package directory,
this execution directory (except the inventory itself) and the run root, plus the total retained new-evidence bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

EXE = Path(__file__).resolve().parent
ROOTS = (EXE.parent / "ulr_20261015_v1", EXE, Path("D:/cgfed-runs/ulr_20261015_v1"))


def main() -> int:
    lines, total = [], 0
    for root in ROOTS:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name != "EVIDENCE_INVENTORY_SHA256.txt":
                n = p.stat().st_size
                total += n
                lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()} {n} {p.as_posix()}")
    head = f"# ULR-20261015-v1 evidence inventory: {len(lines)} files, {total} bytes (cap 536870912)"
    (EXE / "EVIDENCE_INVENTORY_SHA256.txt").open("x", encoding="utf-8", newline="\n").write(
        "\n".join([head, *lines]) + "\n"
    )
    print(head)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
