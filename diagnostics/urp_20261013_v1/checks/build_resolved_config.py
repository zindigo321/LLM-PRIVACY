"""Write ``resolved_config.json`` once: the URP-v1 scientific configuration resolved from the pinned sources (frozen
UID config, r1 execution file, URP config, registered input files). Refuses to overwrite.
``python -B checks/build_resolved_config.py``."""

import hashlib
import json
import sys
from pathlib import Path

URP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(URP))

import urp_control as U  # noqa: E402


def main() -> int:
    out = URP / U.RESOLVED_NAME
    if out.exists():
        raise SystemExit(f"{out} exists; never overwritten")
    resolved = U.resolve(U.load_urp_config())
    data = (json.dumps(resolved, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with out.open("xb") as fh:
        fh.write(data)
    print(f"wrote {out} ({len(data)} bytes) sha256={hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
