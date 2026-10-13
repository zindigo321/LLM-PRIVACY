"""ENGINEERING FIXTURE: run the r1 supervisor once on the fixture config in a fresh temporary root.

``python -B checks/fixture_run.py <tmp_root> <arms_json> [<plan_overrides_json> [<hooks_json>]]``; prints the
invocation summary as JSON. Never touches the registered config, run root or ledger mirror.
"""

import json
import sys
from pathlib import Path

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))

import uid_r1_control as K  # noqa: E402
import uid_r1_supervise as S  # noqa: E402

FIXTURE = R1 / "checks" / "fixture_config.json"


def main() -> int:
    tmp_root = Path(sys.argv[1])
    arms = json.loads(sys.argv[2])
    over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    hooks = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    for forbidden in (Path("D:/cgfed-runs").resolve(), (R1 / "scientific_ledger_mirror").resolve()):
        if forbidden in tmp_root.resolve().parents or tmp_root.resolve() == forbidden:
            raise SystemExit("fixture roots must not be under the scientific run root or mirror")
    plan = K.fixture_plan(K.load_json(FIXTURE), tmp_root, arms, **over)
    summary = S.invoke(plan, FIXTURE, [str(K.FROZEN_DIR), str(R1)], {"fixture": True}, hooks=hooks)
    print(json.dumps(K.json_safe(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except K.ControlError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
