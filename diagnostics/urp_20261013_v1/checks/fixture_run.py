"""ENGINEERING FIXTURE: run the URP supervisor once on the fixture config in a fresh temporary root.

``python -B checks/fixture_run.py <tmp_root> <arms_json> [<plan_overrides_json> [<hooks_json>]]``; prints the
invocation summary as JSON. Never touches the registered config, input, run root or ledger mirror.
"""

import json
import sys
from pathlib import Path

URP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(URP))

import urp_control as U  # noqa: E402
import urp_supervise as S  # noqa: E402

K = U.K
FIXTURE = URP / "checks" / "fixture_config.json"


def main() -> int:
    tmp_root = Path(sys.argv[1])
    arms = json.loads(sys.argv[2])
    over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    hooks = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    registered = U.load_urp_config()["execution"]
    for forbidden in (Path("D:/cgfed-runs"), Path(registered["run_root"]), Path(registered["ledger_mirror"])):
        f = forbidden.resolve()
        if f in tmp_root.resolve().parents or tmp_root.resolve() == f:
            raise SystemExit("fixture roots must not be under a scientific run root or ledger mirror")
    plan = K.fixture_plan(K.load_json(FIXTURE), tmp_root, arms, **over)
    summary = S.invoke(
        plan, FIXTURE, [str(URP), str(U.FROZEN_DIR), str(U.R1_DIR)], {"fixture": True}, hooks=hooks
    )
    print(json.dumps(K.json_safe(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except K.ControlError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
