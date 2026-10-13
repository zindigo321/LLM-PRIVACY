"""ENGINEERING FIXTURE: run the UCOV supervisor once on the fixture config in a fresh temporary root.

``python -B checks/fixture_run.py <tmp_root> <arms_json> [<plan_overrides_json> [<hooks_json>]]``; prints the
invocation summary as JSON. Never touches the registered config, inputs, run root or ledger mirror.
"""

import json
import sys
from pathlib import Path

UCOV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UCOV))

import ucov_control as U  # noqa: E402
import ucov_supervise as S  # noqa: E402

K = U.K
FIXTURE = UCOV / "checks" / "fixture_config.json"


def main() -> int:
    tmp_root = Path(sys.argv[1])
    arms = json.loads(sys.argv[2])
    over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    hooks = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    registered = U.load_ucov_config()["execution"]
    for forbidden in (Path("D:/cgfed-runs"), Path(registered["run_root"]), Path(registered["ledger_mirror"])):
        f = forbidden.resolve()
        if f in tmp_root.resolve().parents or tmp_root.resolve() == f:
            raise SystemExit("fixture roots must not be under a scientific run root or ledger mirror")
    pol = K.load_json(FIXTURE)["primary_policy"]
    base = {
        "primary_policy_id": pol["policy_id"],
        "primary_policy_sha256": pol["policy_sha256"],
        "max_training_starts": len(arms),
    }
    plan = K.fixture_plan(K.load_json(FIXTURE), tmp_root, arms, **{**base, **over})
    sys_path = [str(UCOV), str(U.URP_DIR), str(U.UPP_DIR), str(U.FROZEN_DIR), str(U.R1_DIR)]
    summary = S.invoke(plan, FIXTURE, sys_path, {"fixture": True}, hooks=hooks)
    print(json.dumps(K.json_safe(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except K.ControlError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
