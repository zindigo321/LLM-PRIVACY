"""ENGINEERING CHECK: the registered model binding and the four tile input paths, without any update.

With the registered CPU settings (frozen ``configure_cpu``, 24/24 threads, deterministic algorithms), this:
- builds the registered model through ``ucov_control.build_model`` (the frozen ``uid_common.build_model(cfg,
  "U-FIT")`` via the frozen URP binding), and requires its initial state hash to equal the U-FIT record's;
- loads each registered tile through ``ucov_control.load_tile`` and re-checks the input and both baselines against
  the registered values;
- for each tile, makes one forward pass on a disposable copy, under ``torch.no_grad()`` and the RNG fork of the r1
  ``isolated``.

Only shapes, finiteness and the Tanh range are recorded. No metric of any output is computed or printed, so no model
performance on a registered tile is observed. No optimizer is created, no parameter is updated and nothing is saved.
The fingerprint of the model state and of the global RNG must be unchanged afterwards.

``python -B checks/check_registered_binding.py``; prints JSON; exit 0 when every check holds.
"""

import json
import sys
from pathlib import Path

UCOV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UCOV))

import ucov_control as U  # noqa: E402


def main() -> int:
    resolved = U.load_resolved()
    U.check_pinned()
    import uid_common as C

    cpu = C.configure_cpu(resolved["cpu"]["intra_op_threads"], resolved["cpu"]["inter_op_threads"])
    import torch
    import uid_r1_measure as M

    model = U.build_model(resolved)
    init_sha = C.state_sha256(model)
    before = M.state_fingerprint(model)
    arms = {}
    for arm in U.ARMS:
        x, x_n, digest = U.load_tile(resolved, arm)
        rec = {
            "input": {"original": C.describe_input(x), "normalized": C.describe_input(x_n)},
            "baselines": {"original": C.baselines(x), "normalized": C.baselines(x_n)},
        }
        checks = U.input_checks(rec, resolved["inputs_expected"][arm], float(x_n.abs().max()))
        with M.isolated(model, "eval") as twin, torch.no_grad():
            z = twin.encode(x_n)
            y_n = twin.decode(z)
        arms[arm] = {
            "tile_file_sha256": digest,
            "input_checks_all_equal": checks["all_equal"],
            "latent_shape": list(z.shape),
            "output_shape": list(y_n.shape),
            "output_finite": bool(torch.isfinite(y_n).all()),
            "output_within_tanh_range": bool((y_n.abs() < 1).all()),
        }
    after = M.state_fingerprint(model)
    report = {
        "cpu": cpu,
        "procedure": "fresh registered model; one forward per tile on a disposable copy; no optimizer, no metric",
        "parameters": model.num_parameters(),
        "state_sha256_init": init_sha,
        "init_equals_u_fit_record": init_sha == resolved["reference"]["model_state_sha256_init"],
        "arms": arms,
        "state_and_rng_unchanged": before == after,
        "optimizer_created": False,
        "primary_policy": {k: resolved["primary_policy"][k] for k in ("policy_id", "policy_sha256")},
    }
    report["all_ok"] = (
        report["init_equals_u_fit_record"]
        and report["parameters"]["total"] == 685507
        and all(
            a["input_checks_all_equal"]
            and a["latent_shape"] == [1, 64, 32, 32]
            and a["output_shape"] == [1, 1, 32, 32]
            and a["output_finite"]
            for a in arms.values()
        )
        and report["state_and_rng_unchanged"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
