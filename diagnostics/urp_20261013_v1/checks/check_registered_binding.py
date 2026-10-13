"""ENGINEERING CHECK: the registered model binding and input path, without any update.

With the registered CPU settings (frozen ``configure_cpu``, 24/24 threads, deterministic algorithms), this builds the
registered model through ``urp_control.build_model`` (the frozen ``uid_common.build_model(cfg, "U-FIT")``) and
requires its initial state hash to equal the U-FIT record's ``model_state_sha256_init``. It loads the registered patch
through ``urp_control.load_input`` and re-checks the input and both baselines against ``input_expected``. It then makes
one forward pass on a disposable copy, under ``torch.no_grad()`` and the RNG fork of the r1 ``isolated``: the output
shape, its Tanh range and finiteness, and the latent shape are recorded. No metric of this output is computed or
printed, so no model performance on the registered patch is observed.

No optimizer is created, no parameter is updated, nothing is saved. The fingerprint of the model state and of the
global RNG must be unchanged afterwards. ``python -B checks/check_registered_binding.py``; prints JSON; exit 0 when
every check holds.
"""

import json
import sys
from pathlib import Path

URP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(URP))

import urp_control as U  # noqa: E402


def main() -> int:
    resolved = U.load_resolved()
    U.check_pinned()
    import uid_common as C

    cpu = C.configure_cpu(resolved["cpu"]["intra_op_threads"], resolved["cpu"]["inter_op_threads"])
    import torch
    import uid_r1_measure as M

    model = U.build_model(resolved)
    init_sha = C.state_sha256(model)
    x, x_n, digest = U.load_input(resolved)
    rec = {
        "input": {"original": C.describe_input(x), "normalized": C.describe_input(x_n)},
        "baselines": {"original": C.baselines(x), "normalized": C.baselines(x_n)},
    }
    checks = U.input_checks(rec, resolved, float(x_n.abs().max()))
    before = M.state_fingerprint(model)
    with M.isolated(model, "eval") as twin, torch.no_grad():
        z = twin.encode(x_n)
        y_n = twin.decode(z)
    after = M.state_fingerprint(model)
    report = {
        "cpu": cpu,
        "procedure": "fresh registered model; one forward on a disposable copy; no optimizer, no update, no save",
        "model_class": type(model).__name__,
        "model_config": model.cfg.to_dict(),
        "parameters": model.num_parameters(),
        "state_sha256_init": init_sha,
        "u_fit_recorded_state_sha256_init": resolved["reference"]["model_state_sha256_init"],
        "init_equals_u_fit_record": init_sha == resolved["reference"]["model_state_sha256_init"],
        "patch_file_sha256": digest,
        "input_checks_all_equal": checks["all_equal"],
        "forward": {
            "latent_shape": list(z.shape),
            "latent_element_ratio": z.numel() / x_n.numel(),
            "output_shape": list(y_n.shape),
            "output_finite": bool(torch.isfinite(y_n).all()),
            "output_within_tanh_range": bool((y_n.abs() < 1).all()),
        },
        "state_and_rng_unchanged": before == after,
        "optimizer_created": False,
    }
    report["all_ok"] = (
        report["init_equals_u_fit_record"]
        and report["input_checks_all_equal"]
        and report["forward"]["output_shape"] == [1, 1, 32, 32]
        and report["forward"]["latent_shape"] == [1, 64, 32, 32]
        and report["forward"]["output_finite"]
        and report["state_and_rng_unchanged"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
