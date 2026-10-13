"""ENGINEERING PROBE: CPU cost and run-to-run determinism of forward+backward for the frozen U-FIT architecture.

The architecture is built from the frozen config, but with an engineering model seed (not 1) and an engineering
input (not data seed 20261012). No optimizer is created and no ``.step()`` is taken. Each pass computes a
forward/backward and discards the gradients. Threads and determinism are set via the frozen ``configure_cpu`` with the
registered thread counts, before any tensor work.

The probe measures (a) seconds per forward+backward, i.e. the cost of one update without its Adam step, for budget
feasibility, and (b) whether two identical forward/backward passes give bit-identical outputs and gradients.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

R1 = Path(__file__).resolve().parents[1]
FROZEN = R1.parent / "uid_20261012_v2"
sys.path[:0] = [str(FROZEN), str(R1)]


def main() -> int:
    import uid_common as C  # frozen

    cfg = json.loads((FROZEN / "config.json").read_text(encoding="utf-8"))
    cpu = C.configure_cpu(cfg["cpu_config"]["intra_op_threads"], cfg["cpu_config"]["inter_op_threads"])
    import psutil
    import torch

    probe_cfg = json.loads(json.dumps(cfg))
    probe_cfg["model_init_seed"] = 31337  # engineering seed
    probe_cfg["input"]["data_seed"] = 777  # engineering input
    model = C.build_model(probe_cfg, "U-FIT")
    model.train()
    x = C.make_input(probe_cfg)
    passes, times, digests = int(sys.argv[1]) if len(sys.argv) > 1 else 5, [], []
    state0 = C.state_sha256(model)
    for _ in range(passes):
        # a fresh copy each pass: no state accumulates (BN buffers would otherwise drift between passes)
        m = C.build_model(probe_cfg, "U-FIT")
        m.train()
        t = time.perf_counter()
        y = C.reconstruct(m, x)
        loss = torch.nn.functional.mse_loss(y, x)
        loss.backward()
        times.append(time.perf_counter() - t)
        h = hashlib.sha256(y.detach().numpy().tobytes())
        for _, p in sorted(m.named_parameters()):
            h.update(p.grad.numpy().tobytes())
        digests.append(h.hexdigest())
    rss = psutil.Process().memory_info()
    report = {
        "cpu_observed": cpu,
        "passes": passes,
        "seconds_per_forward_backward": times,
        "first_pass_s": times[0],
        "median_after_first_s": sorted(times[1:])[len(times[1:]) // 2] if passes > 1 else None,
        "bit_identical_across_passes": len(set(digests)) == 1,
        "digests": digests,
        "state_sha256_of_probe_model_unchanged": C.state_sha256(model) == state0,
        "rss_bytes_end": rss.rss,
        "os_peak_wset_bytes": getattr(rss, "peak_wset", None),
        "registered_budget": {"updates": cfg["u_fit"]["max_updates"], "seconds": cfg["u_fit"]["max_seconds"]},
        "note": "forward+backward only; the Adam step and measurement overhead are not included",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
