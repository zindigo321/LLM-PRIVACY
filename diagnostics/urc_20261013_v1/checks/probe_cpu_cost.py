"""ENGINEERING PROBE: CPU cost and run-to-run determinism of one forward+backward of the full-size R-FIT model.

The architecture is the registered one (``resolved_config.json`` model block), but with an engineering model seed
(31337, not 1) and an engineering input (data seed 777, not 20261012). No optimizer is created and no ``.step()`` is
taken; each pass uses a fresh model, so no state accumulates. Threads and determinism are set with the frozen
``configure_cpu`` and the registered thread counts, before any tensor work.

Measures (a) seconds per forward+backward, the cost of one update without its Adam step, and (b) whether repeated
identical passes give bit-identical outputs and gradients. ``python -B checks/probe_cpu_cost.py [passes]``.
"""

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

URC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(URC))

import urc_control as U  # noqa: E402


def main() -> int:
    resolved = U.load_resolved()
    import uid_common as C

    cpu = C.configure_cpu(resolved["cpu"]["intra_op_threads"], resolved["cpu"]["inter_op_threads"])
    import psutil
    import torch

    probe = {**resolved, "model_init_seed": 31337, "input": {**resolved["input"], "data_seed": 777}}
    x = C.make_input(probe)
    passes = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    times, digests = [], []
    for _ in range(passes):
        m = U.build_model(probe).train()
        t = time.perf_counter()
        y = C.reconstruct(m, x)
        torch.nn.functional.mse_loss(y, x).backward()
        times.append(time.perf_counter() - t)
        h = hashlib.sha256(y.detach().numpy().tobytes())
        for _, p in sorted(m.named_parameters()):
            h.update(p.grad.numpy().tobytes())
        digests.append(h.hexdigest())
    med = statistics.median(times[1:] if len(times) > 1 else times)
    report = {
        "engineering_only": {"model_seed": 31337, "data_seed": 777, "optimizer": None},
        "cpu": cpu,
        "passes": passes,
        "seconds_per_forward_backward": times,
        "median_excluding_first_s": med,
        "projected_3000_updates_s_without_adam": med * resolved["max_updates"],
        "arm_limit_s": resolved["max_seconds"],
        "bit_identical_across_passes": len(set(digests)) == 1,
        "rss_bytes_after": psutil.Process().memory_info().rss,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["bit_identical_across_passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
