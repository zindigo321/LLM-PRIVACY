"""UID-20261012-v2 runner: single-invocation training and final evaluation under resource limits.

One training start per arm (U-CHECK and U-FIT); step-zero, per-100, and final metrics; train/eval mode pair
at the end; structured JSON evidence; graceful resource stop. No automatic retries or epoch restarts.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import nn, optim

import uid_common as C

log = logging.getLogger(__name__)


class Runner:
    def __init__(self, cfg: dict[str, Any], arm: str, root: Path) -> None:
        self.cfg = cfg
        self.arm = arm
        self.arm_cfg = cfg[arm.lower().replace("-", "_")]
        self.root = Path(root)
        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.x: torch.Tensor | None = None
        self.results: dict[str, Any] = {"arm": arm, "status": "NOT_RUN"}
        self.start_t = time.time()
        self.last_gc_t = self.start_t
        self.steps_done = 0
        self.stop_reason: str | None = None
        self.process = psutil.Process()

    def check_resources(self) -> None:
        """Check wall time and RSS; raise ResourceStop if a limit is exceeded."""
        now = time.time()
        elapsed = now - self.start_t
        if elapsed > float(self.cfg["global_limits"]["max_scientific_seconds"]):
            raise ResourceStop(f"wall time {elapsed:.1f}s > limit")
        rss_bytes = self.process.memory_info().rss
        limit_bytes = int(self.cfg["global_limits"]["max_rss_bytes"])
        if rss_bytes > limit_bytes:
            raise ResourceStop(f"RSS {rss_bytes / (1 << 30):.2f} GiB > {limit_bytes / (1 << 30):.2f} GiB")
        if now - self.last_gc_t > 30:
            gc.collect()
            self.last_gc_t = now

    def record_step(self, step: int, x_hat: torch.Tensor, loss_value: float) -> None:
        m = C.metrics(x_hat.detach(), self.x)
        self.results.setdefault("steps", []).append({"step": step, "loss": loss_value, "metrics": m})

    def train_loop(self) -> None:
        """Train from step 1 to ``max_updates``, record every 100 updates plus the final."""
        self.model.train()
        max_updates = int(self.arm_cfg["max_updates"])
        log_every = 100
        for step in range(1, max_updates + 1):
            self.check_resources()
            self.optimizer.zero_grad()
            x_hat = C.reconstruct(self.model, self.x)
            loss = nn.functional.mse_loss(x_hat, self.x)
            loss.backward()
            self.optimizer.step()
            self.steps_done = step
            if step % log_every == 0 or step == max_updates:
                self.record_step(step, x_hat, float(loss))

    def final_evaluation(self) -> dict[str, Any]:
        """Paired train/eval reconstruction after the last optimizer update, using independent model copies."""
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        train_model = C.build_model(self.cfg, self.arm)
        train_model.load_state_dict(state)
        train_model.train()
        eval_model = C.build_model(self.cfg, self.arm)
        eval_model.load_state_dict(state)
        eval_model.eval()
        with torch.no_grad():
            train_hat = C.reconstruct(train_model, self.x)
            eval_hat = C.reconstruct(eval_model, self.x)
        return {
            "train_mode": C.metrics(train_hat, self.x),
            "eval_mode": C.metrics(eval_hat, self.x),
            "train_eval_sse_delta": float((train_hat - eval_hat).square().sum()),
        }

    def run(self) -> dict[str, Any]:
        """The main entry: build model, input and optimizer; train; final eval; save evidence."""
        try:
            self.check_resources()
            self.model = C.build_model(self.cfg, self.arm)
            self.x = C.make_input(self.cfg).requires_grad_(False)
            lr = float(self.arm_cfg["optimizer"]["lr"])
            betas = tuple(float(v) for v in self.arm_cfg["optimizer"]["betas"])
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas, eps=1e-8, weight_decay=0)
            self.results["model_params"] = self.model.num_parameters() if hasattr(self.model, "num_parameters") else {}
            self.results["model_state_sha256_init"] = C.state_sha256(self.model)
            self.results["input"] = C.describe_input(self.x)
            self.results["baselines"] = C.baselines(self.x)
            with torch.no_grad():
                x_hat_init = C.reconstruct(self.model, self.x)
            self.record_step(0, x_hat_init, float(nn.functional.mse_loss(x_hat_init, self.x)))
            log.info(f"{self.arm} step 0 recorded")
            self.train_loop()
            self.results["model_state_sha256_final"] = C.state_sha256(self.model)
            self.results["final"] = self.final_evaluation()
            self.results["status"] = self._verdict()
            self.results["wall_seconds"] = time.time() - self.start_t
            log.info(f"{self.arm} {self.results['status']} in {self.results['wall_seconds']:.1f}s")
        except ResourceStop as e:
            self.stop_reason = str(e)
            self.results["status"] = "RESOURCE_STOP"
            self.results["stop_reason"] = self.stop_reason
            self.results["wall_seconds"] = time.time() - self.start_t
            log.warning(f"{self.arm} stopped: {self.stop_reason}")
        except Exception as e:
            self.results["status"] = "EXECUTION_ERROR"
            self.results["error"] = {"type": type(e).__name__, "message": str(e)}
            self.results["wall_seconds"] = time.time() - self.start_t
            log.exception(f"{self.arm} error")
        finally:
            self.results["steps_done"] = self.steps_done
        out = self.root / f"{self.arm.lower().replace('-', '_')}.json"
        C.write_json_durable(out, self.results)
        return self.results

    def _verdict(self) -> str:
        """PASS / FIT_FAIL from the final eval-mode metrics."""
        m = self.results["final"]["eval_mode"]
        criteria = self.cfg["criteria"]
        if not m["finite"]:
            return "COMPLETED_FIT_FAIL"
        if C.passes(m, criteria):
            return "COMPLETED_PASS"
        return "COMPLETED_FIT_FAIL"


class ResourceStop(RuntimeError):
    """A resource limit was reached; training stops gracefully with RESOURCE_STOP."""


def main() -> None:
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="approved config.json")
    parser.add_argument("--arm", required=True, choices=["U-CHECK", "U-FIT"])
    parser.add_argument("--root", required=True, help="output directory")
    args = parser.parse_args()
    cfg, _, _ = C.load_config(args.config)
    runner = Runner(cfg, args.arm, Path(args.root))
    result = runner.run()
    sys.exit(0 if result["status"] in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL") else 1)


if __name__ == "__main__":
    main()
