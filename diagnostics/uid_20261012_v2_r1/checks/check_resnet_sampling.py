"""ENGINEERING CHECK: the frozen AE's re-declared sampling layers vs the package's ResNet-3 sampling layers.

``uformer_ae.py`` says its sampling layers are this repository's ResNet-3 reconstruction, re-declared so that the
diagnostic never imports package code. This check (a separate process; the r1 runner never imports ``cg_fedllm``)
imports ``cg_fedllm.compression.autoencoder`` read-only from the repository ``src`` and compares, with mapped weights,
the encoder sampling stack and the decoder sampling stack layer by layer: module types and hyper-parameters, forward
output and gradients. The ResNet residual blocks are not compared: the BasicUformer cores replace them by design.

Fixture tensors and an engineering seed only; no optimizer; nothing registered.
"""

import json
import sys
from pathlib import Path

import torch

R1 = Path(__file__).resolve().parents[1]
FROZEN = R1.parent / "uid_20261012_v2"
REPO_SRC = R1.parents[1] / "src"
TOL = {"dtype": "float64", "max_abs": 1e-12}  # declared before running


def describe(m: torch.nn.Module) -> list:
    out = []
    for layer in m:
        d = {"type": type(layer).__name__}
        for a in (
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "groups",
            "num_features",
            "eps",
            "momentum",
            "affine",
            "track_running_stats",
            "inplace",
        ):
            if hasattr(layer, a):
                d[a] = getattr(layer, a)
        if hasattr(layer, "bias") and isinstance(layer.bias, (torch.Tensor, type(None))):
            d["bias"] = layer.bias is not None
        out.append(d)
    return out


def main() -> int:
    sys.path[:0] = [str(FROZEN), str(REPO_SRC)]
    import cg_fedllm.compression.autoencoder as P  # package ResNet-3 (read-only import)
    from uformer_ae import UformerAEConfig, UformerAutoEncoder  # frozen

    cfg = UformerAEConfig(
        **json.loads((FROZEN / "config.json").read_text(encoding="utf-8"))["u_fit"]["model"]
    )
    pcfg = P.StridedResNetAEConfig(down_strides=cfg.down_strides)
    torch.manual_seed(20261015)  # engineering-only seed
    ufo, res = UformerAutoEncoder(cfg).double(), P.ResNetAutoEncoder(pcfg).double()
    n_down = 4 + 3 * len(cfg.down_channels)  # stem (4 modules) + 3 per stage
    pairs = {
        "encoder_sampling": (ufo.encoder.sampling, torch.nn.Sequential(*list(res.encoder.net)[:n_down])),
        # the package decoder starts with its num_res_blocks residual blocks, which the Uformer cores replace
        "decoder_sampling": (
            ufo.decoder.sampling,
            torch.nn.Sequential(*list(res.decoder.net)[pcfg.num_res_blocks :]),
        ),
    }
    g = torch.Generator().manual_seed(20261016)
    report = {
        "tolerances_declared_before_run": TOL,
        "package_file": str(Path(P.__file__).resolve()),
        "cases": [],
    }
    for name, (a, b) in pairs.items():
        da, db = describe(a), describe(b)
        with torch.no_grad():
            for p in b.parameters():
                p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.3)
        a.load_state_dict(b.state_dict(), strict=True)
        c_in = cfg.in_channels if name == "encoder_sampling" else cfg.latent_channels
        x = torch.randn((2, c_in, *cfg.input_hw), generator=g, dtype=torch.float64)
        res_case = {"case": name, "layers": len(da), "structure_identical": da == db}
        for mode in ("train", "eval"):
            a.train(mode == "train")
            b.train(mode == "train")
            xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
            ya, yb = a(xa), b(xb)
            w = torch.randn(ya.shape, generator=g, dtype=torch.float64)
            (ya * w).sum().backward()
            (yb * w).sum().backward()
            fwd = float((ya - yb).abs().max())
            gin = float((xa.grad - xb.grad).abs().max())
            gp = max(float((pa.grad - pb.grad).abs().max()) for pa, pb in zip(a.parameters(), b.parameters()))
            res_case[mode] = {"forward_max_abs": fwd, "input_grad_max_abs": gin, "param_grad_max_abs": gp}
            for p in (*a.parameters(), *b.parameters()):
                p.grad = None
        res_case["ok"] = res_case["structure_identical"] and all(
            v <= TOL["max_abs"] for m in ("train", "eval") for v in res_case[m].values()
        )
        report["cases"].append(res_case)
    report["all_ok"] = all(c["ok"] for c in report["cases"])
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
