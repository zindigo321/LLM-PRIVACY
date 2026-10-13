"""ENGINEERING CHECK: numerical correspondence of the vendored blocks with the pinned upstream Uformer.

Compares, with identical mapped weights and an unregistered synthetic fixture, the configured path of
``uformer_blocks`` (frozen, imported unchanged) with ``third_party/uformer_65fc970/model.py`` (pinned, executed
unmodified from its bytes). Upstream imports are satisfied without installing anything:

* einops: the einops 0.8.1 wheel downloaded during PREPARE, imported read-only from the zip (SHA-256 checked);
* timm: three stand-ins. ``DropPath`` is never constructed on the compared path (drop_path = 0); ``to_2tuple`` is
  the plain pair; ``trunc_normal_`` only affects initialisation, which the weight mapping overwrites.

No optimizer, no registered model, input or ledger. Tolerances are fixed here, before any comparison is run.
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import torch

R1 = Path(__file__).resolve().parents[1]
FROZEN = R1.parent / "uid_20261012_v2"
UPSTREAM = FROZEN / "third_party" / "uformer_65fc970" / "model.py"
UPSTREAM_SHA256 = "03a2c6e50b4342e03a457dd10537c14d0f832ce186da49c781fbd9d7027904a9"
EINOPS_WHEEL = Path.home() / "AppData/Local/Temp/uid_20261012_v2/wheels/einops-0.8.1-py3-none-any.whl"
EINOPS_WHEEL_SHA256 = "919387eb55330f5757c6bea9165c5ff5cfe63a642682ea788a6d472576d81737"
# declared before running: float64 comparisons of forward outputs and parameter gradients
TOL = {"dtype": "float64", "max_abs": 1e-12, "max_rel_l2": 1e-12}
# ``torch.meshgrid`` without ``indexing`` (upstream) emits a UserWarning; it equals indexing="ij" (vendored)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_upstream():
    if sha(UPSTREAM) != UPSTREAM_SHA256 or sha(EINOPS_WHEEL) != EINOPS_WHEEL_SHA256:
        raise SystemExit("pinned upstream file or einops wheel does not have the recorded SHA-256")
    sys.path.insert(0, str(EINOPS_WHEEL))  # zipimport, read-only
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")

    class DropPath(torch.nn.Module):  # never constructed on the compared path
        def __init__(self, *a, **k):
            raise AssertionError("DropPath constructed: drop_path must be 0 on the compared path")

    layers.DropPath = DropPath
    layers.to_2tuple = lambda v: v if isinstance(v, tuple) else (v, v)
    layers.trunc_normal_ = lambda t, std=1.0, **k: torch.nn.init.trunc_normal_(t, std=std, a=-2.0, b=2.0)
    timm.models, models.layers = models, layers
    sys.modules.update({"timm": timm, "timm.models": models, "timm.models.layers": layers})
    mod = types.ModuleType("upstream_uformer_65fc970")
    mod.__file__ = str(UPSTREAM)
    exec(compile(UPSTREAM.read_bytes(), str(UPSTREAM), "exec"), mod.__dict__)  # unmodified bytes
    import einops

    return mod, einops


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a - b).abs()
    rel = float((a - b).norm() / b.norm()) if float(b.norm()) > 0 else float(d.max())
    return {
        "max_abs": float(d.max()),
        "rel_l2": rel,
        "ok": float(d.max()) <= TOL["max_abs"] and rel <= TOL["max_rel_l2"],
    }


def map_weights(dst: torch.nn.Module, src: torch.nn.Module) -> list[str]:
    """Copy every upstream parameter/buffer into the vendored module by name; both must have exactly the same keys
    and shapes (the vendored code keeps the upstream attribute names)."""
    s, d = src.state_dict(), dst.state_dict()
    if set(s) != set(d):
        raise AssertionError(
            f"key sets differ: only upstream {sorted(set(s) - set(d))}, "
            f"only vendored {sorted(set(d) - set(s))}"
        )
    for k in s:
        if s[k].shape != d[k].shape:
            raise AssertionError(f"shape differs for {k}: {tuple(s[k].shape)} vs {tuple(d[k].shape)}")
    dst.load_state_dict(s, strict=True)
    return sorted(s)


def randomize(m: torch.nn.Module, g: torch.Generator) -> None:
    """Non-trivial values everywhere (zero biases or a zero relative-position table would hide errors)."""
    with torch.no_grad():
        for _, p in m.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.2)


def run_pair(name: str, make_up, make_vd, x_shape, g, mask=None) -> dict:
    up, vd = make_up().double(), make_vd().double()
    randomize(up, g)
    keys = map_weights(vd, up)
    x = torch.randn(x_shape, generator=g, dtype=torch.float64)
    xu, xv = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    yu = up(xu) if mask is None else up(xu, mask=mask)
    yv = vd(xv) if mask is None else vd(xv, mask=mask)
    w = torch.randn(yu.shape, generator=g, dtype=torch.float64)  # random cotangent
    (yu * w).sum().backward()
    (yv * w).sum().backward()
    res = {
        "case": name,
        "state_keys": len(keys),
        "forward": compare(yv.detach(), yu.detach()),
        "input_grad": compare(xv.grad, xu.grad),
    }
    pu, pv = dict(up.named_parameters()), dict(vd.named_parameters())
    worst = {"max_abs": 0.0, "rel_l2": 0.0, "ok": True, "worst_param": None}
    for k in pu:
        c = compare(pv[k].grad, pu[k].grad)
        if c["max_abs"] >= worst["max_abs"]:
            worst.update(max_abs=c["max_abs"], worst_param=k)
        worst["rel_l2"] = max(worst["rel_l2"], c["rel_l2"])
        worst["ok"] = worst["ok"] and c["ok"]
    res["param_grads"] = worst
    res["ok"] = res["forward"]["ok"] and res["input_grad"]["ok"] and worst["ok"]
    return res


def main() -> int:
    up, einops = load_upstream()
    sys.path[:0] = [str(FROZEN)]
    import uformer_blocks as V  # frozen vendored module, unchanged

    g = torch.Generator().manual_seed(20261013)  # engineering-only seed, not a registered one
    dim, heads, win, grid = 64, 2, 8, 32  # the configured values of the registered U-FIT (fixture data only)
    cases = []
    # window partition / reverse (pure tensor ops)
    xw = torch.randn((2, grid, grid, dim), generator=g, dtype=torch.float64)
    pu, pv = up.window_partition(xw, win), V.window_partition(xw, win)
    cases.append({"case": "window_partition", "forward": compare(pv, pu), "ok": compare(pv, pu)["ok"]})
    ru, rv = up.window_reverse(pu, win, grid, grid), V.window_reverse(pv, win, grid, grid)
    cases.append(
        {
            "case": "window_reverse",
            "forward": compare(rv, ru),
            "roundtrip_exact": bool(torch.equal(rv, xw)),
            "ok": compare(rv, ru)["ok"] and torch.equal(rv, xw),
        }
    )
    # shifted-window mask, rebuilt the upstream way (copied verbatim from upstream forward) vs the vendored helper
    shift = win // 2
    ref = torch.zeros((1, grid, grid, 1), dtype=torch.float64)
    cnt = 0
    for hs in (slice(0, -win), slice(-win, -shift), slice(-shift, None)):
        for ws in (slice(0, -win), slice(-win, -shift), slice(-shift, None)):
            ref[:, hs, ws, :] = cnt
            cnt += 1
    mw = up.window_partition(ref, win).view(-1, win * win)
    um = mw.unsqueeze(1) - mw.unsqueeze(2)
    um = um.masked_fill(um != 0, float(-100.0)).masked_fill(um == 0, float(0.0))  # noqa: UP018 (verbatim upstream)
    vm = V.shift_attention_mask(grid, grid, win, shift, ref)
    cases.append(
        {
            "case": "shift_attention_mask",
            "forward": compare(vm, um),
            "exact": bool(torch.equal(vm, um)),
            "ok": bool(torch.equal(vm, um)),
        }
    )
    # relative position index buffer
    wa_u = up.WindowAttention(dim, (win, win), heads)
    wa_v = V.WindowAttention(dim, (win, win), heads)
    same_idx = bool(torch.equal(wa_u.relative_position_index, wa_v.relative_position_index))
    cases.append({"case": "relative_position_index", "exact": same_idx, "ok": same_idx})
    # modules with mapped weights: forward, input gradient, every parameter gradient
    # QKV projection: q, k and v separately (the attention scale is applied in WindowAttention, compared below)
    lp_u = up.LinearProjection(dim, heads, dim // heads).double()
    lp_v = V.LinearProjection(dim, heads, dim // heads).double()
    randomize(lp_u, g)
    map_weights(lp_v, lp_u)
    xq = torch.randn((16, win * win, dim), generator=g, dtype=torch.float64)
    qkv = [compare(a, b) for a, b in zip(lp_v(xq), lp_u(xq))]
    cases.append(
        {
            "case": "LinearProjection q/k/v",
            "forward": {"q": qkv[0], "k": qkv[1], "v": qkv[2]},
            "ok": all(c["ok"] for c in qkv),
        }
    )
    scale_same = wa_u.scale == wa_v.scale == (dim // heads) ** -0.5
    cases.append(
        {
            "case": "attention scale (dim/heads)^-0.5",
            "upstream": wa_u.scale,
            "vendored": wa_v.scale,
            "ok": bool(scale_same),
        }
    )
    cases.append(
        run_pair(
            "WindowAttention (W-MSA)",
            lambda: up.WindowAttention(dim, (win, win), heads),
            lambda: V.WindowAttention(dim, (win, win), heads),
            (16, win * win, dim),
            g,
        )
    )
    cases.append(
        run_pair(
            "WindowAttention (SW-MSA mask)",
            lambda: up.WindowAttention(dim, (win, win), heads),
            lambda: V.WindowAttention(dim, (win, win), heads),
            (16, win * win, dim),
            g,
            mask=um.double(),
        )
    )
    cases.append(
        run_pair(
            "LeFF", lambda: up.LeFF(dim, 4 * dim), lambda: V.LeFF(dim, 4 * dim), (2, grid * grid, dim), g
        )
    )
    for shift_size in (0, shift):
        cases.append(
            run_pair(
                f"LeWinTransformerBlock shift={shift_size}",
                lambda s=shift_size: up.LeWinTransformerBlock(
                    dim, (grid, grid), heads, win_size=win, shift_size=s, mlp_ratio=4.0, token_mlp="leff"
                ),
                lambda s=shift_size: V.LeWinTransformerBlock(
                    dim, (grid, grid), heads, win_size=win, shift_size=s, mlp_ratio=4.0
                ),
                (2, grid * grid, dim),
                g,
            )
        )
    cases.append(
        run_pair(
            "BasicUformerLayer depth=2 shift_flag",
            lambda: up.BasicUformerLayer(
                dim, dim, (grid, grid), 2, heads, win, mlp_ratio=4.0, token_mlp="leff", shift_flag=True
            ),
            lambda: V.BasicUformerLayer(dim, (grid, grid), 2, heads, win, mlp_ratio=4.0, shift_flag=True),
            (2, grid * grid, dim),
            g,
        )
    )
    report = {
        "tolerances_declared_before_run": TOL,
        "upstream_sha256": UPSTREAM_SHA256,
        "einops": {
            "version": einops.__version__,
            "wheel_sha256": EINOPS_WHEEL_SHA256,
            "file": str(Path(einops.__file__)),
        },
        "vendored_file": str(Path(V.__file__).resolve()),
        "vendored_sha256": sha(Path(V.__file__)),
        "cases": cases,
        "all_ok": all(c["ok"] for c in cases),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
