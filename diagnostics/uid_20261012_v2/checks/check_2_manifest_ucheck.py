"""CPU engineering check 2: verify manifest file hashes and U-CHECK tiny-model sanity."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import uid_common as C


def main() -> None:
    print("=== UID-20261012-v2 engineering check 2: manifest verification and U-CHECK ===\n")
    root = Path(__file__).parent.parent
    manifest = C.verify_manifest(root / "manifest.json", root=root)
    print(f"Manifest verification: {len(manifest['files'])} files match")
    for entry in manifest["files"]:
        if entry.get("verify"):
            print(f"  ✓ {entry['path']} ({entry['bytes']} bytes)")
    print()
    cfg_dict, raw_sha, canonical = C.load_config(root / "config.json")
    print(f"config.json raw SHA-256:       {raw_sha}")
    print(f"config.json canonical SHA-256: {canonical}\n")
    model = C.build_model(cfg_dict, "U-CHECK")
    print(f"U-CHECK model: {model}")
    print(f"  parameters: {sum(p.numel() for p in model.parameters())}")
    x = C.make_input(cfg_dict)
    print(f"  input: {x.shape}, SHA-256: {C.tensor_sha256(x, 'x')[:16]}...")
    model.train()
    out = model(x)
    print(f"  output: {out.shape}, range [{out.min():.4f}, {out.max():.4f}]")
    loss = torch.nn.functional.mse_loss(out, x)
    print(f"  MSE loss: {loss.item():.6f}")
    loss.backward()
    w_grad = model.weight.grad
    print(f"  weight gradient: norm {w_grad.norm():.6e}, finite {torch.isfinite(w_grad).all()}")
    print("\nCHECK 2 PASS: manifest verified, U-CHECK sanity complete")


if __name__ == "__main__":
    main()
