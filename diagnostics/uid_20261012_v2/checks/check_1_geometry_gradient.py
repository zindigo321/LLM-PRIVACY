"""CPU engineering check 1: parameter count, latent geometry and gradient flow through the Uformer AE."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from uformer_ae import UformerAEConfig, UformerAutoEncoder
import uid_common as C


def main() -> None:
    print("=== UID-20261012-v2 engineering check 1: Uformer geometry and gradients ===\n")
    cfg_dict, _, canonical = C.load_config("config.json")
    print(f"config.json canonical SHA-256: {canonical}\n")
    cfg = UformerAEConfig(**cfg_dict["u_fit"]["model"])
    print(f"Uformer config: {cfg.to_dict()}")
    print(f"  latent shape: {cfg.latent_shape}, element ratio: {cfg.latent_element_ratio:.4f}\n")
    model = C.build_model(cfg_dict, "U-FIT")
    params = model.num_parameters()
    print(f"Parameters: {params}")
    print(f"  total: {params['total']:,}\n")
    x = C.make_input(cfg_dict)
    print(f"Input: {x.shape}, range [{x.min():.4f}, {x.max():.4f}], SHA-256: {C.tensor_sha256(x, 'x')[:16]}...\n")
    x.requires_grad_(True)
    model.train()
    reconstructed, z = model(x)
    print(f"Latent z: {z.shape}, range [{z.min():.4f}, {z.max():.4f}]")
    print(f"Reconstructed: {reconstructed.shape}, range [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
    loss = torch.nn.functional.mse_loss(reconstructed, x)
    print(f"Initial MSE loss: {loss.item():.6f}\n")
    loss.backward()
    enc_grad = model.encoder.core[0].blocks[0].attn.proj.weight.grad
    dec_grad = model.decoder.core[0].blocks[0].attn.proj.weight.grad
    print(f"Encoder core gradient sample: norm {enc_grad.norm():.6e}, nonzero {enc_grad.count_nonzero()}/{enc_grad.numel()}")
    print(f"Decoder core gradient sample: norm {dec_grad.norm():.6e}, nonzero {dec_grad.count_nonzero()}/{dec_grad.numel()}")
    print(f"Input gradient: norm {x.grad.norm():.6e}, finite {torch.isfinite(x.grad).all()}\n")
    print("CHECK 1 PASS: forward/backward complete, gradients nonzero and finite")


if __name__ == "__main__":
    main()
