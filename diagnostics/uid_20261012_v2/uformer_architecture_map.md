# UID-20261012-v2 Uformer Architecture Map

**Target**: CG-FedLLM Appendix B.4 Uformer autoencoder adaptation  
**Diagnostic variant**: all-stride-1, weak spatial compression bottleneck  
**Implementation**: `uformer_ae.py`, `uformer_blocks.py`

## Summary

- **Input**: [1, 1, 32, 32] single-channel float32
- **Latent**: [1, 64, 32, 32], element ratio 64:1 (expansion, not compression)
- **Encoder**: ResNet sampling (6 Conv3x3 stride-1 stages, channels 1→64) + 3 BasicUformerLayer
- **Decoder**: 3 BasicUformerLayer + ResNet sampling (6 ConvTranspose3x3 stride-1 stages, channels 64→1)
- **Core**: 6 LeWinTransformerBlock per side (3 layers × 2 blocks each)
- **Attention**: W-MSA/SW-MSA, 2 heads, window 8, relative position bias
- **FFN**: LeFF (Linear-GELU, depthwise 3×3, Linear-GELU)
- **No global shortcut**: `forward` returns `(decode(encode(x)), z)`, not `x + decode(...)`
- **No encoder-decoder skip**: decoder receives only `z`, no skip features
- **Parameters**: 685,507 total (342,797 encoder + 342,710 decoder)

## Detailed Architecture

### Encoder

```
Input [1, 1, 32, 32]
  ↓ ReflectionPad2d(1) + Conv3x3(1→1, no bias) + BN + ReLU
  ↓ Conv3x3 stride 1 pad 1 (1→2) + BN + ReLU
  ↓ Conv3x3 stride 1 pad 1 (2→4) + BN + ReLU
  ↓ Conv3x3 stride 1 pad 1 (4→8) + BN + ReLU
  ↓ Conv3x3 stride 1 pad 1 (8→16) + BN + ReLU
  ↓ Conv3x3 stride 1 pad 1 (16→32) + BN + ReLU
  ↓ Conv3x3 stride 1 pad 1 (32→64) + BN + ReLU
[1, 64, 32, 32] → flatten to tokens [1, 1024, 64]
  ↓ BasicUformerLayer(dim=64, depth=2, heads=2, win=8, shift_flag=True) [layer 0]
  ↓ BasicUformerLayer(dim=64, depth=2, heads=2, win=8, shift_flag=True) [layer 1]
  ↓ BasicUformerLayer(dim=64, depth=2, heads=2, win=8, shift_flag=True) [layer 2]
[1, 1024, 64] → reshape to [1, 64, 32, 32]
Latent z
```

### Decoder

```
Latent z [1, 64, 32, 32] → flatten to tokens [1, 1024, 64]
  ↓ BasicUformerLayer(dim=64, depth=2, heads=2, win=8, shift_flag=True) [layer 0]
  ↓ BasicUformerLayer(dim=64, depth=2, heads=2, win=8, shift_flag=True) [layer 1]
  ↓ BasicUformerLayer(dim=64, depth=2, heads=2, win=8, shift_flag=True) [layer 2]
[1, 1024, 64] → reshape to [1, 64, 32, 32]
  ↓ ConvTranspose3x3 stride 1 pad 1 output_pad 0 (64→32) + BN + ReLU
  ↓ ConvTranspose3x3 stride 1 pad 1 output_pad 0 (32→16) + BN + ReLU
  ↓ ConvTranspose3x3 stride 1 pad 1 output_pad 0 (16→8) + BN + ReLU
  ↓ ConvTranspose3x3 stride 1 pad 1 output_pad 0 (8→4) + BN + ReLU
  ↓ ConvTranspose3x3 stride 1 pad 1 output_pad 0 (4→2) + BN + ReLU
  ↓ ConvTranspose3x3 stride 1 pad 1 output_pad 0 (2→1) + BN + ReLU
  ↓ ReflectionPad2d(3) + Conv7x7(1→1, bias) + Tanh
Output [1, 1, 32, 32]
```

### BasicUformerLayer (depth=2, shift_flag=True)

Each layer contains 2 LeWinTransformerBlock:
- Block 0: shift_size = 0 (W-MSA)
- Block 1: shift_size = 4 (SW-MSA, half window)

### LeWinTransformerBlock

```
x [B, L, C] where L=1024, C=64, grid=32×32
  ↓ LayerNorm(64)
  ↓ reshape to [B, 32, 32, 64]
  ↓ cyclic shift (if shift_size > 0)
  ↓ window_partition → [B*16, 8, 8, 64] (16 windows of 8×8)
  ↓ flatten windows → [B*16, 64, 64]
  ↓ WindowAttention(dim=64, win=(8,8), heads=2)
      ├─ LinearProjection: Q, K, V from [B*16, 64, 64]
      ├─ scaled dot-product attention (head_dim=32, scale=1/√32)
      ├─ relative position bias [15×15, 2 heads] → [2, 64, 64]
      ├─ shift attention mask [16, 64, 64] if SW-MSA
      └─ proj Linear(64→64)
  ↓ window_reverse → [B, 32, 32, 64]
  ↓ reverse cyclic shift (if shift_size > 0)
  ↓ reshape to [B, 1024, 64]
  ↓ residual: x = shortcut + x
  ↓ LayerNorm(64)
  ↓ LeFF(dim=64, hidden=256)
      ├─ Linear(64→256) + GELU
      ├─ reshape to [B, 256, 32, 32]
      ├─ depthwise Conv2d(256→256, groups=256, k=3, s=1, p=1) + GELU
      ├─ reshape to [B, 1024, 256]
      └─ Linear(256→64)
  ↓ residual: x = x + mlp(x)
Output [B, 1024, 64]
```

### WindowAttention Details

- **Heads**: 2, head_dim = 64/2 = 32
- **Relative position bias table**: learnable [15×15, 2] (2*8-1 = 15 per axis)
- **QKV projection**: Linear 64→64 (Q), Linear 64→128 (K,V concatenated)
- **Attention**: softmax((Q @ K^T / √32) + relative_bias + shift_mask)
- **Output**: proj(attention @ V), Linear 64→64

### LeFF (Locally-enhanced Feed-Forward)

- **Hidden dimension**: 4 × 64 = 256 (mlp_ratio=4.0)
- **Structure**: Linear-GELU, spatial depthwise conv, Linear-GELU
- **Depthwise conv**: 3×3, groups=hidden_dim, preserves spatial resolution

## Paper-Specified vs. Implementation Choices

### Paper (App. B.4, Table 7)

- Retain ResNet AE sampling layers ✓
- Replace core blocks with BasicUformer (2 LeWinTransformerBlock each) ✓
- Encoder: DownSampling + 3 BasicUformer ✓
- Decoder: 3 BasicUformer + UpSampling ✓
- Table 7: 6 conv2d down, 3 Uformer Block, 3 Uformer Block, 6 deconv2d up ✓

### NOT Specified (Implementation Choices)

- **Heads**: 2 (not stated; 8 is Uformer author's default for larger models)
- **Window size**: 8 (author default, not stated in paper)
- **Shift**: shift_flag=True, shift_size=win_size//2 (SW-MSA, author default)
- **MLP ratio**: 4.0 (author default, not stated)
- **QKV bias**: True (author default, not stated)
- **FFN type**: LeFF (author default; paper does not specify FFN structure)
- **Token layout**: row-major [B, h*w, C] (upstream choice)
- **Position**: relative position bias only (no absolute position embedding)
- **Initialization**: trunc_normal_(std=0.02) for Linear/LayerNorm (upstream)
- **Input size**: 32×32 (derived from win_size=8 divisibility)
- **All-stride-1**: diagnostic intervention for weak bottleneck

### Diagnostic Differences from Paper's Intended Use

1. **All-stride-1 sampling**: latent grid = input grid, 64:1 element expansion  
   - Paper presumably uses stride-2 stages (Table 7 shows 2,2 strides)
   - This diagnostic tests the BasicUformer blocks without strong spatial compression
2. **32×32 input**: small, square, divisible by window (architectural requirement)
3. **No stochastic regularization**: dropout=0, attn_drop=0, drop_path=0 (deterministic)
4. **Single channel**: paper's U-Former uses 8-channel input ([8,1024,1024])

## Upstream Equivalence

The vendored blocks (`uformer_blocks.py`) reproduce the computational path of:
- `ZhendongWang6/Uformer@65fc970a`, `model.py` (SHA-256 `03a2c6e5...04a9`)
- MIT License, Copyright (c) 2022 Zhendong Wang

Adaptations (none change the configured path's computation):
- `einops.rearrange/repeat` → equivalent `view/permute/reshape`
- `timm.to_2tuple/trunc_normal_` → tuple / `torch.nn.init.trunc_normal_`
- Only configured options kept: linear projection, LeFF without ECA, no modulator
- Dropout/attention_drop/drop_path omitted (0 in this diagnostic)
- Explicit guards: square grid, divisible by window

`checks/test_upstream_equivalence.py` (planned) would compare outputs and gradients.

## What a PASS Establishes

- The tested Uformer variant can fit the tested input under the tested budget
- **Does NOT establish**:
  - Generalization to unseen inputs
  - Useful compression (64:1 expansion is not compression)
  - Success with stride-2 sampling or larger inputs
  - Successful federated training with this architecture

## What a FAIL Does NOT Establish

- Mathematical inability to represent the identity (one run, one seed, bounded budget)
- Failure of the paper's intended configuration (this is all-stride-1, not the paper setup)
- Absence of mode-related artifacts (requires comparing train/eval with known modules)

## BatchNorm and Train/Eval

- **Expected**: train/eval difference due to running statistics vs. frozen statistics
- **Not a bug**: small sse_delta between modes with BatchNorm present
- **Would be a bug**: large difference with no stochastic regularization and no BatchNorm

This diagnostic includes BatchNorm in the ResNet sampling layers, so train/eval differences are expected and documented.

---

**Frozen**: 2026-10-12T09:45:00+08:00 alongside `protocol.md` and `config.json`
