# ULR-20261015-v1 report tables (generated from analysis/analysis.json)

## A. Element reduction (observed in EVAL)

| Arm state | Wire latent | Elements | Payload B | Element ratio | Receiver tensors / B | Verified |
|---|---|---:|---:|---:|---|---|
| C1-REDUCED:FINAL | [1, 4, 4] | 16 | 64 | 0.0156 | 148 / 153088 | yes |
| C1-REDUCED:STEP0 | [1, 4, 4] | 16 | 64 | 0.0156 | 148 / 153088 | yes |
| C16-CONTROL:FINAL | [16, 4, 4] | 256 | 1024 | 0.2500 | 148 / 154048 | yes |
| C16-CONTROL:STEP0 | [16, 4, 4] | 256 | 1024 | 0.2500 | 148 / 154048 | yes |
| C4-REDUCED:FINAL | [4, 4, 4] | 64 | 256 | 0.0625 | 148 / 153280 | yes |
| C4-REDUCED:STEP0 | [4, 4, 4] | 64 | 256 | 0.0625 | 148 / 153280 | yes |

Step-zero payload bytes by c (INIT, TRAIN): {'1': 64, '16': 1024, '2': 128, '4': 256, '8': 512}; all verified: yes

## B. UPP-20261014-v1 (vendored pinned classifier)

| Unit | Role | EARLY | LATE | Study | TRAIN | Tiles PASS E / L |
|---|---|---|---|---|---|---|
| C1-REDUCED:FINAL | primary | PASS | PASS | PASS | PASS | 168/168, 335/336 |
| C1-REDUCED:STEP0 | step zero (descriptive) | PASS | PASS | PASS | PASS | 142/168, 284/336 |
| C16-CONTROL:FINAL | primary | PASS | PASS | PASS | PASS | 168/168, 336/336 |
| C16-CONTROL:STEP0 | step zero (descriptive) | PASS | PASS | PASS | PASS | 168/168, 336/336 |
| C4-REDUCED:FINAL | primary | PASS | PASS | PASS | PASS | 168/168, 336/336 |
| C4-REDUCED:STEP0 | step zero (descriptive) | PASS | PASS | PASS | PASS | 168/168, 336/336 |
| INT8-SHARED | comparator (report only) | PASS | PASS | PASS | PASS | 168/168, 336/336 |
| PCA-16 | comparator (report only) | PASS | PASS | PASS | PASS | 168/168, 336/336 |
| PCA-256 | comparator (report only) | PASS | PASS | PASS | PASS | 168/168, 336/336 |
| PCA-64 | comparator (report only) | PASS | PASS | PASS | PASS | 168/168, 336/336 |
| SCALAR-4 | comparator (report only) | PASS | PASS | PASS | PASS | 168/168, 336/336 |
| SOURCE-LEAKY:FLOAT | reference | PASS | PASS | PASS | PASS | 168/168, 336/336 |
| SOURCE-LEAKY:INT8LAT | reference | PASS | PASS | PASS | PASS | 168/168, 336/336 |

Smallest tested adapted element count with study PASS: 16

## C1. Main results (pooled; normalized MSE direct for neural, raw / s32^2 for codecs)

| Method | Elements | Bytes | TRAIN RSE | TRAIN nMSE | EARLY RSE / cos | EARLY nMSE | LATE RSE / cos | LATE nMSE |
|---|---:|---:|---:|---:|---|---:|---|---:|
| SOURCE-LEAKY:FLOAT | 256 | 1024 | 0.6612 | 0.015294 | 0.6839 / 0.5630 | 0.014808 | 0.6842 / 0.5625 | 0.015488 |
| C16-CONTROL:STEP0 | 256 | 1024 | 0.6612 | 0.015294 | 0.6839 / 0.5630 | 0.014808 | 0.6842 / 0.5625 | 0.015488 |
| C16-CONTROL:FINAL | 256 | 1024 | 0.6326 | 0.014632 | 0.6587 / 0.5885 | 0.014264 | 0.6622 / 0.5851 | 0.014990 |
| C4-REDUCED:STEP0 | 64 | 256 | 0.8924 | 0.020642 | 0.8988 / 0.3203 | 0.019461 | 0.9014 / 0.3159 | 0.020403 |
| C4-REDUCED:FINAL | 64 | 256 | 0.8423 | 0.019484 | 0.8682 / 0.3646 | 0.018800 | 0.8742 / 0.3561 | 0.019788 |
| C1-REDUCED:STEP0 | 16 | 64 | 0.9791 | 0.022647 | 0.9828 / 0.1426 | 0.021282 | 0.9833 / 0.1407 | 0.022258 |
| C1-REDUCED:FINAL | 16 | 64 | 0.9390 | 0.021720 | 0.9511 / 0.2226 | 0.020596 | 0.9531 / 0.2183 | 0.021574 |
| PCA-256 | 256 | 1024 | 0.3042 | 0.007035 | 0.5966 / 0.6351 | 0.012918 | 0.6006 / 0.6320 | 0.013594 |
| PCA-64 | 64 | 256 | 0.6819 | 0.015772 | 0.8328 / 0.4090 | 0.018032 | 0.8398 / 0.4002 | 0.019010 |
| PCA-16 | 16 | 64 | 0.8789 | 0.020331 | 0.9307 / 0.2633 | 0.020152 | 0.9377 / 0.2497 | 0.021225 |
| SCALAR-4 | 1024 | 256 | 0.1224 | 0.002831 | 0.1201 / 0.9384 | 0.002600 | 0.1182 / 0.9392 | 0.002675 |
| SOURCE-LEAKY:INT8LAT | 256 | 256 | 0.6612 | 0.015295 | 0.6839 / 0.5630 | 0.014809 | 0.6843 / 0.5624 | 0.015489 |
| INT8-SHARED | 1024 | 1024 | 2.015e-04 | 4.662e-06 | 2.155e-04 / 0.9999 | 4.666e-06 | 2.058e-04 / 0.9999 | 4.659e-06 |

## C2. Damage (step 0 / source), recovery (final / step 0), final / final C16 (normalized MSE ratios)

| Arm | Cohort | step0 / source | final / step0 | fraction of damage recovered | final / source | final / C16 final | paired: final lower than step0 |
|---|---|---:|---:|---:|---:|---:|---|
| C16-CONTROL | TRAIN | 1.0000 | 0.9568 | -4.032e+07 | 0.9568 | 1.0000 | 1389/1440 |
| C16-CONTROL | EARLY | 1.0000 | 0.9633 | -2.264e+07 | 0.9633 | 1.0000 | 163/168 |
| C16-CONTROL | LATE | 1.0000 | 0.9678 | -1.650e+07 | 0.9678 | 1.0000 | 303/336 |
| C4-REDUCED | TRAIN | 1.3497 | 0.9439 | 0.2164 | 1.2740 | 1.3316 | 1430/1440 |
| C4-REDUCED | EARLY | 1.3143 | 0.9660 | 0.1421 | 1.2696 | 1.3180 | 158/168 |
| C4-REDUCED | LATE | 1.3173 | 0.9699 | 0.1251 | 1.2776 | 1.3201 | 317/336 |
| C1-REDUCED | TRAIN | 1.4808 | 0.9591 | 0.1261 | 1.4202 | 1.4843 | 1407/1440 |
| C1-REDUCED | EARLY | 1.4372 | 0.9678 | 0.1060 | 1.3908 | 1.4439 | 160/168 |
| C1-REDUCED | LATE | 1.4371 | 0.9693 | 0.1010 | 1.3929 | 1.4393 | 321/336 |

## C3. TRAIN learning curves (eval mode, wire path)

| Arm | 0 | 1,000 | 3,000 | 6,000 (normalized MSE / RSE / cos) |
|---|---|---|---|---|
| C16-CONTROL | 0.015294 / 0.6612 / 0.5823 | 0.015150 / 0.6549 / 0.5879 | 0.014879 / 0.6432 / 0.5974 | 0.014632 / 0.6326 / 0.6090 |
| C4-REDUCED | 0.020642 / 0.8924 / 0.3295 | 0.020005 / 0.8648 / 0.3679 | 0.019738 / 0.8533 / 0.3831 | 0.019484 / 0.8423 / 0.3973 |
| C1-REDUCED | 0.022647 / 0.9791 / 0.1532 | 0.022240 / 0.9615 / 0.1971 | 0.021886 / 0.9462 / 0.2321 | 0.021720 / 0.9390 / 0.2471 |

Step-zero TRAIN by width (PCA bridge, no adaptation; c = 8 and 2 forward-only): c=16 (256 el.): RSE 0.6612, cos 0.5823; c=8 (128 el.): RSE 0.7993, cos 0.4487; c=4 (64 el.): RSE 0.8924, cos 0.3295; c=2 (32 el.): RSE 0.9492, cos 0.2288; c=1 (16 el.): RSE 0.9791, cos 0.1532

## C4. Input dependence (held-out; tiles where the correct payload has lower SSE; pooled SSE ratio correct / wrong)

| Method | EARLY vs C-WRONG | LATE vs C-WRONG | EARLY vs zero wire / latent | LATE vs zero wire / latent |
|---|---|---|---|---|
| C16-CONTROL:FINAL | 168/168 (0.461) | 336/336 (0.456) | 168/168 (0.627) | 336/336 (0.633) |
| C4-REDUCED:FINAL | 168/168 (0.747) | 336/336 (0.751) | 168/168 (0.858) | 336/336 (0.863) |
| C1-REDUCED:FINAL | 168/168 (0.893) | 336/336 (0.895) | 168/168 (0.901) | 336/336 (0.905) |
| C16-CONTROL:STEP0 | 168/168 (0.508) | 336/336 (0.500) | 168/168 (0.677) | 336/336 (0.677) |
| C4-REDUCED:STEP0 | 168/168 (0.794) | 336/336 (0.798) | 168/168 (0.889) | 336/336 (0.892) |
| C1-REDUCED:STEP0 | 168/168 (0.942) | 336/336 (0.945) | 168/168 (0.973) | 336/336 (0.973) |
| SOURCE-LEAKY:FLOAT | 168/168 (0.508) | 336/336 (0.500) | 168/168 (0.309) | 336/336 (0.315) |
| PCA-256 | 168/168 (0.426) | 336/336 (0.420) | - | - |
| PCA-64 | 168/168 (0.716) | 336/336 (0.712) | - | - |
| PCA-16 | 168/168 (0.875) | 336/336 (0.874) | - | - |
| SCALAR-4 | 168/168 (0.063) | 336/336 (0.061) | - | - |

## D1. Matching element counts (final arm vs input PCA with the same number of fp32 elements)

| Arm | PCA | Cohort | arm RSE | PCA RSE | arm nMSE | PCA nMSE | tiles arm lower | pooled SSE arm / PCA |
|---|---|---|---:|---:|---:|---:|---|---:|
| C1-REDUCED | PCA-16 | TRAIN | 0.9390 | 0.8789 | 0.021720 | 0.020331 | 142/1440 | 1.0683 |
| C1-REDUCED | PCA-16 | EARLY | 0.9511 | 0.9307 | 0.020596 | 0.020152 | 36/168 | 1.0220 |
| C1-REDUCED | PCA-16 | LATE | 0.9531 | 0.9377 | 0.021574 | 0.021225 | 90/336 | 1.0165 |
| C16-CONTROL | PCA-256 | TRAIN | 0.6326 | 0.3042 | 0.014632 | 0.007035 | 0/1440 | 2.0798 |
| C16-CONTROL | PCA-256 | EARLY | 0.6587 | 0.5966 | 0.014264 | 0.012918 | 7/168 | 1.1042 |
| C16-CONTROL | PCA-256 | LATE | 0.6622 | 0.6006 | 0.014990 | 0.013594 | 13/336 | 1.1027 |
| C4-REDUCED | PCA-64 | TRAIN | 0.8423 | 0.6819 | 0.019484 | 0.015772 | 5/1440 | 1.2353 |
| C4-REDUCED | PCA-64 | EARLY | 0.8682 | 0.8328 | 0.018800 | 0.018032 | 22/168 | 1.0426 |
| C4-REDUCED | PCA-64 | LATE | 0.8742 | 0.8398 | 0.019788 | 0.019010 | 50/336 | 1.0409 |

## D2. Matching bytes (ranking by pooled SSE; element kinds differ)

- 1024B TRAIN: INT8-SHARED (RSE 0.0002; 1024 int8 code) < PCA-256 (RSE 0.3042; 256 fp32 PCA coefficient) < C16-CONTROL:FINAL (RSE 0.6326; 256 fp32 wire latent) < SOURCE-LEAKY:FLOAT (RSE 0.6612; 256 fp32 latent)
- 1024B EARLY: INT8-SHARED (RSE 0.0002; 1024 int8 code) < PCA-256 (RSE 0.5966; 256 fp32 PCA coefficient) < C16-CONTROL:FINAL (RSE 0.6587; 256 fp32 wire latent) < SOURCE-LEAKY:FLOAT (RSE 0.6839; 256 fp32 latent)
- 1024B LATE: INT8-SHARED (RSE 0.0002; 1024 int8 code) < PCA-256 (RSE 0.6006; 256 fp32 PCA coefficient) < C16-CONTROL:FINAL (RSE 0.6622; 256 fp32 wire latent) < SOURCE-LEAKY:FLOAT (RSE 0.6842; 256 fp32 latent)
- 256B TRAIN: SCALAR-4 (RSE 0.1224; 1024 2-bit index) < SOURCE-LEAKY:INT8LAT (RSE 0.6612; 256 int8 latent code) < PCA-64 (RSE 0.6819; 64 fp32 PCA coefficient) < C4-REDUCED:FINAL (RSE 0.8423; 64 fp32 wire latent)
- 256B EARLY: SCALAR-4 (RSE 0.1201; 1024 2-bit index) < SOURCE-LEAKY:INT8LAT (RSE 0.6839; 256 int8 latent code) < PCA-64 (RSE 0.8328; 64 fp32 PCA coefficient) < C4-REDUCED:FINAL (RSE 0.8682; 64 fp32 wire latent)
- 256B LATE: SCALAR-4 (RSE 0.1182; 1024 2-bit index) < SOURCE-LEAKY:INT8LAT (RSE 0.6843; 256 int8 latent code) < PCA-64 (RSE 0.8398; 64 fp32 PCA coefficient) < C4-REDUCED:FINAL (RSE 0.8742; 64 fp32 wire latent)
- 64B TRAIN: PCA-16 (RSE 0.8789; 16 fp32 PCA coefficient) < C1-REDUCED:FINAL (RSE 0.9390; 16 fp32 wire latent)
- 64B EARLY: PCA-16 (RSE 0.9307; 16 fp32 PCA coefficient) < C1-REDUCED:FINAL (RSE 0.9511; 16 fp32 wire latent)
- 64B LATE: PCA-16 (RSE 0.9377; 16 fp32 PCA coefficient) < C1-REDUCED:FINAL (RSE 0.9531; 16 fp32 wire latent)

## D3. Secondary target: normalized MSE vs 1e-4

| Method | TRAIN | EARLY | LATE | x target (TRAIN) |
|---|---:|---:|---:|---:|
| C16-CONTROL:FINAL | 0.014632 | 0.014264 | 0.014990 | 146.3 |
| C4-REDUCED:FINAL | 0.019484 | 0.018800 | 0.019788 | 194.8 |
| C1-REDUCED:FINAL | 0.021720 | 0.020596 | 0.021574 | 217.2 |
| SOURCE-LEAKY:FLOAT | 0.015294 | 0.014808 | 0.015488 | 152.9 |
| PCA-16 | 0.020331 | 0.020152 | 0.021225 | 203.3 |
| PCA-64 | 0.015772 | 0.018032 | 0.019010 | 157.7 |
| PCA-256 | 0.007035 | 0.012918 | 0.013594 | 70.4 |
| SCALAR-4 | 0.002831 | 0.002600 | 0.002675 | 28.3 |
| INT8-SHARED | 4.662e-06 | 4.666e-06 | 4.659e-06 | 0.0 |

## E. Bytes

| Arm | Payload B | Receiver one-time B | Sender one-time B | Bridge parameters (reduce / expand) | Amortized B at 504 / 3,072 / 64,512 |
|---|---:|---:|---:|---|---|
| C1-REDUCED | 64 | 153092 | 153132 | 17 / 32 | 368 / 114 / 66 |
| C16-CONTROL | 1024 | 154052 | 154152 | 272 / 272 | 1330 / 1074 / 1026 |
| C4-REDUCED | 256 | 153284 | 153336 | 68 / 80 | 560 / 306 / 258 |
