"""ULR-20261015-v1 registered constants (``protocol.md``). Standard library only.

Every scientific choice of the study is a literal here; the manifest pins this file. Paths and hashes of the
external inputs (the UOA source checkpoint, the UCX / UOA codec state and tiles), the schedule hash and the pilot
outcomes are in ``ulr_config.json``.
"""

from __future__ import annotations

PROTOCOL_ID = "ULR-20261015-v1"
FIXTURE_PROTOCOL_ID = "ULR-V1-ENGINEERING-FIXTURE"
POLICY_ID = "UPP-20261014-v1"

# ---- data (the UCX materialization reused by UOA) --------------------------------------------------------------
TILE = 32
N_TILE = TILE * TILE  # 1,024 input elements per tile
INPUT_TILE_BYTES = 4 * N_TILE  # 4,096 B in fp32
COHORTS = ("TRAIN", "EARLY", "LATE")
HELDOUT = ("EARLY", "LATE")
ROWS = (8, 24, 40, 56)
LAYERS = (1, 5, 9, 13, 17, 21)
TILES_PER_SNAPSHOT = len(ROWS) * len(LAYERS)
BLOCKS = ("q_proj.A^T", "q_proj.B", "k_proj.A^T", "k_proj.B")
BLOCK_COLS = 8
TARGET_NORMALIZED_MSE = 1e-4  # SECONDARY outcome only (LOSS_CONTRACT of UOA, reading R2)

# ---- the common source (UOA U-LEAKY-LONG final 12,000-update state) ---------------------------------------------
MODEL = {
    "input_hw": [32, 32],
    "in_channels": 1,
    "stem_kernel": 3,
    "stem_channels": 1,
    "down_channels": [2, 4, 8, 16, 32, 16],
    "down_strides": [1, 1, 1, 2, 2, 2],
    "final_kernel": 7,
    "core_layers": 3,
    "core_depth": 2,
    "num_heads": 2,
    "win_size": 4,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "shift_flag": True,
}
MODEL_INIT_SEED = (
    1  # the builder seed of the source architecture (its tensors are then replaced by the source state)
)
SOURCE_ACTIVATION = "leaky_relu"
SOURCE_UPDATES = 12000  # inherited training cost of the source (UOA), disclosed with every adapted result
BACKBONE_INTERNAL_SHAPE = (16, 4, 4)
GRID_HW = (4, 4)
GRID_N = 16

# ---- the communication bridge and the arms ---------------------------------------------------------------------
ARMS = {"C16-CONTROL": 16, "C4-REDUCED": 4, "C1-REDUCED": 1}
ARM_ORDER = ("C16-CONTROL", "C4-REDUCED", "C1-REDUCED")
STEP_ZERO_FORWARD_ONLY_C = (8, 2)  # TRAIN-only step-zero mappings from the same basis; never trained
ALL_C = (16, 8, 4, 2, 1)
SIGN_RULE = "each principal direction (row) is flipped so that its largest-|entry| component is positive; ties go to the first index"
C16_IDENTITY_TOL = {"latent_max_abs": 1e-4, "output_max_abs": 1e-4, "pooled_nmse_rel": 1e-4}


def wire_shape(c: int) -> tuple[int, int, int]:
    return (c, *GRID_HW)


def wire_numel(c: int) -> int:
    return c * GRID_N


def wire_bytes(c: int) -> int:
    return 4 * wire_numel(c)


# ---- adaptation (identical for every arm) ----------------------------------------------------------------------
OPTIMIZER = {"name": "Adam", "lr": 0.0002, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0}
BATCH_SIZE = 8
MAX_UPDATES = 6000
CHECKPOINTS = (0, 1000, 3000, 6000)
SCHEDULE_SEED = 20261015
CURVE_EVERY = 100
CPU_THREADS = {"intra_op_threads": 24, "inter_op_threads": 24}
TORCH_VERSION = "2.14.0+cu130"
DIAG_SCHEDULE_SEED = 20261014  # the UCX / UOA registered batches used by the pre-training BN diagnostic
DIAG_BATCHES = 4

# ---- methods ---------------------------------------------------------------------------------------------------
SOURCE = "SOURCE-LEAKY"  # the inherited, unmodified UOA U-LEAKY-LONG final model


def arm_method(arm: str, point: str) -> str:
    return f"{arm}:{point}"  # point in ("STEP0", "FINAL")


ARM_FINAL = tuple(arm_method(a, "FINAL") for a in ARM_ORDER)
ARM_STEP0 = tuple(arm_method(a, "STEP0") for a in ARM_ORDER)
INPUT_PCA = {"PCA-256": 256, "PCA-64": 64, "PCA-16": 16}
REFERENCES = (f"{SOURCE}:FLOAT", f"{SOURCE}:INT8LAT", "SCALAR-4", "INT8-SHARED")
ELEMENTS = {
    **{arm_method(a, p): wire_numel(c) for a, c in ARMS.items() for p in ("STEP0", "FINAL")},
    **INPUT_PCA,
    f"{SOURCE}:FLOAT": 256,
    f"{SOURCE}:INT8LAT": 256,
    "SCALAR-4": 1024,
    "INT8-SHARED": 1024,
}
ELEMENT_KIND = {
    **{arm_method(a, p): "fp32 wire latent" for a in ARMS for p in ("STEP0", "FINAL")},
    **{m: "fp32 PCA coefficient" for m in INPUT_PCA},
    f"{SOURCE}:FLOAT": "fp32 latent",
    f"{SOURCE}:INT8LAT": "int8 latent code",
    "SCALAR-4": "2-bit index",
    "INT8-SHARED": "int8 code",
}
PAYLOAD_BYTES = {
    **{arm_method(a, p): wire_bytes(c) for a, c in ARMS.items() for p in ("STEP0", "FINAL")},
    **{m: 4 * k for m, k in INPUT_PCA.items()},
    f"{SOURCE}:FLOAT": 1024,
    f"{SOURCE}:INT8LAT": 256,
    "SCALAR-4": 256,
    "INT8-SHARED": 1024,
}
PRIMARY_UNITS = ARM_FINAL
CONTROLS = ("C-ZERO", "C-MEAN", "C-WRONG:<method>", "C-ZWIRE:<arm state>", "C-ZLAT:SOURCE-LEAKY")

# ---- invocation ------------------------------------------------------------------------------------------------
STEPS = ("INIT", "C16-CONTROL", "C4-REDUCED", "C1-REDUCED", "EVAL")
STEP_PHASE = {
    "INIT": "init",
    "C16-CONTROL": "train",
    "C4-REDUCED": "train",
    "C1-REDUCED": "train",
    "EVAL": "eval",
}
NEURAL_STEPS = ARM_ORDER
LIMITS = {
    "neural_arm_s": 1800.0,  # per arm
    "init_s": 600.0,  # preparation / initialization computations (INIT)
    "eval_child_s": 1500.0,
    "verification_reserve_s": 500.0,  # independent verification + replay after the invocation
    "evaluation_and_verification_s": 2000.0,  # EVAL + verification, cumulative
    "aggregate_s": 9000.0,  # every scientific workload, the pilots and the verification
    "pilot_s": 120.0,
    "rss_cap_bytes": 4 * 2**30,
    "rss_available_fraction": 0.5,
    "evidence_cap_bytes": 512 * 2**20,
    "max_invocations": 1,
    "max_neural_starts": 3,
    "max_channel_pca_fits": 1,
    "max_pilots": 3,
}
MECHANICS = {"end_margin_s": 20.0, "save_reserve_s": 60.0, "poll_interval_s": 0.25, "kill_wait_s": 10.0}
REQUIRED_LIMITS = (
    "neural_arm_s",
    "init_s",
    "eval_child_s",
    "aggregate_s",
    "rss_bytes",
    "evidence_bytes",
    "invocations",
    "neural_starts",
    "channel_pca_fits",
)
