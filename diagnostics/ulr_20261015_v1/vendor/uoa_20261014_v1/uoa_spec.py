"""UOA-20261014-v1 registered constants (``protocol.md``, ``LOSS_CONTRACT.md``). Standard library only.

Every scientific choice of the study is a literal here; the manifest pins this file. Paths and hashes of external
inputs (the UCX artifacts), the schedule hashes and the resource-pilot outcome are in ``uoa_config.json``.
"""

from __future__ import annotations

PROTOCOL_ID = "UOA-20261014-v1"
FIXTURE_PROTOCOL_ID = "UOA-V1-ENGINEERING-FIXTURE"
POLICY_ID = "UPP-20261014-v1"

# ---- data (reused UCX materialization, rule sections 3-5 of UCX) ----------------------------------------------
TILE = 32
N_TILE = TILE * TILE
INPUT_TILE_BYTES = 4 * N_TILE
COHORTS = ("TRAIN", "EARLY", "LATE")
HELDOUT = ("EARLY", "LATE")
ROWS = (8, 24, 40, 56)
LAYERS = (1, 5, 9, 13, 17, 21)
TILES_PER_SNAPSHOT = len(ROWS) * len(LAYERS)
BLOCKS = ("q_proj.A^T", "q_proj.B", "k_proj.A^T", "k_proj.B")  # 8-column groups of a tile, in column order
BLOCK_COLS = 8

# ---- loss target (LOSS_CONTRACT.md) ------------------------------------------------------------------------------
TARGET_NORMALIZED_MSE = (
    1e-4  # inclusive; float64; final full-TRAIN eval (declared), also reported on EARLY / LATE
)

# ---- neural arms (protocol section 5) ----------------------------------------------------------------------------
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
LATENT_SHAPE = (16, 4, 4)
LATENT_N = 16 * 4 * 4
MODEL_INIT_SEED = 1
OPTIMIZER = {"name": "Adam", "lr": 0.0002, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0}
BATCH_SIZE = 8
MAX_UPDATES = 12000
CHECKPOINTS = (3000, 6000, 9000, 12000)
ANCHOR_UPDATE = 3000  # Arm A at this update is the UCX reproduction anchor
SCHEDULE_SEED = 20261014
CURVE_EVERY = 100
CPU_THREADS = {"intra_op_threads": 24, "inter_op_threads": 24}
TORCH_VERSION = "2.14.0+cu130"
LEAKY_SLOPE = 0.1
# the 13 sampling-path ReLU modules of the registered model (preparation/model_audit.json); nothing else is replaced
RELU_PATHS = (
    "encoder.sampling.3",
    "encoder.sampling.6",
    "encoder.sampling.9",
    "encoder.sampling.12",
    "encoder.sampling.15",
    "encoder.sampling.18",
    "encoder.sampling.21",
    "decoder.sampling.2",
    "decoder.sampling.5",
    "decoder.sampling.8",
    "decoder.sampling.11",
    "decoder.sampling.14",
    "decoder.sampling.17",
)
ARMS = {
    "U-RELU-LONG": {"activation": "relu"},
    "U-LEAKY-LONG": {"activation": "leaky_relu"},
}
ARM_ORDER = ("U-RELU-LONG", "U-LEAKY-LONG")
UCX_HIST = "UCX-HIST"  # the completed UCX U-QUARTER final state (3,000 updates, ReLU), forward only
NEURAL_MODELS = (UCX_HIST, *ARM_ORDER)
NEURAL_ACTIVATION = {UCX_HIST: "relu", "U-RELU-LONG": "relu", "U-LEAKY-LONG": "leaky_relu"}

# ---- Stage A: 256-B baselines (protocol section 6) ---------------------------------------------------------------
PCA_FIT_COMPONENTS = 128  # one fresh float64 decomposition; PCA-64 uses its first 64 rows
PCA_SOLVER = "numpy.linalg.svd(Xc, full_matrices=False) (LAPACK gesdd), OPENBLAS_NUM_THREADS=1"
PCA_RANK = {"PCA-64": 64, "PCA-128-HALF": 128}
UNIFORM_GRID_TENTHS = tuple(range(5, 61))  # a = (k / 10) * TRAIN RMS for k = 5..60, plus a = TRAIN max |x|
LLOYD_ITERATIONS = 50
LLOYD_INIT_QUANTILES = (0.125, 0.375, 0.625, 0.875)
QUANTILE_METHOD = "linear"
LEVELS = 4

# ---- Stage C: the latent int8 codec ------------------------------------------------------------------------------
INT8_MAX_CODE = 127

# ---- methods and payload bytes per tile --------------------------------------------------------------------------
BASELINES_256 = ("PCA-64", "PCA-128-HALF", "UNIFORM-2BIT", "SCALAR-4")
REFERENCES_1024 = ("PCA-256", "INT8-SHARED")  # UCX's persisted decoder / step, forward replay


def float_method(model: str) -> str:
    return f"{model}:FLOAT"


def int8_method(model: str) -> str:
    return f"{model}:INT8LAT"


METHODS_256 = (*BASELINES_256, *(int8_method(m) for m in NEURAL_MODELS))
METHODS_1024 = (*REFERENCES_1024, *(float_method(m) for m in NEURAL_MODELS))
PAYLOAD_BYTES = {
    **{m: 256 for m in METHODS_256},
    **{m: 1024 for m in METHODS_1024},
}
PRIMARY_NEURAL_UNITS = tuple(f(m) for m in ARM_ORDER for f in (float_method, int8_method))

# ---- controls (protocol section 9) -------------------------------------------------------------------------------
CONTROLS = ("C-ZERO", "C-MEAN", "C-WRONG:<method>", "C-ZLAT:<model>")

# ---- invocation steps, in order ----------------------------------------------------------------------------------
STEPS = ("DIAG", "STAGE-A", "U-RELU-LONG", "U-LEAKY-LONG", "CALIB", "EVAL")
STEP_PHASE = {
    "DIAG": "diag",
    "STAGE-A": "stage_a",
    "U-RELU-LONG": "train",
    "U-LEAKY-LONG": "train",
    "CALIB": "calib",
    "EVAL": "eval",
}
NEURAL_STEPS = ("U-RELU-LONG", "U-LEAKY-LONG")

# ---- limits (seconds unless stated) ------------------------------------------------------------------------------
LIMITS = {
    "neural_arm_s": 3000.0,  # per arm
    "baseline_fit_s": 600.0,  # STAGE-A
    "diag_s": 300.0,
    "calib_s": 300.0,
    "eval_child_s": 900.0,
    "verification_reserve_s": 300.0,  # independent verification + replay after the invocation
    "evaluation_and_verification_s": 1800.0,  # DIAG + CALIB + EVAL + verification, cumulative
    "aggregate_s": 9000.0,  # every scientific workload, the two pilots and the verification
    "pilot_s": 120.0,
    "rss_cap_bytes": 4 * 2**30,
    "rss_available_fraction": 0.5,
    "evidence_cap_bytes": 512 * 2**20,
    "max_invocations": 1,
    "max_neural_starts": 2,
    "max_pca_fits": 1,
    "max_pilots": 2,
}
MECHANICS = {"end_margin_s": 20.0, "save_reserve_s": 60.0, "poll_interval_s": 0.25, "kill_wait_s": 10.0}
REQUIRED_LIMITS = (
    "neural_arm_s",
    "baseline_fit_s",
    "eval_child_s",
    "aggregate_s",
    "rss_bytes",
    "evidence_bytes",
    "invocations",
    "neural_starts",
    "pca_fits",
)
