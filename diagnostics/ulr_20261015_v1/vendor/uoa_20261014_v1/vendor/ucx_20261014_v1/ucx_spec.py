"""UCX-20261014-v1 registered constants (``input/SELECTION_RULE.md`` and ``protocol.md``). Standard library only.

Every scientific choice of the study is a literal here; the manifest pins this file. Paths and hashes of external
inputs, the pinned frozen modules and the resource-pilot outcome are in ``ucx_config.json``.
"""

from __future__ import annotations

PROTOCOL_ID = "UCX-20261014-v1"
FIXTURE_PROTOCOL_ID = "UCX-V1-ENGINEERING-FIXTURE"
RULE_SHA256 = "a41d5b236c731144c015572629c4fa9012ad8415218643c10162fc7204c682e0"

# ---- split (section 3 of the rule) -----------------------------------------------------------------------------
TRAIN_TIMES = tuple(range(0, 12))
EARLY_TIMES = tuple(range(12, 16))
LATE_TIMES = tuple(range(16, 20))
COHORTS = ("EARLY", "LATE")
# ---- tiles (section 4) ------------------------------------------------------------------------------------------
TILE = 32
GRID = (64, 48)
ROWS = (8, 24, 40, 56)
LAYERS = (1, 5, 9, 13, 17, 21)
TILES_PER_SNAPSHOT = len(ROWS) * len(LAYERS)
BLOCKS = (("q_proj", "A"), ("q_proj", "B"), ("k_proj", "A"), ("k_proj", "B"))
LORA_SCALE, LORA_RANK = 2.0, 8
LAYOUT_ID = "layer_major_qkvo_AtB"
PHI_SHAPE = (1, 2048, 1536)
GEOMETRY = {"hidden": 2048, "modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "num_layers": 24, "rank": 8}
# ---- normalization (section 5) ----------------------------------------------------------------------------------
MAXABS_TARGET = 0.95
NORMALIZATION_MODE = "global_exact_maxabs_train (UCX TRAIN tiles only)"
# ---- U-1/4 (section 6) ------------------------------------------------------------------------------------------
ARM = "U-QUARTER"
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
MODEL_INIT_SEED = 1
OPTIMIZER = {"name": "Adam", "lr": 0.0002, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0}
BATCH_SIZE = 8
MAX_UPDATES = 3000
SCHEDULE_SEED = 20261014
CURVE_EVERY = 100
CPU_THREADS = {"intra_op_threads": 24, "inter_op_threads": 24}
TORCH_VERSION = "2.14.0+cu130"
# ---- comparators (section 6) ------------------------------------------------------------------------------------
PCA_COMPONENTS = 256
PCA_SOLVER = "numpy.linalg.svd(Xc, full_matrices=False) (LAPACK gesdd), OPENBLAS_NUM_THREADS=1"
INT8_MAX_CODE = 127
METHODS = ("U-QUARTER", "PCA-256", "INT8-SHARED", "INT8-TILE")
PAYLOAD_BYTES = {"U-QUARTER": 1024, "PCA-256": 1024, "INT8-SHARED": 1024, "INT8-TILE": 1028}
INPUT_TILE_BYTES = 4096
UPDATES_PLANNED = {"U-QUARTER": MAX_UPDATES, "PCA-256": 0, "INT8-SHARED": 0, "INT8-TILE": 0}
# ---- controls (section 7) ---------------------------------------------------------------------------------------
CONTROLS = ("C-ZERO", "C-MEAN", "C-INIT", "C-ZLAT-U", "C-ZLAT-PCA", "C-WRONG-U", "C-WRONG-PCA")
# ---- invocation steps, in order (section 9) ---------------------------------------------------------------------
STEPS = ("BUILD-TRAIN", "PCA-FIT", "U-QUARTER", "BUILD-HELDOUT", "EVAL")
STEP_PHASE = {
    "BUILD-TRAIN": "build_train",
    "PCA-FIT": "pca_fit",
    "U-QUARTER": "train",
    "BUILD-HELDOUT": "build_heldout",
    "EVAL": "eval",
}
# ---- limits (seconds unless stated) -----------------------------------------------------------------------------
LIMITS = {
    "data_construction_s": 1800.0,  # BUILD-TRAIN + BUILD-HELDOUT, cumulative
    "pca_fit_s": 300.0,
    "neural_arm_s": 5400.0,
    "eval_child_s": 900.0,  # part of the 1,200 s for evaluation, controls and verification computation
    "verification_reserve_s": 300.0,  # independent verification + forward-only replay (the rest of the 1,200 s)
    "evaluation_and_verification_s": 1200.0,
    "aggregate_s": 9000.0,  # every scientific workload plus the synthetic resource pilot
    "rss_cap_bytes": 4 * 2**30,
    "rss_available_fraction": 0.5,
    "evidence_cap_bytes": 512 * 2**20,
    "max_invocations": 1,
    "max_neural_starts": 1,
    "max_pca_fits": 1,
}
MECHANICS = {"end_margin_s": 20.0, "save_reserve_s": 60.0, "poll_interval_s": 0.25, "kill_wait_s": 10.0}
# the limits every primary unit's run record must carry (UPP-20261014-v1 ``required_limits``)
REQUIRED_LIMITS = (
    "data_construction_s",
    "pca_fit_s",
    "neural_arm_s",
    "eval_child_s",
    "aggregate_s",
    "rss_bytes",
    "evidence_bytes",
    "invocations",
    "neural_starts",
    "pca_fits",
)
POLICY_ID = "UPP-20261014-v1"

# ---- the registered record lists of the rule (section 3), recomputed from the index and compared ---------------
TRAIN_KEYS = (
    (0, 2), (0, 26), (0, 55), (0, 75), (0, 86), (1, 33), (1, 80), (1, 81), (1, 84), (1, 93),
    (2, 16), (2, 24), (2, 30), (2, 56), (2, 83), (3, 6), (3, 64), (3, 67), (3, 93), (3, 96),
    (4, 10), (4, 16), (4, 20), (4, 63), (4, 96), (5, 28), (5, 32), (5, 46), (5, 66), (5, 74),
    (6, 0), (6, 17), (6, 34), (6, 45), (6, 60), (7, 26), (7, 37), (7, 49), (7, 78), (7, 91),
    (8, 1), (8, 17), (8, 23), (8, 44), (8, 55), (9, 3), (9, 42), (9, 46), (9, 68), (9, 75),
    (10, 14), (10, 19), (10, 37), (10, 43), (10, 66), (11, 22), (11, 41), (11, 46), (11, 49), (11, 58),
)  # fmt: skip
EARLY_KEYS = ((12, 92), (13, 62), (14, 35), (14, 39), (15, 36), (15, 51), (15, 57))
LATE_KEYS = (
    (16, 7), (16, 25), (16, 47), (16, 71), (17, 4), (17, 9), (17, 71),
    (17, 73), (18, 47), (18, 53), (18, 54), (19, 18), (19, 72), (19, 85),
)  # fmt: skip
EXCLUDED_KEYS = (
    (12, 14), (12, 17), (12, 41), (12, 68), (13, 14), (13, 37), (13, 43), (13, 83), (14, 24), (14, 44),
    (14, 55), (15, 46), (15, 84), (16, 42), (17, 28), (18, 33), (18, 43), (19, 2), (19, 16),
)  # fmt: skip
INDEX_SHA256 = "b6834e476f21f3d6b07f6489b5113576e566ab3987a2f4593405664476d07240"


def wrong_map(keys: tuple) -> dict:
    """Section 7: each snapshot of a cohort takes the payload of the next one (sorted by (time, client)); the last
    takes the first's."""
    return {keys[i]: keys[(i + 1) % len(keys)] for i in range(len(keys))}
