

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# Set before PyTorch is imported elsewhere.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

# ---------------------------------------------------------------------------
# 1. Project and paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

# Defaults copied from the original OurModel_1.py runner.
DEFAULT_DATA_ROOT = Path(r"D:YourPath\data")
DEFAULT_RESULTS_ROOT = Path(r"D:YourPath\Results")

DATA_ROOT = Path(os.environ.get("DVAGRAPH_GP_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()
RESULTS_ROOT = Path(os.environ.get("DVAGRAPH_GP_RESULTS_ROOT", str(DEFAULT_RESULTS_ROOT))).expanduser()

DATASET_NAME = "Wheat2000"
METHOD_NAME = "DVAGraph_GP_mathematical_architecture"
RUN_NAME = "Wheat2000_DVAGraph_GP_locked_3x10_no_preprocessing_v1"

WHEAT_DATA_DIR = DATA_ROOT / "wheat2000"
GENOTYPE_PATH = WHEAT_DATA_DIR / "genotype_markers.csv"
PHENOTYPE_DIR = DATA_ROOT

# The repository protocol/ folder should contain the existing locked files.
# Override this path if the protocol files remain in the original external folder.
PROTOCOL_DIR = Path(
    os.environ.get("DVAGRAPH_GP_PROTOCOL_DIR", str(PROJECT_DIR / "protocol"))
).expanduser()

OUTPUT_ROOT = RESULTS_ROOT / "DVAGraph-GP" / "wheat" / RUN_NAME

OUTER_SPLIT_PATH = PROTOCOL_DIR / "wheat2000_repeated_outer_cv_splits.csv"
OUTER_AUDIT_PATH = PROTOCOL_DIR / "wheat2000_repeated_outer_cv_split_audit.csv"
INNER_SPLIT_PATH = PROTOCOL_DIR / "wheat2000_repeated_inner_validation_splits.csv"
INNER_AUDIT_PATH = PROTOCOL_DIR / "wheat2000_repeated_inner_validation_split_audit.csv"
FINGERPRINT_PATH = PROTOCOL_DIR / "wheat2000_data_fingerprints.json"
COST_POLICY_PATH = PROTOCOL_DIR / "benchmark_cost_policy.json"
TRAINING_POLICY_PATH = PROTOCOL_DIR / "model_training_policy.json"
PROTOCOL_MANIFEST_PATH = PROTOCOL_DIR / "protocol_manifest.json"

REQUIRED_PROTOCOL_FILES = (
    OUTER_SPLIT_PATH,
    OUTER_AUDIT_PATH,
    INNER_SPLIT_PATH,
    INNER_AUDIT_PATH,
    FINGERPRINT_PATH,
    COST_POLICY_PATH,
    TRAINING_POLICY_PATH,
    PROTOCOL_MANIFEST_PATH,
)

# ---------------------------------------------------------------------------
# 2. Dataset and traits
# ---------------------------------------------------------------------------

GENOTYPE_SAMPLE_ID_COLUMN = "sample"
PHENOTYPE_SAMPLE_ID_COLUMN: Optional[str] = None

PHENOTYPE_SAMPLE_ID_CANDIDATES = (
    "sample", "Sample", "SAMPLE", "sample_id", "Sample_ID", "SAMPLE_ID",
    "id", "ID", "IID", "accession", "Accession",
)

TRAIT_CONFIGS: Dict[str, Dict[str, str]] = {
    "TKW": {"file": "2000_1_phe.csv", "column": "tkw"},
    "TW":  {"file": "2000_2_phe.csv", "column": "testw"},
    "GL":  {"file": "2000_3_phe.csv", "column": "length"},
    "GW":  {"file": "2000_4_phe.csv", "column": "width"},
    "GH":  {"file": "2000_5_phe.csv", "column": "Hard"},
    "GP":  {"file": "2000_6_phe.csv", "column": "Prot"},
}
TRAITS_TO_RUN = ("TKW", "TW", "GL", "GW", "GH", "GP")

EXPECTED_N_SAMPLES = 2000
EXPECTED_N_MARKERS = 17694
PRESERVE_MARKER_COLUMN_ORDER = True
ALIGN_PHENOTYPES_BY_SAMPLE_ID = True
EXCLUDE_SAMPLE_ID_FROM_MODEL_INPUT = True

# ---------------------------------------------------------------------------
# 3. Data policy: no additional preprocessing
# ---------------------------------------------------------------------------

PREPROCESSING_MODE = "none_use_prepared_input_as_supplied"
GENOTYPE_IMPUTATION = False
GENOTYPE_SCALING = False
GENOTYPE_NORMALIZATION = False
PHENOTYPE_IMPUTATION = False
PHENOTYPE_SCALING = False
PHENOTYPE_NORMALIZATION = False
PHENOTYPE_TRANSFORMATION = False
FEATURE_SELECTION_METHOD = "none_all_supplied_markers_retained"

# Unexpected missing/non-finite values cause an explicit error; no silent repair.
MISSING_VALUE_POLICY = "raise"
NONFINITE_VALUE_POLICY = "raise"
INPUT_NUMPY_DTYPE = "float32"
PHENOTYPE_TARGET_SCALE = "as_supplied_no_transform"

# ---------------------------------------------------------------------------
# 4. Locked repeated cross-validation
# ---------------------------------------------------------------------------

CV_SEEDS = (42, 2024, 2025)
N_REPEATS = 3
N_FOLDS = 10
N_OUTER_EVALUATIONS_PER_TRAIT = N_REPEATS * N_FOLDS
N_TOTAL_EXPECTED_FOLD_EVALUATIONS = len(TRAITS_TO_RUN) * N_OUTER_EVALUATIONS_PER_TRAIT

OUTER_TEST_FRACTION = 0.10
INNER_VALIDATION_FRACTION = 0.10

EXPECTED_OUTER_TRAIN_N = 1800
EXPECTED_OUTER_TEST_N = 200
EXPECTED_INNER_TRAIN_N = 1620
EXPECTED_INNER_VALIDATION_N = 180

USE_LOCKED_SPLITS = True
GENERATE_SPLITS_DURING_MAIN_RUN = False
REQUIRE_ALL_PROTOCOL_FILES = True
REQUIRE_PROTOCOL_AUDIT_PASS = True
REQUIRE_DATA_FINGERPRINT_MATCH = True
REQUIRE_EXACT_SPLIT_COUNTS = True

# Read this value from the locked inner split file for each fold.
INNER_SPLIT_SEED_SOURCE = "locked_inner_validation_split_file"

# Original OurModel_1.py seed derivation.
# selection = cv_seed * 10000 + fold * 101 + 17
# refit     = cv_seed * 10000 + fold * 101 + 37
CV_SEED_MULTIPLIER = 10000
FOLD_INDEX_MULTIPLIER = 101
SELECTION_SEED_OFFSET = 17
REFIT_SEED_OFFSET = 37

# ---------------------------------------------------------------------------
# 5. DCF-VAE architecture
# ---------------------------------------------------------------------------

INITIAL_PROJECTION_DIM = 256
INITIAL_PROJECTION_ACTIVATION = "GELU"

MAXPOOL_KERNEL_SIZE = 2
MAXPOOL_STRIDE = 2
EXPECTED_POOLED_LENGTH = 128

DILATION_RATES = (2, 4, 6)
BRANCH_KERNEL_SIZES = (3, 3, 3)
BRANCH_CHANNELS = 64
BRANCH_INPUT_CHANNELS = 1
BRANCH_STRIDE = 1
BRANCH_PADDING_MODE = "dilation_same_length"
BRANCH_ACTIVATION = "GELU"

USE_BRANCH_LAYER_NORM = True
LAYER_NORM_CHANNELS = 64

FUSION_INPUT_CHANNELS = 192
FUSION_OUTPUT_CHANNELS = 32
FUSION_KERNEL_SIZE = 1
FUSION_ACTIVATION = "GELU"
DROPOUT_RATE = 0.20
FLATTENED_ENCODER_DIM = 4096
LATENT_DIM = 64

# The user-approved implementation choice for dimensions not specified
# explicitly in Eqs. 10-12: 64 -> 512 -> 512 -> D.
DECODER_HIDDEN_DIMS = (512, 512)
DECODER_HIDDEN_ACTIVATION = "GELU"
DECODER_OUTPUT_ACTIVATION = "linear"

# ---------------------------------------------------------------------------
# 6. DAGSC and Chebyshev graph convolution
# ---------------------------------------------------------------------------

GRAPH_MODE = "sparse_dynamic"
GRAPH_TOPK = 8
GRAPH_SIMILARITY = "columnwise_cosine_similarity"
GRAPH_SIMILARITY_ACTIVATION = "ELU"
GRAPH_NORMALIZATION = "row_softmax"
GRAPH_TOPK_EXCLUDES_DIAGONAL = True
GRAPH_SYMMETRIZATION = "elementwise_max"
GRAPH_SELF_LOOP_WEIGHT = 1.0
GRAPH_NORMALIZATION_EPSILON = 1e-8

# Minibatch training uses sampled Z; validation/test graphs are constructed
# from posterior means of the corresponding training partition only.
GRAPH_TRAINING_LATENT_SOURCE = "sampled_latent_Z"
GRAPH_VALIDATION_REFERENCE_SOURCE = "all_inner_training_posterior_means"
GRAPH_OUTER_TEST_REFERENCE_SOURCE = "all_outer_training_posterior_means"
GRAPH_INFERENCE_LATENT_SOURCE = "posterior_mean_mu"
USE_GRAPH_EMA = False

# R is polynomial degree. R=2 therefore gives three terms: T0, T1, T2.
CHEBYSHEV_DEGREE = 2
CHEBYSHEV_NUM_TERMS = 3
CHEBYSHEV_LAMBDA_MAX_APPROX = 2.0
CHEBYSHEV_SCALED_LAPLACIAN = "normalized_laplacian_minus_identity"

REGRESSION_POOLING = "global_mean_pooling"
REGRESSION_HEAD = "final_fully_connected"
USE_POSTERIOR_MEAN_FOR_INFERENCE = True

# ---------------------------------------------------------------------------
# 7. Loss function
# ---------------------------------------------------------------------------

PREDICTION_LOSS = "mean_squared_error"
RECONSTRUCTION_LOSS = "mean_squared_error"
KL_LOSS = "diagonal_gaussian_kl_mean"

LAMBDA_PRED = 1.0
LAMBDA_RECON = 0.25
LAMBDA_KL = 0.001

# ---------------------------------------------------------------------------
# 8. Training and epoch selection
# ---------------------------------------------------------------------------

OPTIMIZER_NAME = "AdamW"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 220
GRAD_CLIP_NORM = 5.0

EARLY_STOPPING_PATIENCE = 35
MIN_EPOCHS_BEFORE_STOPPING = 45
EARLY_STOPPING_MIN_DELTA_ORIGINAL_MAE = 1e-4

EPOCH_SELECTION_METRIC = "inner_validation_MAE_original_scale_minimum"
EPOCH_SELECTION_TIE_BREAK = "earliest_epoch"

SHUFFLE_INNER_TRAIN = True
SHUFFLE_OUTER_TRAIN = True
SHUFFLE_VALIDATION = False
SHUFFLE_OUTER_TEST = False
NUM_WORKERS = 0
PIN_MEMORY_IF_CUDA = True

# ---------------------------------------------------------------------------
# 9. Computational-efficiency and accuracy-cost benchmarking
# ---------------------------------------------------------------------------

BENCHMARK_SCHEMA_VERSION = "4.1.1"

MEASURE_MODEL_PARAMETER_COUNT = True
REPORT_TOTAL_TRAINABLE_PARAMETERS = True
REPORT_INFERENCE_ACTIVE_PARAMETERS = True
PARAMETER_COUNT_SCOPE = (
    "report both: all trainable parameters in the full training model and "
    "parameters active on the prediction-only inference path"
)
MEASURE_ANALYTICAL_FORWARD_FLOPS = True
MEASURE_TRAINING_TIME_PER_OUTER_FOLD = True
MEASURE_INFERENCE_LATENCY_PER_SAMPLE = True
MEASURE_PEAK_GPU_MEMORY = True
MEASURE_PEAK_SYSTEM_RAM = True

MAC_TO_FLOP_MULTIPLIER = 2
FLOPS_BATCH_SIZE = 1
FLOPS_COUNTING_SCOPE = (
    "prediction_only: encoder_to_mu, frozen_graph_chebyshev_filter, "
    "global_mean_pooling, final_regression"
)
FLOPS_EXCLUDE_DECODER = True
FLOPS_EXCLUDE_RECONSTRUCTION_AND_KL = True
FLOPS_EXCLUDE_GRAPH_CONSTRUCTION = True
FLOPS_EXCLUDE_DATA_LOADING_AND_TRANSFER = True

MEASURE_INNER_SELECTION_TIME = True
MEASURE_OUTER_REFIT_TIME = True
MEASURE_GRAPH_CONSTRUCTION_TIME = True
SAVE_TOTAL_OUTER_FOLD_TIME = True

MEASURE_BATCH1_LATENCY_EVERY_FOLD = True
BATCH1_WARMUP_REPETITIONS = 10
BATCH1_MODEL_FORWARD_REPETITIONS = 100
BATCH1_END_TO_END_REPETITIONS = 20

RSS_MONITOR_INTERVAL_SECONDS = 0.05
REQUIRE_PSUTIL_FOR_FINAL_BENCHMARK = True

DEVICE_PREFERENCE = "auto"
DETERMINISTIC_TORCH = True
CPU_NUM_THREADS_IF_NO_CUDA = 1
SYNCHRONIZE_CUDA_FOR_TIMING = True

# ---------------------------------------------------------------------------
# 10. Output and figure settings
# ---------------------------------------------------------------------------

SAVE_FOLD_METRICS_AFTER_EACH_FOLD = True
SAVE_OUTER_TEST_PREDICTIONS_AFTER_EACH_FOLD = True
SAVE_TRAINING_HISTORY = True
SAVE_FOLD_RESOURCE_RECORDS_AFTER_EACH_FOLD = True
SAVE_PROTOCOL_AUDIT = True
SAVE_RUN_MANIFEST = True

SAVE_FOLD_CHECKPOINTS = False
SAVE_FINAL_MODEL_CHECKPOINTS = False
OVERWRITE_EXISTING = False
RESUME_COMPLETED_FOLDS = False

FIGURE_DPI = 300
FIGURE_FORMATS = ("png", "tiff", "pdf", "eps", "svg")
FIGURE_FONT_FAMILY = "Times New Roman"
PREDICTION_STD_EPS = 1e-8

RESULTS_SUBDIRECTORIES = (
    "predictive_performance",
    "predictions",
    "training_history",
    "computational_efficiency",
    "accuracy_cost_tradeoff",
    "figures",
    "logs",
    "fold_records",
)

# ---------------------------------------------------------------------------
# 11. Immutable model configuration passed to model.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    input_projection_dim: int = INITIAL_PROJECTION_DIM
    pooled_length: int = EXPECTED_POOLED_LENGTH
    dilation_rates: Tuple[int, ...] = DILATION_RATES
    branch_kernel_sizes: Tuple[int, ...] = BRANCH_KERNEL_SIZES
    branch_channels: int = BRANCH_CHANNELS
    fusion_channels: int = FUSION_OUTPUT_CHANNELS
    latent_dim: int = LATENT_DIM
    dropout_rate: float = DROPOUT_RATE
    decoder_hidden_dims: Tuple[int, int] = DECODER_HIDDEN_DIMS

    graph_mode: str = GRAPH_MODE
    graph_topk: int = GRAPH_TOPK
    graph_self_loop_weight: float = GRAPH_SELF_LOOP_WEIGHT
    graph_norm_epsilon: float = GRAPH_NORMALIZATION_EPSILON

    chebyshev_degree: int = CHEBYSHEV_DEGREE
    chebyshev_num_terms: int = CHEBYSHEV_NUM_TERMS
    chebyshev_lambda_max_approx: float = CHEBYSHEV_LAMBDA_MAX_APPROX

    lambda_pred: float = LAMBDA_PRED
    lambda_recon: float = LAMBDA_RECON
    lambda_kl: float = LAMBDA_KL

    label: str = "dvagraph_gp_mathematical_specification_R2"

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if not self.dilation_rates or len(self.dilation_rates) != len(self.branch_kernel_sizes):
            raise ValueError("Dilation rates and kernel sizes must have equal nonzero lengths.")
        if any(v < 1 for v in self.dilation_rates + self.branch_kernel_sizes):
            raise ValueError("Dilation rates and kernel sizes must be positive.")
        if self.graph_mode not in {"sparse_dynamic", "none"}:
            raise ValueError(f"Unsupported graph mode: {self.graph_mode}")
        if self.graph_mode == "sparse_dynamic" and not 1 <= self.graph_topk < self.latent_dim:
            raise ValueError("Sparse graph requires 1 <= graph_topk < latent_dim.")
        if self.chebyshev_degree != 2 or self.chebyshev_num_terms != 3:
            raise ValueError("The approved Chebyshev configuration is R=2 with T0, T1, T2.")
        if len(self.decoder_hidden_dims) != 2 or any(v < 1 for v in self.decoder_hidden_dims):
            raise ValueError("Decoder must have two positive hidden dimensions.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")
        if min(self.lambda_pred, self.lambda_recon, self.lambda_kl) < 0:
            raise ValueError("Loss weights cannot be negative.")


MODEL_CONFIG = ModelConfig()


# ---------------------------------------------------------------------------
# 12. Configuration and input-path validation
# ---------------------------------------------------------------------------

def validate_config() -> None:
    """Validate fixed settings without requiring data files to be present."""

    MODEL_CONFIG.validate()

    if len(CV_SEEDS) != N_REPEATS or len(set(CV_SEEDS)) != len(CV_SEEDS):
        raise ValueError("CV_SEEDS must contain three unique seeds.")
    if N_REPEATS != 3 or N_FOLDS != 10 or N_OUTER_EVALUATIONS_PER_TRAIT != 30:
        raise ValueError("The locked protocol requires 3 repetitions x 10 folds.")
    if EXPECTED_OUTER_TRAIN_N + EXPECTED_OUTER_TEST_N != EXPECTED_N_SAMPLES:
        raise ValueError("Outer train/test counts do not sum to the sample count.")
    if EXPECTED_INNER_TRAIN_N + EXPECTED_INNER_VALIDATION_N != EXPECTED_OUTER_TRAIN_N:
        raise ValueError("Inner train/validation counts do not sum to outer training.")
    if CHEBYSHEV_DEGREE != 2 or CHEBYSHEV_NUM_TERMS != CHEBYSHEV_DEGREE + 1:
        raise ValueError("Chebyshev degree must be R=2 with three polynomial terms.")
    if GENERATE_SPLITS_DURING_MAIN_RUN or not USE_LOCKED_SPLITS:
        raise ValueError("The main run must use the existing locked split files.")
    if PREPROCESSING_MODE != "none_use_prepared_input_as_supplied":
        raise ValueError("This project requires direct use of prepared input values.")
    if any((GENOTYPE_IMPUTATION, GENOTYPE_SCALING, GENOTYPE_NORMALIZATION,
            PHENOTYPE_IMPUTATION, PHENOTYPE_SCALING, PHENOTYPE_NORMALIZATION,
            PHENOTYPE_TRANSFORMATION)):
        raise ValueError("Additional data preprocessing must remain disabled.")
    if set(TRAITS_TO_RUN) != set(TRAIT_CONFIGS):
        raise ValueError("TRAITS_TO_RUN and TRAIT_CONFIGS do not match.")
    if BATCH_SIZE < 1 or MAX_EPOCHS < 1 or EARLY_STOPPING_PATIENCE < 1:
        raise ValueError("Training settings must be positive.")
    if MIN_EPOCHS_BEFORE_STOPPING > MAX_EPOCHS:
        raise ValueError("Minimum epochs cannot exceed maximum epochs.")
    if RSS_MONITOR_INTERVAL_SECONDS <= 0:
        raise ValueError("RSS monitor interval must be positive.")


def validate_required_files() -> None:
    """Check data and locked protocol files before main.py starts training."""

    missing = []

    if not GENOTYPE_PATH.is_file():
        missing.append(f"Genotype file: {GENOTYPE_PATH}")

    for trait in TRAITS_TO_RUN:
        phenotype_path = PHENOTYPE_DIR / TRAIT_CONFIGS[trait]["file"]
        if not phenotype_path.is_file():
            missing.append(f"{trait} phenotype file: {phenotype_path}")

    if REQUIRE_ALL_PROTOCOL_FILES:
        for protocol_path in REQUIRED_PROTOCOL_FILES:
            if not protocol_path.is_file():
                missing.append(f"Locked protocol file: {protocol_path}")

    if missing:
        details = "\n".join(f" - {item}" for item in missing)
        raise FileNotFoundError("Required input files are missing:\n" + details)


validate_config()
