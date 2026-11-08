
"""
DVAGraph-GP Configuration
-------------------------
Central configuration for the DVAGraph-GP model.

This file contains dataset paths, model hyperparameters,
training settings, cross-validation settings and evaluation settings.

Initial configuration is preserved from the original runner.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

# Set before importing PyTorch for deterministic CUDA operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


# ============================================================
# 1. PROJECT AND DATASET PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# Original paths from OurModel_1.py
# Change these if your data or results are stored elsewhere.

DATA_ROOT = Path(r"yourpath\data")

RESULTS_ROOT = Path(r"D:yourpath\Results")

WHEAT_DATA_DIR = DATA_ROOT / "wheat2000"

GENOTYPE_PATH = WHEAT_DATA_DIR / "genotype_markers.csv"

PHENO_DIR = DATA_ROOT

# Output directory
DATASET_NAME = "Wheat2000"

METHOD_NAME = "DVAGraph_GP_revised_sparse_feature_graph_main"

RUN_NAME = "reviewer_ready_outerinner_v3_1_harmonized_tensor_schemafix"

OUTPUT_ROOT = (
    RESULTS_ROOT
    / "DVAGraph-GP"
    / "wheat"
    / RUN_NAME
)

# Locked cross-validation protocol directory
PROTOCOL_DIR = (
    RESULTS_ROOT
    / "Shared_Benchmark_Protocol"
    / "Wheat2000_Outer3x10_InnerVal_v5"
)

OUTER_SPLIT_PATH = PROTOCOL_DIR / "wheat2000_repeated_outer_cv_splits.csv"
OUTER_AUDIT_PATH = PROTOCOL_DIR / "wheat2000_repeated_outer_cv_split_audit.csv"

INNER_SPLIT_PATH = PROTOCOL_DIR / "wheat2000_repeated_inner_validation_splits.csv"
INNER_AUDIT_PATH = PROTOCOL_DIR / "wheat2000_repeated_inner_validation_split_audit.csv"

FINGERPRINT_PATH = PROTOCOL_DIR / "wheat2000_data_fingerprints.json"
COST_POLICY_PATH = PROTOCOL_DIR / "benchmark_cost_policy.json"
TRAINING_POLICY_PATH = PROTOCOL_DIR / "model_training_policy.json"
PROTOCOL_MANIFEST_PATH = PROTOCOL_DIR / "protocol_manifest.json"


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

GENOTYPE_SAMPLE_ID_COLUMN = "sample"

PHENOTYPE_SAMPLE_ID_COLUMN = None

PHENOTYPE_SAMPLE_ID_CANDIDATES = (
    "sample",
    "Sample",
    "SAMPLE",
    "sample_id",
    "Sample_ID",
    "SAMPLE_ID",
    "id",
    "ID",
    "IID",
    "accession",
    "Accession",
)

# Phenotype files and corresponding trait columns
TRAIT_CONFIGS = {
    "TKW": {"file": "2000_1_phe.csv", "column": "tkw"},
    "TW":  {"file": "2000_2_phe.csv", "column": "testw"},
    "GL":  {"file": "2000_3_phe.csv", "column": "length"},
    "GW":  {"file": "2000_4_phe.csv", "column": "width"},
    "GH":  {"file": "2000_5_phe.csv", "column": "Hard"},
    "GP":  {"file": "2000_6_phe.csv", "column": "Prot"},
}

TRAITS_TO_RUN = ["TKW", "TW", "GL", "GW", "GH", "GP"]


# ============================================================
# 3. CROSS-VALIDATION CONFIGURATION
# ============================================================

CV_SEEDS = (42, 2024, 2025)

N_FOLDS = 10

# Total evaluations: 3 repetitions x 10 folds = 30


# ============================================================
# 4. MODEL ARCHITECTURE PARAMETERS
# ============================================================

# DCF-VAE
MAIN_LATENT_DIM = 64

MAIN_DILATION_RATES = (2, 4, 6)

MAIN_BRANCH_KERNEL_SIZES = (3, 3, 3)

MAIN_DROPOUT = 0.20

# Adaptive feature graph
MAIN_GRAPH_TOPK = 8

MAIN_GRAPH_MODE = "sparse_dynamic"

MAIN_SELF_LOOP_WEIGHT = 1.0

MAIN_GRAPH_EMA_MOMENTUM = 0.90

# Use latent mean for phenotype prediction
USE_MU_FOR_PREDICTION = True

# Chebyshev graph convolution
# 3 terms correspond to T0, T1 and T2 (polynomial degree 2).
MAIN_CHEBYSHEV_ORDER = 2


# ============================================================
# 5. LOSS FUNCTION CONFIGURATION
# ============================================================

LAMBDA_PRED = 1.0

LAMBDA_RECON = 0.25

LAMBDA_KL = 0.001

RECONSTRUCTION_LOSS = "mse_binary_01"


# ============================================================
# 6. TRAINING HYPERPARAMETERS
# ============================================================

MAX_EPOCHS = 220

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

GRAD_CLIP_NORM = 5.0

OPTIMIZER_NAME = "AdamW"

# Early stopping
EARLY_STOPPING_PATIENCE = 35

MIN_EPOCHS_BEFORE_STOPPING = 45

EARLY_STOPPING_MIN_DELTA_ORIGINAL_MAE = 1e-4

EPOCH_SELECTION_METRIC = "inner_validation_MAE_original_minimum"

EPOCH_SELECTION_TIE_BREAK = "earliest_epoch"


# ============================================================
# 7. PREPROCESSING CONFIGURATION
# ============================================================

MARKER_PREPROCESSING = (
    "train_only_most_frequent_imputation_then_raw_binary_0_1_retained"
)

FEATURE_SELECTION_METHOD = "none_all_observed_markers_retained"

MARKER_ORDER_STATUS = (
    "unverified_input_column_order_after_clone_representative_preprocessing"
)

MODEL_MARKER_ORDER = (
    "input_column_order; convolution_over_learned_256_feature_embedding_"
    "not_direct_physical_genome_positions"
)


# ============================================================
# 8. EVALUATION AND FIGURE SETTINGS
# ============================================================

FIGURE_DPI = 300

FIGURE_FORMATS = ("png", "tiff", "pdf", "eps", "svg")

FIGURE_FONT_FAMILY = "Times New Roman"

PREDICTION_STD_EPS = 1e-8

BENCHMARK_SCHEMA_VERSION = "4.1.1"


# ============================================================
# 9. COMPUTATIONAL SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DETERMINISTIC_TORCH = True

RSS_MONITOR_INTERVAL_SECONDS = 0.05

REQUIRE_PSUTIL_FOR_FINAL_BENCHMARK = True

MEASURE_BATCH1_LATENCY_EVERY_FOLD = True

BATCH1_WARMUP_REPETITIONS = 10

BATCH1_MODEL_FORWARD_REPETITIONS = 100

BATCH1_END_TO_END_REPETITIONS = 20


# ============================================================
# 10. OPTIONAL EXPERIMENT SETTINGS
# ============================================================

SAVE_FOLD_CHECKPOINTS = False

SAVE_FOLD_GRADIENT_TOP_MARKERS = False

FOLD_GRADIENT_TOP_MARKERS = 100

# Retained for compatibility with additional analyses
INTERPRETABILITY_EPOCHS_IF_MAIN_NOT_AVAILABLE = 100

INTERPRETABILITY_MAX_SAMPLES_FOR_GRADIENTS = 256

NODE_TRACE_TOP_MARKERS = 50

PREDICTION_TRACE_TOP_MARKERS = 100

UMAP_RANDOM_SEED = 42

# Optional annotation files
MARKER_MAPPING_PATH = WHEAT_DATA_DIR / "marker_mapping.csv"

MARKER_ANNOTATION_PATH: Optional[Path] = None

QTL_INTERVAL_PATH: Optional[Path] = None

GENE_TO_GO_PATH: Optional[Path] = None

POPULATION_LABEL_PATH: Optional[Path] = None

OVERWRITE_EXISTING = False


# ============================================================
# 11. MODEL CONFIGURATION CLASS
# ============================================================

@dataclass(frozen=True)
class ModelConfig:

    latent_dim: int = MAIN_LATENT_DIM

    dilation_rates: Tuple[int, ...] = MAIN_DILATION_RATES

    branch_kernel_sizes: Tuple[int, ...] = MAIN_BRANCH_KERNEL_SIZES

    graph_topk: Optional[int] = MAIN_GRAPH_TOPK

    graph_mode: str = MAIN_GRAPH_MODE

    dropout_rate: float = MAIN_DROPOUT

    chebyshev_order: int = MAIN_CHEBYSHEV_ORDER

    self_loop_weight: float = MAIN_SELF_LOOP_WEIGHT

    graph_ema_momentum: float = MAIN_GRAPH_EMA_MOMENTUM

    lambda_pred: float = LAMBDA_PRED

    lambda_recon: float = LAMBDA_RECON

    lambda_kl: float = LAMBDA_KL

    label: str = "main_pre_specified"

    def validate(self):

        if (
            len(self.dilation_rates) != len(self.branch_kernel_sizes)
            or not self.dilation_rates
        ):
            raise ValueError(
                "Dilation rates and kernel sizes must have equal nonzero lengths."
            )

        if self.graph_mode not in {
            "sparse_dynamic",
            "dense_dynamic",
            "static_sparse",
            "none",
        }:
            raise ValueError(
                f"Unknown graph mode: {self.graph_mode}"
            )

        if self.latent_dim < 4 or self.chebyshev_order < 1:
            raise ValueError(
                "latent_dim must be >= 4 and chebyshev_order >= 1."
            )

        if self.graph_mode in {"sparse_dynamic", "static_sparse"}:

            if self.graph_topk is None or self.graph_topk < 1:

                raise ValueError(
                    "Sparse graph modes require graph_topk >= 1."
                )
