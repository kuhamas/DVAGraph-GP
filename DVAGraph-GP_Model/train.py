"""
DVAGraph-GP: reusable training engine
=====================================

This module implements the model-fitting procedure independently of any one
dataset. Dataset loading, preprocessing, sample alignment, and construction /
fingerprinting of locked CV splits belong to data.py. Experiment-specific
values are supplied through TrainingSettings (normally created from config.py).

Approved model contract
-----------------------
The model is imported from model.py and is not redefined here.

Training:
    sampled latent Z
    dynamic minibatch graph built by model.py
    prediction + genotype reconstruction + KL objective

Validation:
    posterior means from the fitted inner-training partition
    one frozen graph built from those inner-training means only
    deterministic prediction on inner-validation samples
    select epoch by minimum original-scale validation MAE

Outer refit:
    fresh model initialization
    refit on the complete outer-training partition for the selected epoch count
    one final graph built from outer-training posterior means only
    deterministic prediction on the untouched outer-test partition

No preprocessing, imputation, feature selection, phenotype scaling, graph EMA,
or split generation is performed in this module.
"""

from __future__ import annotations

import os

# Must be set before importing torch. config.py also sets this before its own
# torch import; keeping it here makes direct imports of train.py safer.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gc
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import (
    LATENT_DIM,
    GRAPH_TOP_K,
    CHEBYSHEV_DEGREE,
    DILATION_RATES,
    BRANCH_KERNEL_SIZE,
    DROPOUT_RATE,
    LAMBDA_PRED as MODEL_LAMBDA_PRED,
    LAMBDA_RECON as MODEL_LAMBDA_RECON,
    LAMBDA_KL as MODEL_LAMBDA_KL,
    DVAGraphGP,
    dvagraph_loss,
)


# ---------------------------------------------------------------------------
# 1. Fold-specification interface
# ---------------------------------------------------------------------------

class FoldSpecLike(Protocol):
    """Structural interface expected from data.py's OuterInnerFoldSpec."""

    repeat: int
    cv_seed: int
    fold: int
    inner_split_seed: int
    inner_train_indices: np.ndarray
    inner_validation_indices: np.ndarray
    outer_train_indices: np.ndarray
    outer_test_indices: np.ndarray


# ---------------------------------------------------------------------------
# 2. Reusable training settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingSettings:
    """
    Settings for the reusable training engine.

    These are experiment settings, not model-architecture parameters.
    The mathematical architecture is fixed in model.py.

    For a new dataset or CV protocol, create a new instance or construct one
    using TrainingSettings.from_config(config).
    """

    batch_size: int = 32
    max_epochs: int = 220
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    optimizer_name: str = "AdamW"

    early_stopping_patience: int = 35
    min_epochs_before_stopping: int = 45
    early_stopping_min_delta: float = 1e-4
    epoch_selection_metric: str = "inner_validation_MAE_original_minimum"
    epoch_selection_tie_break: str = "earliest_epoch"

    lambda_pred: float = MODEL_LAMBDA_PRED
    lambda_recon: float = MODEL_LAMBDA_RECON
    lambda_kl: float = MODEL_LAMBDA_KL

    device: Optional[Union[str, torch.device]] = None
    deterministic_torch: bool = True
    graph_freeze_batch_size: int = 256
    num_workers: int = 0

    # Optional protocol expectations. If supplied, these are checked against
    # fold_specs; train.py does not generate or modify the protocol.
    expected_cv_seeds: Optional[Tuple[int, ...]] = None
    expected_folds_per_repeat: Optional[int] = None
    require_each_sample_once_per_repeat: bool = True
    require_outer_partition_complete: bool = True
    require_inner_partition_complete: bool = True

    # Preserve the seed derivation used by the original Wheat2000 runner.
    # Other experiments can override these constants in their own config.
    cv_seed_multiplier: int = 10000
    fold_seed_multiplier: int = 101
    selection_seed_offset: int = 17
    refit_seed_offset: int = 37

    save_fold_checkpoints: bool = False
    overwrite_existing: bool = False
    output_dir: Optional[Union[str, Path]] = None

    dataset_name: str = "unspecified_dataset"
    method_name: str = "DVAGraph-GP"

    @classmethod
    def from_config(cls, config_module: Any, **overrides: Any) -> "TrainingSettings":
        """
        Build settings from the constants in config.py.

        Missing optional config constants fall back to the defaults above.
        This keeps train.py usable with different experiment configurations.
        """

        mapping = {
            "batch_size": "BATCH_SIZE",
            "max_epochs": "MAX_EPOCHS",
            "learning_rate": "LEARNING_RATE",
            "weight_decay": "WEIGHT_DECAY",
            "grad_clip_norm": "GRAD_CLIP_NORM",
            "optimizer_name": "OPTIMIZER_NAME",
            "early_stopping_patience": "EARLY_STOPPING_PATIENCE",
            "min_epochs_before_stopping": "MIN_EPOCHS_BEFORE_STOPPING",
            "early_stopping_min_delta": "EARLY_STOPPING_MIN_DELTA_ORIGINAL_MAE",
            "epoch_selection_metric": "EPOCH_SELECTION_METRIC",
            "epoch_selection_tie_break": "EPOCH_SELECTION_TIE_BREAK",
            "lambda_pred": "LAMBDA_PRED",
            "lambda_recon": "LAMBDA_RECON",
            "lambda_kl": "LAMBDA_KL",
            "device": "DEVICE",
            "deterministic_torch": "DETERMINISTIC_TORCH",
            "save_fold_checkpoints": "SAVE_FOLD_CHECKPOINTS",
            "overwrite_existing": "OVERWRITE_EXISTING",
            "output_dir": "OUTPUT_ROOT",
            "dataset_name": "DATASET_NAME",
            "method_name": "METHOD_NAME",
            "expected_cv_seeds": "CV_SEEDS",
            "expected_folds_per_repeat": "N_FOLDS",
            "graph_freeze_batch_size": "GRAPH_FREEZE_BATCH_SIZE",
        }

        values: Dict[str, Any] = {}
        defaults = cls()

        for field_name, config_name in mapping.items():
            if hasattr(config_module, config_name):
                value = getattr(config_module, config_name)
                if field_name == "expected_cv_seeds":
                    value = tuple(int(v) for v in value)
                values[field_name] = value
            else:
                values[field_name] = getattr(defaults, field_name)

        values.update(overrides)
        return cls(**values)

    @property
    def resolved_device(self) -> torch.device:
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ---------------------------------------------------------------------------
# 3. Result containers
# ---------------------------------------------------------------------------

@dataclass
class InnerSelectionResult:
    selected_epoch: int
    best_validation_mae_original: float
    best_validation_mse_original: float
    epochs_completed: int
    selection_seed: int
    history: pd.DataFrame
    checkpoint_path: Optional[str] = None


@dataclass
class OuterRefitResult:
    y_true_original: np.ndarray
    y_pred_original: np.ndarray
    test_indices: np.ndarray
    test_sample_ids: np.ndarray
    history: pd.DataFrame
    final_adjacency: np.ndarray
    trainable_parameters: int
    refit_seed: int
    checkpoint_path: Optional[str] = None


@dataclass
class FoldTrainingResult:
    dataset: str
    method: str
    trait: str

    repeat: int
    cv_seed: int
    fold: int
    inner_split_seed: int

    selection_seed: int
    refit_seed: int

    selected_epoch: int
    inner_epochs_completed: int
    best_inner_validation_mae_original: float
    best_inner_validation_mse_original: float

    inner_history: pd.DataFrame
    outer_history: pd.DataFrame

    test_indices: np.ndarray
    test_sample_ids: np.ndarray
    y_true_original: np.ndarray
    y_pred_original: np.ndarray

    final_adjacency: np.ndarray
    trainable_parameters: int
    partition_audit: Dict[str, bool]

    inner_checkpoint_path: Optional[str] = None
    outer_checkpoint_path: Optional[str] = None

    @property
    def prediction_frame(self) -> pd.DataFrame:
        """Held-out predictions in original phenotype units."""

        return pd.DataFrame({
            "dataset": self.dataset,
            "method": self.method,
            "trait": self.trait,
            "repeat": self.repeat,
            "cv_split_seed": self.cv_seed,
            "fold": self.fold,
            "inner_split_seed": self.inner_split_seed,
            "selection_seed": self.selection_seed,
            "refit_seed": self.refit_seed,
            "selected_epoch": self.selected_epoch,
            "sample_id": self.test_sample_ids,
            "sample_index_after_y_filter": self.test_indices,
            "y_true_original": self.y_true_original,
            "y_pred_original": self.y_pred_original,
            "residual_original": self.y_true_original - self.y_pred_original,
        })


# ---------------------------------------------------------------------------
# 4. Configuration and model-contract validation
# ---------------------------------------------------------------------------

def validate_training_settings(settings: TrainingSettings) -> None:
    """Validate optimization settings without imposing dataset-specific sizes."""

    if settings.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if settings.max_epochs < 1:
        raise ValueError("max_epochs must be >= 1.")
    if settings.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if settings.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
    if settings.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive.")
    if settings.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be >= 1.")
    if settings.min_epochs_before_stopping < 1:
        raise ValueError("min_epochs_before_stopping must be >= 1.")
    if settings.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta cannot be negative.")
    if settings.graph_freeze_batch_size < 1:
        raise ValueError("graph_freeze_batch_size must be >= 1.")
    if settings.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if settings.optimizer_name.lower() != "adamw":
        raise NotImplementedError(
            "This approved training engine currently implements AdamW only."
        )
    if settings.epoch_selection_metric != "inner_validation_MAE_original_minimum":
        raise ValueError(
            "The approved epoch-selection criterion is minimum original-scale "
            "inner-validation MAE."
        )
    if settings.epoch_selection_tie_break != "earliest_epoch":
        raise ValueError("The approved tie-break rule is earliest_epoch.")
    if min(settings.lambda_pred, settings.lambda_recon, settings.lambda_kl) < 0:
        raise ValueError("Loss weights cannot be negative.")


def validate_model_contract() -> None:
    """
    Guard the fixed approved main architecture.

    These are model-level constants, not dataset-specific training settings.
    Changing the architecture requires an explicit model revision.
    """

    expected = {
        "latent_dim": (LATENT_DIM, 64),
        "graph_top_k": (GRAPH_TOP_K, 8),
        "Chebyshev degree": (CHEBYSHEV_DEGREE, 2),
        "dilation rates": (tuple(DILATION_RATES), (2, 4, 6)),
        "branch kernel size": (BRANCH_KERNEL_SIZE, 3),
        "dropout": (DROPOUT_RATE, 0.20),
    }
    mismatches = [
        f"{name}: model.py={actual!r}, approved={required!r}"
        for name, (actual, required) in expected.items()
        if actual != required
    ]
    if mismatches:
        raise RuntimeError(
            "model.py does not match the approved DVAGraph-GP architecture:\n- "
            + "\n- ".join(mismatches)
        )


# ---------------------------------------------------------------------------
# 5. Reproducibility, loaders, and optimizer
# ---------------------------------------------------------------------------

def set_global_seed(seed: int, settings: TrainingSettings) -> None:
    """Set Python, NumPy, CPU, and CUDA seeds for one training stage."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(settings.deterministic_torch)

    torch.use_deterministic_algorithms(
        bool(settings.deterministic_torch),
        warn_only=True,
    )


def _validate_xy(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    name: str = "partition",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Validate already-prepared arrays.

    No imputation, normalization, scaling, encoding, or feature selection occurs.
    Conversion to float32 is solely for PyTorch computation.
    """

    X_array = np.asarray(X)
    if X_array.ndim != 2:
        raise ValueError(f"{name}: X must be [n_samples, n_features], got {X_array.shape}.")
    if X_array.shape[0] < 1 or X_array.shape[1] < 1:
        raise ValueError(f"{name}: X must have nonzero sample and feature dimensions.")
    if not np.isfinite(X_array).all():
        raise ValueError(
            f"{name}: X contains NaN or infinite values. train.py does not impute."
        )

    X_array = np.ascontiguousarray(X_array, dtype=np.float32)

    if y is None:
        return X_array, None

    y_array = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_array.shape[0] != X_array.shape[0]:
        raise ValueError(
            f"{name}: X/y sample mismatch ({X_array.shape[0]} vs {y_array.shape[0]})."
        )
    if not np.isfinite(y_array).all():
        raise ValueError(
            f"{name}: y contains NaN or infinite values. train.py does not impute."
        )

    # Keep targets in their supplied/original units. Conversion to float32 is
    # performed only when creating the PyTorch tensor.
    return X_array, y_array


def make_data_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    settings: TrainingSettings,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    """Build a deterministic loader without changing the data values."""

    X_array, y_array = _validate_xy(X, y, name="DataLoader")
    assert y_array is not None

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    dataset = TensorDataset(
        torch.from_numpy(X_array),
        torch.from_numpy(np.ascontiguousarray(y_array.reshape(-1, 1), dtype=np.float32)),
    )

    return DataLoader(
        dataset,
        batch_size=int(settings.batch_size),
        shuffle=bool(shuffle),
        num_workers=int(settings.num_workers),
        generator=generator if shuffle else None,
        pin_memory=settings.resolved_device.type == "cuda",
        drop_last=False,
    )


def make_optimizer(
    model: nn.Module,
    settings: TrainingSettings,
) -> torch.optim.Optimizer:
    """Construct the optimizer specified for the experiment."""

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
    )


# ---------------------------------------------------------------------------
# 6. One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: DVAGraphGP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    settings: TrainingSettings,
) -> Dict[str, float]:
    """
    Fit one epoch using sampled Z and the dynamic minibatch graph.

    The model's training-mode forward pass is responsible for:
      - VAE reparameterization;
      - genotype reconstruction;
      - dynamic sparse graph construction from sampled Z;
      - degree-2 Chebyshev filtering;
      - global mean pooling and phenotype regression.

    The loss is the approved weighted sum of prediction MSE, reconstruction
    MSE, and KL divergence. Phenotype targets are not scaled in this module.
    """

    model.train()
    device = settings.resolved_device

    metric_names = (
        "prediction_loss",
        "reconstruction_loss",
        "kl_loss",
        "weighted_prediction_loss",
        "weighted_reconstruction_loss",
        "weighted_kl_loss",
        "total_loss",
    )
    totals = {name: 0.0 for name in metric_names}
    gradient_norm_sum = 0.0
    n_samples = 0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        y_batch = y_batch.to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        output = model(X_batch)
        loss, components = dvagraph_loss(
            output,
            y_batch,
            lambda_pred=float(settings.lambda_pred),
            lambda_recon=float(settings.lambda_recon),
            lambda_kl=float(settings.lambda_kl),
        )

        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite total loss during training.")

        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(settings.grad_clip_norm),
            error_if_nonfinite=True,
        )
        optimizer.step()

        batch_size = int(X_batch.shape[0])
        for name in metric_names:
            totals[name] += float(components[name].detach().item()) * batch_size

        gradient_norm_sum += float(gradient_norm.detach().item())
        n_samples += batch_size
        n_batches += 1

    if n_samples == 0:
        raise RuntimeError("The training loader produced no observations.")

    metrics = {name: value / n_samples for name, value in totals.items()}
    metrics["mean_preclip_gradient_norm"] = gradient_norm_sum / max(n_batches, 1)
    metrics["n_training_observations"] = int(n_samples)
    metrics["n_training_batches"] = int(n_batches)
    return metrics


# ---------------------------------------------------------------------------
# 7. Training-partition graph freezing and deterministic prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def freeze_training_partition_graph(
    model: DVAGraphGP,
    X_training: np.ndarray,
    settings: TrainingSettings,
) -> np.ndarray:
    """
    Freeze the graph using posterior means from the supplied training partition.

    The model's own method temporarily switches to eval mode, encodes all
    training rows in chunks, constructs one graph from their posterior means,
    stores that graph, and restores the previous model mode.
    """

    X_train, _ = _validate_xy(X_training, name="graph-source training partition")

    adjacency = model.freeze_graph_from_training_data(
        torch.from_numpy(X_train),
        batch_size=int(settings.graph_freeze_batch_size),
    )

    adjacency_np = adjacency.detach().cpu().numpy().copy()
    expected_shape = (LATENT_DIM, LATENT_DIM)

    if adjacency_np.shape != expected_shape:
        raise RuntimeError(
            f"Frozen graph shape must be {expected_shape}; got {adjacency_np.shape}."
        )
    if not np.isfinite(adjacency_np).all():
        raise FloatingPointError("Frozen graph contains non-finite values.")
    if not np.allclose(adjacency_np, adjacency_np.T, atol=1e-6):
        raise RuntimeError("Frozen graph must be symmetric.")
    if not np.allclose(np.diag(adjacency_np), 1.0, atol=1e-6):
        raise RuntimeError("Frozen graph must contain unit self-loops.")

    return adjacency_np


@torch.no_grad()
def predict_partition(
    model: DVAGraphGP,
    X: np.ndarray,
    *,
    settings: TrainingSettings,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Predict with posterior means and a previously frozen training-derived graph.

    Only genotype inputs are accepted; target labels cannot enter prediction.
    """

    X_array, _ = _validate_xy(X, name="prediction partition")

    if not model.has_frozen_graph:
        raise RuntimeError(
            "Prediction requires a frozen graph from the relevant training partition."
        )

    step = int(batch_size or settings.batch_size)
    if step < 1:
        raise ValueError("Prediction batch_size must be positive.")

    model.eval()
    predictions: List[np.ndarray] = []
    device = settings.resolved_device

    for start in range(0, X_array.shape[0], step):
        xb = torch.from_numpy(X_array[start:start + step]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        output = model(xb)
        prediction = output["prediction"]

        if not torch.isfinite(prediction).all().item():
            raise FloatingPointError("Non-finite prediction encountered.")

        predictions.append(
            prediction.detach().cpu().numpy().reshape(-1).astype(np.float64)
        )

    return np.concatenate(predictions)


def calculate_original_scale_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """Return MAE and MSE in the units supplied by the phenotype data."""

    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={truth.shape}, y_pred={pred.shape}.")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise FloatingPointError("Non-finite values in validation metrics.")

    residual = pred - truth
    return float(np.mean(np.abs(residual))), float(np.mean(residual ** 2))


# ---------------------------------------------------------------------------
# 8. Fold specification validation (protocol-agnostic)
# ---------------------------------------------------------------------------

def _get_indices(spec: FoldSpecLike, name: str, n_samples: int) -> np.ndarray:
    if not hasattr(spec, name):
        raise AttributeError(f"Fold specification is missing {name!r}.")

    indices = np.asarray(getattr(spec, name), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D index array.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} contains duplicate indices.")
    if np.any(indices < 0) or np.any(indices >= n_samples):
        raise IndexError(f"{name} contains indices outside [0, {n_samples}).")
    return indices


def validate_one_fold_spec(
    spec: FoldSpecLike,
    n_samples: int,
    settings: TrainingSettings,
) -> Dict[str, bool]:
    """Check partition disjointness, containment, and optional full coverage."""

    inner_train = set(_get_indices(spec, "inner_train_indices", n_samples).tolist())
    inner_val = set(_get_indices(spec, "inner_validation_indices", n_samples).tolist())
    outer_train = set(_get_indices(spec, "outer_train_indices", n_samples).tolist())
    outer_test = set(_get_indices(spec, "outer_test_indices", n_samples).tolist())
    all_indices = set(range(n_samples))

    checks = {
        "inner_train_validation_disjoint": not (inner_train & inner_val),
        "outer_train_test_disjoint": not (outer_train & outer_test),
        "inner_partitions_within_outer_train": (inner_train | inner_val).issubset(outer_train),
        "inner_train_within_outer_train": inner_train.issubset(outer_train),
        "inner_validation_within_outer_train": inner_val.issubset(outer_train),
    }

    if settings.require_inner_partition_complete:
        checks["inner_union_equals_outer_train"] = (inner_train | inner_val) == outer_train

    if settings.require_outer_partition_complete:
        checks["outer_union_equals_dataset"] = (outer_train | outer_test) == all_indices

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            f"Invalid fold partitions for repeat={spec.repeat}, fold={spec.fold}: {failed}"
        )

    return checks


def validate_fold_collection(
    fold_specs: Sequence[FoldSpecLike],
    n_samples: int,
    settings: TrainingSettings,
) -> None:
    """
    Validate the supplied CV design without hard-coding 3 repeats, 10 folds,
    or Wheat2000-specific partition sizes.

    If expected seeds/fold count are provided in config.py, they are checked.
    The exact-once outer-test coverage check is configurable for protocols
    such as repeated K-fold versus repeated holdout.
    """

    if not fold_specs:
        raise ValueError("No fold specifications were supplied.")

    identities = []
    groups: Dict[Tuple[int, int], List[FoldSpecLike]] = {}

    for spec in fold_specs:
        identity = (int(spec.repeat), int(spec.cv_seed), int(spec.fold))
        identities.append(identity)
        groups.setdefault((int(spec.repeat), int(spec.cv_seed)), []).append(spec)
        validate_one_fold_spec(spec, n_samples, settings)

    if len(identities) != len(set(identities)):
        raise RuntimeError("Duplicate repeat/cv_seed/fold identities were supplied.")

    if settings.expected_cv_seeds is not None:
        expected_pairs = {
            (repeat, int(seed))
            for repeat, seed in enumerate(settings.expected_cv_seeds, start=1)
        }
        observed_pairs = set(groups)
        if observed_pairs != expected_pairs:
            raise RuntimeError(
                f"CV repeat/seed identities differ from config. "
                f"Expected={sorted(expected_pairs)}, observed={sorted(observed_pairs)}."
            )

    if settings.expected_folds_per_repeat is not None:
        expected_count = int(settings.expected_folds_per_repeat)
        for key, specs in groups.items():
            fold_numbers = sorted(int(spec.fold) for spec in specs)
            expected_numbers = list(range(1, expected_count + 1))
            if fold_numbers != expected_numbers:
                raise RuntimeError(
                    f"Fold identities for repeat/seed {key} are {fold_numbers}; "
                    f"expected {expected_numbers}."
                )

    if settings.require_each_sample_once_per_repeat:
        for (repeat, seed), specs in groups.items():
            test_counts = np.zeros(n_samples, dtype=np.int64)
            for spec in specs:
                test_idx = _get_indices(spec, "outer_test_indices", n_samples)
                test_counts[test_idx] += 1
            if not np.all(test_counts == 1):
                raise RuntimeError(
                    f"Outer-test coverage must be exactly once per sample "
                    f"for repeat={repeat}, cv_seed={seed}."
                )


# ---------------------------------------------------------------------------
# 9. Seed derivation and checkpointing
# ---------------------------------------------------------------------------

def derive_stage_seeds(
    spec: FoldSpecLike,
    settings: TrainingSettings,
) -> Tuple[int, int]:
    """Use the original runner's selection/refit seed convention by default."""

    base = (
        int(spec.cv_seed) * int(settings.cv_seed_multiplier)
        + int(spec.fold) * int(settings.fold_seed_multiplier)
    )
    return (
        int(base + settings.selection_seed_offset),
        int(base + settings.refit_seed_offset),
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unnamed"


def _checkpoint_path(
    settings: TrainingSettings,
    trait: str,
    spec: FoldSpecLike,
    stage: str,
) -> Optional[Path]:
    if settings.output_dir is None:
        return None
    return (
        Path(settings.output_dir)
        / "checkpoints"
        / _safe_name(trait)
        / f"repeat_{int(spec.repeat)}_seed_{int(spec.cv_seed)}_"
          f"fold_{int(spec.fold)}_{stage}.pt"
    )


def _save_checkpoint(
    model: DVAGraphGP,
    path: Optional[Path],
    metadata: Dict[str, Any],
    settings: TrainingSettings,
) -> Optional[str]:
    if path is None:
        raise ValueError("output_dir is required when checkpoint saving is enabled.")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not settings.overwrite_existing:
        raise FileExistsError(
            f"Checkpoint exists and overwrite_existing=False: {path}"
        )

    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "input_dim": int(model.input_dim),
            "metadata": metadata,
        },
        path,
    )
    return str(path)


# ---------------------------------------------------------------------------
# 10. Inner training and epoch selection
# ---------------------------------------------------------------------------

def select_epoch_on_locked_inner_validation(
    X: np.ndarray,
    y: np.ndarray,
    trait: str,
    spec: FoldSpecLike,
    selection_seed: int,
    *,
    settings: TrainingSettings,
    logger: logging.Logger,
) -> InnerSelectionResult:
    """
    Train on inner-training rows and select epoch by inner-validation MAE.

    After every epoch, the validation graph is rebuilt from posterior means
    of inner-training genotypes only. Validation genotypes and labels are
    excluded from graph construction. The selected model weights are not
    transferred to the outer refit; only the selected epoch count is retained.
    """

    set_global_seed(selection_seed, settings)

    inner_train_idx = _get_indices(spec, "inner_train_indices", len(X))
    inner_val_idx = _get_indices(spec, "inner_validation_indices", len(X))

    X_inner_train = X[inner_train_idx]
    y_inner_train = y[inner_train_idx]
    X_inner_val = X[inner_val_idx]
    y_inner_val = y[inner_val_idx]

    train_loader = make_data_loader(
        X_inner_train,
        y_inner_train,
        settings=settings,
        seed=selection_seed,
        shuffle=True,
    )

    model = DVAGraphGP(input_dim=X.shape[1]).to(settings.resolved_device)
    optimizer = make_optimizer(model, settings)

    best_epoch = 0
    best_mae = float("inf")
    best_mse = float("inf")
    stale_epochs = 0
    history_rows: List[Dict[str, Any]] = []
    best_checkpoint_path: Optional[str] = None

    logger.info(
        "Inner selection | trait=%s | repeat=%d | cv_seed=%d | fold=%d | "
        "n_inner_train=%d | n_inner_validation=%d",
        trait, spec.repeat, spec.cv_seed, spec.fold,
        len(inner_train_idx), len(inner_val_idx),
    )

    for epoch in range(1, int(settings.max_epochs) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, settings)

        # One frozen graph from all inner-training posterior means, never val rows.
        freeze_training_partition_graph(model, X_inner_train, settings)
        val_pred = predict_partition(model, X_inner_val, settings=settings)
        val_mae, val_mse = calculate_original_scale_errors(y_inner_val, val_pred)

        improved = (
            val_mae < best_mae - float(settings.early_stopping_min_delta)
        )

        if improved:
            best_epoch = int(epoch)
            best_mae = float(val_mae)
            best_mse = float(val_mse)
            stale_epochs = 0

            if settings.save_fold_checkpoints:
                best_checkpoint_path = _save_checkpoint(
                    model,
                    _checkpoint_path(settings, trait, spec, "inner_best"),
                    {
                        "dataset": settings.dataset_name,
                        "method": settings.method_name,
                        "trait": trait,
                        "stage": "inner_epoch_selection",
                        "repeat": int(spec.repeat),
                        "cv_seed": int(spec.cv_seed),
                        "fold": int(spec.fold),
                        "selection_seed": int(selection_seed),
                        "selected_epoch": int(best_epoch),
                        "validation_mae_original": float(best_mae),
                        "validation_mse_original": float(best_mse),
                        "graph_source": "inner_training_posterior_means_only",
                    },
                    settings,
                )
        else:
            stale_epochs += 1

        history_rows.append({
            "dataset": settings.dataset_name,
            "method": settings.method_name,
            "trait": trait,
            "repeat": int(spec.repeat),
            "cv_split_seed": int(spec.cv_seed),
            "fold": int(spec.fold),
            "inner_split_seed": int(spec.inner_split_seed),
            "selection_seed": int(selection_seed),
            "stage": "inner_epoch_selection",
            "epoch": int(epoch),
            "validation_MAE_original": float(val_mae),
            "validation_MSE_original": float(val_mse),
            "is_new_best": bool(improved),
            "is_selected_epoch": False,
            "epochs_without_improvement": int(stale_epochs),
            "early_stop_triggered_after_epoch": False,
            **train_metrics,
        })

        if epoch == 1 or epoch % 10 == 0 or improved:
            logger.info(
                "Inner | %s | repeat=%d fold=%d | epoch=%03d | "
                "val_MAE=%.6f | best_epoch=%d | best_MAE=%.6f",
                trait, spec.repeat, spec.fold, epoch, val_mae, best_epoch, best_mae,
            )

        if (
            epoch >= int(settings.min_epochs_before_stopping)
            and stale_epochs >= int(settings.early_stopping_patience)
        ):
            history_rows[-1]["early_stop_triggered_after_epoch"] = True
            logger.info(
                "Early stopping | %s | repeat=%d fold=%d | epoch=%d",
                trait, spec.repeat, spec.fold, epoch,
            )
            break

    if best_epoch < 1 or not np.isfinite(best_mae):
        raise RuntimeError(
            f"No valid epoch selected for trait={trait}, "
            f"repeat={spec.repeat}, fold={spec.fold}."
        )

    history = pd.DataFrame(history_rows)
    history["is_selected_epoch"] = history["epoch"].eq(best_epoch)

    result = InnerSelectionResult(
        selected_epoch=int(best_epoch),
        best_validation_mae_original=float(best_mae),
        best_validation_mse_original=float(best_mse),
        epochs_completed=int(len(history)),
        selection_seed=int(selection_seed),
        history=history,
        checkpoint_path=best_checkpoint_path,
    )

    del model, optimizer, train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# 11. Fresh outer-training refit and held-out prediction
# ---------------------------------------------------------------------------

def refit_on_outer_train_and_predict(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    trait: str,
    spec: FoldSpecLike,
    selected_epoch: int,
    refit_seed: int,
    *,
    settings: TrainingSettings,
    logger: logging.Logger,
) -> OuterRefitResult:
    """
    Freshly initialize and fit on outer-training rows, then predict outer-test.

    Outer-test phenotypes are not sliced/read until after prediction is complete.
    The final frozen graph is constructed from outer-training posterior means.
    """

    if selected_epoch < 1:
        raise ValueError("selected_epoch must be >= 1.")

    set_global_seed(refit_seed, settings)

    outer_train_idx = _get_indices(spec, "outer_train_indices", len(X))
    outer_test_idx = _get_indices(spec, "outer_test_indices", len(X))

    X_outer_train = X[outer_train_idx]
    y_outer_train = y[outer_train_idx]
    X_outer_test = X[outer_test_idx]

    train_loader = make_data_loader(
        X_outer_train,
        y_outer_train,
        settings=settings,
        seed=refit_seed,
        shuffle=True,
    )

    # A new model is initialized; inner-selection weights are not reused.
    model = DVAGraphGP(input_dim=X.shape[1]).to(settings.resolved_device)
    optimizer = make_optimizer(model, settings)

    parameter_count = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    history_rows: List[Dict[str, Any]] = []

    logger.info(
        "Outer refit | trait=%s | repeat=%d | cv_seed=%d | fold=%d | "
        "n_outer_train=%d | selected_epoch=%d",
        trait, spec.repeat, spec.cv_seed, spec.fold,
        len(outer_train_idx), selected_epoch,
    )

    for epoch in range(1, int(selected_epoch) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, settings)
        history_rows.append({
            "dataset": settings.dataset_name,
            "method": settings.method_name,
            "trait": trait,
            "repeat": int(spec.repeat),
            "cv_split_seed": int(spec.cv_seed),
            "fold": int(spec.fold),
            "inner_split_seed": int(spec.inner_split_seed),
            "refit_seed": int(refit_seed),
            "stage": "final_outer_train_refit",
            "epoch": int(epoch),
            "selected_epoch_for_refit": int(selected_epoch),
            **train_metrics,
        })

        if epoch == 1 or epoch % 10 == 0 or epoch == selected_epoch:
            logger.info(
                "Refit | %s | repeat=%d fold=%d | epoch=%03d/%03d | loss=%.6f",
                trait, spec.repeat, spec.fold, epoch, selected_epoch,
                train_metrics["total_loss"],
            )

    # Final graph uses all outer-training posterior means only.
    final_adjacency = freeze_training_partition_graph(
        model, X_outer_train, settings
    )

    # No outer-test target has been supplied to prediction or graph construction.
    y_pred = predict_partition(
        model,
        X_outer_test,
        settings=settings,
    )

    # Access held-out labels only after the prediction has been completed.
    y_true = np.asarray(y[outer_test_idx], dtype=np.float64).copy()

    if y_true.shape != y_pred.shape:
        raise RuntimeError(
            f"Outer-test shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}."
        )
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise FloatingPointError("Non-finite outer-test values.")

    checkpoint_path: Optional[str] = None
    if settings.save_fold_checkpoints:
        checkpoint_path = _save_checkpoint(
            model,
            _checkpoint_path(settings, trait, spec, "outer_final"),
            {
                "dataset": settings.dataset_name,
                "method": settings.method_name,
                "trait": trait,
                "stage": "final_outer_train_refit",
                "repeat": int(spec.repeat),
                "cv_seed": int(spec.cv_seed),
                "fold": int(spec.fold),
                "refit_seed": int(refit_seed),
                "selected_epoch": int(selected_epoch),
                "input_dim": int(X.shape[1]),
                "graph_source": "outer_training_posterior_means_only",
                "genotype_preprocessing_in_train": "none",
                "phenotype_scaling_in_train": "none",
            },
            settings,
        )

    result = OuterRefitResult(
        y_true_original=y_true,
        y_pred_original=y_pred,
        test_indices=outer_test_idx.copy(),
        test_sample_ids=np.asarray(sample_ids)[outer_test_idx].astype(str),
        history=pd.DataFrame(history_rows),
        final_adjacency=final_adjacency,
        trainable_parameters=parameter_count,
        refit_seed=int(refit_seed),
        checkpoint_path=checkpoint_path,
    )

    del model, optimizer, train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# 12. One fold and complete multi-trait CV
# ---------------------------------------------------------------------------

def run_one_fold(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    trait: str,
    spec: FoldSpecLike,
    *,
    settings: TrainingSettings,
    logger: logging.Logger,
) -> FoldTrainingResult:
    """Run inner epoch selection and independent outer refitting for one fold."""

    partition_audit = validate_one_fold_spec(spec, len(X), settings)
    selection_seed, refit_seed = derive_stage_seeds(spec, settings)

    selection = select_epoch_on_locked_inner_validation(
        X, y, trait, spec, selection_seed,
        settings=settings,
        logger=logger,
    )

    refit = refit_on_outer_train_and_predict(
        X, y, sample_ids, trait, spec,
        selection.selected_epoch,
        refit_seed,
        settings=settings,
        logger=logger,
    )

    return FoldTrainingResult(
        dataset=settings.dataset_name,
        method=settings.method_name,
        trait=trait,
        repeat=int(spec.repeat),
        cv_seed=int(spec.cv_seed),
        fold=int(spec.fold),
        inner_split_seed=int(spec.inner_split_seed),
        selection_seed=int(selection_seed),
        refit_seed=int(refit_seed),
        selected_epoch=int(selection.selected_epoch),
        inner_epochs_completed=int(selection.epochs_completed),
        best_inner_validation_mae_original=float(
            selection.best_validation_mae_original
        ),
        best_inner_validation_mse_original=float(
            selection.best_validation_mse_original
        ),
        inner_history=selection.history,
        outer_history=refit.history,
        test_indices=refit.test_indices,
        test_sample_ids=refit.test_sample_ids,
        y_true_original=refit.y_true_original,
        y_pred_original=refit.y_pred_original,
        final_adjacency=refit.final_adjacency,
        trainable_parameters=refit.trainable_parameters,
        partition_audit=partition_audit,
        inner_checkpoint_path=selection.checkpoint_path,
        outer_checkpoint_path=refit.checkpoint_path,
    )


def run_cross_validation(
    X: np.ndarray,
    sample_ids: Optional[Sequence[Any]],
    phenotype_data: Mapping[str, np.ndarray],
    fold_specs: Sequence[FoldSpecLike],
    *,
    traits: Optional[Sequence[str]] = None,
    settings: Optional[TrainingSettings] = None,
    logger: Optional[logging.Logger] = None,
) -> List[FoldTrainingResult]:
    """
    Run the same training engine across any compatible dataset and CV protocol.

    Parameters
    ----------
    X:
        Already-prepared genotype/features matrix [n_samples, n_features].
        No preprocessing is applied here.
    sample_ids:
        IDs aligned row-for-row with X. If None, integer row IDs are generated.
    phenotype_data:
        Mapping from trait name to original-scale target vector.
    fold_specs:
        Fold specifications supplied by data.py. This function never generates
        or changes train/validation/test partitions.
    traits:
        Traits to run. Defaults to all keys in phenotype_data.
    settings:
        Experiment-specific TrainingSettings, normally created from config.py.
    logger:
        Optional experiment logger.

    Returns
    -------
    List[FoldTrainingResult]
        One result for every requested trait x supplied fold specification.
    """

    settings = settings or TrainingSettings()
    validate_training_settings(settings)
    validate_model_contract()

    X_array, _ = _validate_xy(X, name="complete prepared feature matrix")
    n_samples = X_array.shape[0]

    if sample_ids is None:
        sample_id_array = np.arange(n_samples).astype(str)
    else:
        sample_id_array = np.asarray(sample_ids).astype(str)
        if sample_id_array.ndim != 1 or len(sample_id_array) != n_samples:
            raise ValueError(
                "sample_ids must be a 1D array aligned with the feature matrix."
            )
        if len(np.unique(sample_id_array)) != len(sample_id_array):
            raise ValueError("sample_ids must be unique.")

    if traits is None:
        traits = list(phenotype_data.keys())
    else:
        traits = list(traits)

    if not traits:
        raise ValueError("No traits were supplied.")

    missing = [trait for trait in traits if trait not in phenotype_data]
    if missing:
        raise KeyError(f"Missing phenotype vectors for traits: {missing}")

    checked_y: Dict[str, np.ndarray] = {}
    for trait in traits:
        y = np.asarray(phenotype_data[trait], dtype=np.float64).reshape(-1)
        if y.shape[0] != n_samples:
            raise ValueError(
                f"Trait {trait!r} has {len(y)} targets; expected {n_samples}."
            )
        if not np.isfinite(y).all():
            raise ValueError(
                f"Trait {trait!r} has missing/non-finite values. "
                "train.py does not impute or remove samples."
            )
        checked_y[trait] = y

    validate_fold_collection(
        fold_specs,
        n_samples=n_samples,
        settings=settings,
    )

    if logger is None:
        logger = logging.getLogger("DVAGraph-GP.train")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    ordered_specs = sorted(
        fold_specs,
        key=lambda s: (int(s.repeat), int(s.cv_seed), int(s.fold)),
    )

    logger.info("=" * 76)
    logger.info("DVAGraph-GP reusable training engine")
    logger.info("Dataset: %s | Method: %s", settings.dataset_name, settings.method_name)
    logger.info("Prepared feature matrix: %s", X_array.shape)
    logger.info("Traits: %s", list(traits))
    logger.info("CV fold specifications supplied by data.py: %d", len(ordered_specs))
    logger.info("Device: %s", settings.resolved_device)
    logger.info("Batch size: %d | Max epochs: %d", settings.batch_size, settings.max_epochs)
    logger.info("Optimizer: %s | Learning rate: %g", settings.optimizer_name, settings.learning_rate)
    logger.info("Genotype preprocessing in train.py: NONE")
    logger.info("Phenotype scaling in train.py: NONE")
    logger.info("Graph EMA: NONE")
    logger.info("=" * 76)

    all_results: List[FoldTrainingResult] = []

    for trait in traits:
        logger.info("")
        logger.info("#" * 76)
        logger.info("Starting trait: %s", trait)
        logger.info("#" * 76)

        for spec in ordered_specs:
            result = run_one_fold(
                X_array,
                checked_y[trait],
                sample_id_array,
                trait,
                spec,
                settings=settings,
                logger=logger,
            )
            all_results.append(result)

            logger.info(
                "Fold complete | trait=%s | repeat=%d | cv_seed=%d | fold=%d | "
                "selected_epoch=%d | inner_val_MAE=%.6f | n_test=%d",
                trait,
                result.repeat,
                result.cv_seed,
                result.fold,
                result.selected_epoch,
                result.best_inner_validation_mae_original,
                len(result.y_true_original),
            )

    expected_results = len(traits) * len(ordered_specs)
    if len(all_results) != expected_results:
        raise RuntimeError(
            f"Expected {expected_results} fold results, received {len(all_results)}."
        )

    logger.info(
        "Training complete: %d trait-fold evaluations.",
        len(all_results),
    )
    return all_results


__all__ = [
    "FoldSpecLike",
    "TrainingSettings",
    "InnerSelectionResult",
    "OuterRefitResult",
    "FoldTrainingResult",
    "validate_training_settings",
    "validate_model_contract",
    "set_global_seed",
    "make_data_loader",
    "make_optimizer",
    "train_one_epoch",
    "freeze_training_partition_graph",
    "predict_partition",
    "calculate_original_scale_errors",
    "validate_one_fold_spec",
    "validate_fold_collection",
    "derive_stage_seeds",
    "select_epoch_on_locked_inner_validation",
    "refit_on_outer_train_and_predict",
    "run_one_fold",
    "run_cross_validation",
]
