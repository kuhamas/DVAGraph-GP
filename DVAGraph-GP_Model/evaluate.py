"""
DVAGraph-GP Evaluation and Reporting
=====================================

This module evaluates outputs produced by train.py. It does not train models,
modify predictions, alter CV partitions, or fit any preprocessing operation.

Primary fold-level metrics
--------------------------
PCC  : Pearson correlation coefficient
R2   : coefficient of determination
MSE  : mean squared error
MAE  : mean absolute error
RMSE : square root of MSE (retained to match the original runner)

All metrics are calculated from the held-out y_true_original and
y_pred_original values returned by train.py. They are therefore reported in
the phenotype units supplied to the training pipeline.

Primary reporting level
-----------------------
For repeated K-fold evaluation, the primary performance summary is the
mean +/- sample SD of the fold-level metrics across outer folds, matching
the original DVAGraph-GP runner.

Repeated out-of-fold (OOF) predictions are also saved. Predictions for the
same sample are averaged across CV runs only for a supplementary
repeat-averaged OOF analysis. This supplementary analysis does not replace
the fold-level performance summary.

Expected input
--------------
A sequence of FoldTrainingResult objects returned by:
    train.run_cross_validation(...)

Expected output
---------------
For each dataset/method/trait:
    fold_metrics.csv
    summary_metrics.csv
    summary_by_repeat.csv
    oof_predictions.csv
    final_repeated_oof_predictions.csv
    repeated_oof_metrics.csv
    metric_audit.csv
    performance_validity_audit.csv
    epoch_losses.csv
    inner_validation_selection.csv
    inner_selection_summary.csv
    loss_summary.csv
    graph_diagnostics/
    figures/
    evaluation_manifest.json
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

PRIMARY_METRICS = (
    "orig_PCC",
    "orig_R2",
    "orig_MSE",
    "orig_MAE",
)

# RMSE is retained because it was also reported by the original runner.
ALL_METRICS = (
    "orig_PCC",
    "orig_R2",
    "orig_RMSE",
    "orig_MSE",
    "orig_MAE",
)

PREDICTION_COLUMNS = (
    "dataset",
    "method",
    "trait",
    "repeat",
    "cv_split_seed",
    "fold",
    "inner_split_seed",
    "selection_seed",
    "refit_seed",
    "selected_epoch",
    "sample_id",
    "sample_index_after_y_filter",
    "y_true_original",
    "y_pred_original",
    "residual_original",
)

DEFAULT_CONSTANT_EPS = 1e-12


# ---------------------------------------------------------------------------
# 2. Logging and safe file output
# ---------------------------------------------------------------------------

def get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """Return the supplied logger or create a module logger."""

    if logger is not None:
        return logger

    result = logging.getLogger("DVAGraph-GP.evaluate")
    if not result.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        result.addHandler(handler)
    result.setLevel(logging.INFO)
    return result


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write CSV atomically to reduce the risk of partial result files."""

    ensure_directory(path.parent)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write a JSON file atomically, converting NumPy/Pandas values safely."""

    ensure_directory(path.parent)

    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (Path,)):
            return str(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        if pd.isna(value) if np.isscalar(value) else False:
            return None
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            dict(payload),
            handle,
            indent=2,
            ensure_ascii=False,
            default=convert,
            allow_nan=False,
        )
    os.replace(temporary_path, path)


# ---------------------------------------------------------------------------
# 3. Metric calculation: consistent with the original runner
# ---------------------------------------------------------------------------

def safe_corr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    constant_epsilon: float = DEFAULT_CONSTANT_EPS,
) -> float:
    """
    Calculate Pearson's r safely.

    As in the original runner, PCC is undefined when fewer than three pairs
    are available or either vector is effectively constant.
    """

    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")

    if truth.size < 3:
        return float("nan")

    if (
        np.std(truth) < constant_epsilon
        or np.std(prediction) < constant_epsilon
    ):
        return float("nan")

    return float(pearsonr(truth, prediction)[0])


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "orig_",
    *,
    constant_epsilon: float = DEFAULT_CONSTANT_EPS,
) -> Dict[str, Any]:
    """
    Calculate PCC, R2, RMSE, MSE, and MAE.

    Unlike a permissive metric function, this implementation does not silently
    drop missing or infinite predictions. Such values indicate a failed model
    evaluation and must be investigated rather than hidden.
    """

    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if truth.shape != prediction.shape:
        raise ValueError(
            f"Metric input shape mismatch: {truth.shape} vs {prediction.shape}."
        )

    if truth.size == 0:
        raise ValueError("Cannot calculate metrics for an empty test partition.")

    if not np.isfinite(truth).all():
        raise ValueError("y_true contains NaN or infinite values.")

    if not np.isfinite(prediction).all():
        raise ValueError("y_pred contains NaN or infinite values.")

    if truth.size < 2:
        r2 = float("nan")
        mse = float("nan")
        mae = float("nan")
        rmse = float("nan")
    else:
        mse = float(mean_squared_error(truth, prediction))
        mae = float(mean_absolute_error(truth, prediction))
        rmse = float(math.sqrt(mse))
        # Matches sklearn's default behavior in the original runner.
        r2 = float(r2_score(truth, prediction))

    pcc = safe_corr(
        truth,
        prediction,
        constant_epsilon=constant_epsilon,
    )

    return {
        prefix + "PCC": pcc,
        prefix + "R2": r2,
        prefix + "RMSE": rmse,
        prefix + "MSE": mse,
        prefix + "MAE": mae,
        prefix + "MAE2_le_MSE": bool(
            np.isfinite(mse)
            and np.isfinite(mae)
            and mae ** 2 <= mse + 1e-10
        ),
    }


def calibration_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "orig_",
    *,
    constant_epsilon: float = DEFAULT_CONSTANT_EPS,
) -> Dict[str, float]:
    """
    Provide descriptive calibration diagnostics.

    Slope/intercept describe prediction as a linear function of observed values.
    They are supplementary diagnostics, not additional primary performance
    metrics and are not used to select models.
    """

    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if truth.shape != prediction.shape:
        raise ValueError("Calibration input shapes do not match.")

    if truth.size == 0:
        raise ValueError("Calibration diagnostics require at least one pair.")

    observed_mean = float(np.mean(truth))
    predicted_mean = float(np.mean(prediction))
    observed_sd = float(np.std(truth, ddof=1)) if truth.size > 1 else float("nan")
    predicted_sd = (
        float(np.std(prediction, ddof=1)) if prediction.size > 1 else float("nan")
    )

    if np.isfinite(observed_sd) and observed_sd >= constant_epsilon:
        slope = float(
            np.sum((truth - observed_mean) * (prediction - predicted_mean))
            / np.sum((truth - observed_mean) ** 2)
        )
        intercept = float(predicted_mean - slope * observed_mean)
    else:
        slope = float("nan")
        intercept = float("nan")

    if (
        np.isfinite(observed_sd)
        and observed_sd >= constant_epsilon
        and np.isfinite(predicted_sd)
    ):
        sd_ratio = float(predicted_sd / observed_sd)
    else:
        sd_ratio = float("nan")

    return {
        prefix + "observed_mean": observed_mean,
        prefix + "predicted_mean": predicted_mean,
        prefix + "observed_sd": observed_sd,
        prefix + "predicted_sd": predicted_sd,
        prefix + "prediction_sd_ratio": sd_ratio,
        prefix + "mean_bias_pred_minus_obs": float(
            predicted_mean - observed_mean
        ),
        prefix + "calibration_slope_pred_on_obs": slope,
        prefix + "calibration_intercept": intercept,
    }


# ---------------------------------------------------------------------------
# 4. Summary tables
# ---------------------------------------------------------------------------

def mean_sd_table(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Produce long-format mean/SD/median/min/max summaries.

    This matches the original runner's summary_metrics.csv organization.
    SD uses ddof=1 (sample standard deviation).
    """

    missing_groups = [c for c in group_columns if c not in frame.columns]
    if missing_groups:
        raise KeyError(f"Missing summary grouping columns: {missing_groups}")

    missing_metrics = [c for c in metric_columns if c not in frame.columns]
    if missing_metrics:
        raise KeyError(f"Missing summary metric columns: {missing_metrics}")

    rows: List[Dict[str, Any]] = []

    grouped = frame.groupby(
        list(group_columns),
        dropna=False,
        sort=True,
    )

    for key, subset in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        fixed = dict(zip(group_columns, key))

        for metric in metric_columns:
            values = pd.to_numeric(
                subset[metric],
                errors="coerce",
            ).dropna().to_numpy(dtype=np.float64)

            mean = float(np.mean(values)) if values.size else float("nan")
            sd = (
                float(np.std(values, ddof=1))
                if values.size > 1
                else float("nan")
            )

            rows.append({
                **fixed,
                "metric": metric,
                "mean": mean,
                "sd": sd,
                "median": (
                    float(np.median(values)) if values.size else float("nan")
                ),
                "minimum": (
                    float(np.min(values)) if values.size else float("nan")
                ),
                "maximum": (
                    float(np.max(values)) if values.size else float("nan")
                ),
                "n_valid": int(values.size),
                "formatted": (
                    f"{mean:.4f} ± {sd:.4f}"
                    if values.size > 1
                    else (f"{mean:.4f}" if values.size == 1 else "NA")
                ),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Convert train.py results into a consistent prediction table
# ---------------------------------------------------------------------------

def _result_value(result: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from a result object with a safe default."""

    return getattr(result, name, default)


def _prediction_frame_from_result(result: Any) -> pd.DataFrame:
    """Build a standardized prediction frame from one FoldTrainingResult."""

    required = (
        "dataset",
        "method",
        "trait",
        "repeat",
        "cv_seed",
        "fold",
        "inner_split_seed",
        "selection_seed",
        "refit_seed",
        "selected_epoch",
        "test_indices",
        "test_sample_ids",
        "y_true_original",
        "y_pred_original",
    )

    missing = [
        name for name in required
        if not hasattr(result, name)
    ]
    if missing:
        raise TypeError(
            "A fold result is incompatible with evaluate.py. Missing fields: "
            + ", ".join(missing)
        )

    indices = np.asarray(result.test_indices, dtype=np.int64).reshape(-1)
    sample_ids = np.asarray(result.test_sample_ids).astype(str).reshape(-1)
    truth = np.asarray(result.y_true_original, dtype=np.float64).reshape(-1)
    prediction = np.asarray(result.y_pred_original, dtype=np.float64).reshape(-1)

    lengths = {
        len(indices),
        len(sample_ids),
        len(truth),
        len(prediction),
    }
    if len(lengths) != 1:
        raise ValueError(
            "Fold result has inconsistent lengths for test indices, sample IDs, "
            "observations, and predictions."
        )

    if len(indices) == 0:
        raise ValueError("Fold result contains no held-out predictions.")

    if np.unique(indices).size != indices.size:
        raise ValueError(
            f"Duplicate test indices in repeat={result.repeat}, fold={result.fold}."
        )

    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise FloatingPointError(
            f"Non-finite held-out values in trait={result.trait}, "
            f"repeat={result.repeat}, fold={result.fold}."
        )

    frame = pd.DataFrame({
        "dataset": str(result.dataset),
        "method": str(result.method),
        "trait": str(result.trait),
        "repeat": int(result.repeat),
        "cv_split_seed": int(result.cv_seed),
        "fold": int(result.fold),
        "inner_split_seed": int(result.inner_split_seed),
        "selection_seed": int(result.selection_seed),
        "refit_seed": int(result.refit_seed),
        "selected_epoch": int(result.selected_epoch),
        "sample_id": sample_ids,
        "sample_index_after_y_filter": indices,
        "y_true_original": truth,
        "y_pred_original": prediction,
        "residual_original": truth - prediction,
    })

    return frame.loc[:, PREDICTION_COLUMNS]


def _validate_prediction_identity(predictions: pd.DataFrame) -> None:
    """Ensure sample IDs and row indices define a one-to-one identity mapping."""

    index_to_ids = predictions.groupby(
        "sample_index_after_y_filter"
    )["sample_id"].nunique()

    ids_to_indices = predictions.groupby(
        "sample_id"
    )["sample_index_after_y_filter"].nunique()

    if (index_to_ids > 1).any():
        raise RuntimeError(
            "A sample index maps to multiple sample IDs across fold results."
        )

    if (ids_to_indices > 1).any():
        raise RuntimeError(
            "A sample ID maps to multiple sample indices across fold results."
        )


def _validate_truth_consistency(
    predictions: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> None:
    """The same sample must have the same observed phenotype across repeats."""

    truth_range = predictions.groupby(
        ["sample_id", "sample_index_after_y_filter"]
    )["y_true_original"].agg(lambda values: float(values.max() - values.min()))

    if (truth_range > tolerance).any():
        bad = truth_range[truth_range > tolerance].head(10)
        raise RuntimeError(
            "Observed phenotype values differ for the same sample across CV runs. "
            f"Examples:\n{bad.to_string()}"
        )


# ---------------------------------------------------------------------------
# 6. Fold-level metrics and audit
# ---------------------------------------------------------------------------

def build_fold_metrics(
    fold_results: Sequence[Any],
    prediction_frames: Mapping[Tuple[str, str, str, int, int, int], pd.DataFrame],
    *,
    constant_epsilon: float = DEFAULT_CONSTANT_EPS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate fold-level metrics and one audit record per fold."""

    metric_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for result in fold_results:
        key = (
            str(result.dataset),
            str(result.method),
            str(result.trait),
            int(result.repeat),
            int(result.cv_seed),
            int(result.fold),
        )
        pred = prediction_frames[key]

        truth = pred["y_true_original"].to_numpy(dtype=np.float64)
        prediction = pred["y_pred_original"].to_numpy(dtype=np.float64)

        metrics = compute_metrics(
            truth,
            prediction,
            prefix="orig_",
            constant_epsilon=constant_epsilon,
        )
        calibration = calibration_diagnostics(
            truth,
            prediction,
            prefix="orig_",
            constant_epsilon=constant_epsilon,
        )

        prediction_sd = (
            float(np.std(prediction, ddof=1))
            if prediction.size > 1
            else 0.0
        )
        predictions_constant = bool(
            not np.isfinite(prediction_sd)
            or prediction_sd < constant_epsilon
        )

        partition_audit = _result_value(
            result,
            "partition_audit",
            {},
        )
        if partition_audit is None:
            partition_audit = {}

        partition_audit_pass = (
            bool(all(bool(v) for v in partition_audit.values()))
            if partition_audit
            else False
        )

        expected_n = len(np.asarray(result.test_indices).reshape(-1))
        predictions_present = bool(
            len(prediction) == expected_n
            and np.isfinite(prediction).all()
            and np.isfinite(truth).all()
        )

        row = {
            "dataset": str(result.dataset),
            "trait": str(result.trait),
            "method": str(result.method),
            "repeat": int(result.repeat),
            "cv_split_seed": int(result.cv_seed),
            "fold": int(result.fold),
            "inner_split_seed": int(result.inner_split_seed),
            "selection_seed": int(result.selection_seed),
            "refit_seed": int(result.refit_seed),
            "selected_epoch": int(result.selected_epoch),
            "inner_epochs_completed": int(result.inner_epochs_completed),
            "inner_validation_selected_MAE_original": float(
                result.best_inner_validation_mae_original
            ),
            "inner_validation_selected_MSE_original": float(
                result.best_inner_validation_mse_original
            ),
            "n_test": int(len(prediction)),
            "trainable_parameters": int(result.trainable_parameters),
            "prediction_unique_count": int(np.unique(prediction).size),
            "predictions_effectively_constant": predictions_constant,
            "orig_pcc_is_finite": bool(np.isfinite(metrics["orig_PCC"])),
            "metrics_calculated_on_original_scale": True,
            "all_test_predictions_present": predictions_present,
            "partition_audit_pass": partition_audit_pass,
            **metrics,
            **calibration,
        }
        metric_rows.append(row)

        audit = {
            "dataset": str(result.dataset),
            "trait": str(result.trait),
            "method": str(result.method),
            "repeat": int(result.repeat),
            "cv_split_seed": int(result.cv_seed),
            "fold": int(result.fold),
            "no_duplicate_test_indices_within_fold": bool(
                pred["sample_index_after_y_filter"].is_unique
            ),
            "all_test_predictions_present": predictions_present,
            "metrics_calculated_on_original_scale": True,
            "orig_PCC_is_finite": bool(np.isfinite(metrics["orig_PCC"])),
            "prediction_nonconstant": not predictions_constant,
            "orig_MAE2_le_MSE": bool(metrics["orig_MAE2_le_MSE"]),
            "partition_audit_pass": partition_audit_pass,
            "selected_epoch_valid": bool(
                int(result.selected_epoch) >= 1
                and int(result.inner_epochs_completed) >= int(result.selected_epoch)
            ),
        }
        audit["pass"] = bool(
            audit["no_duplicate_test_indices_within_fold"]
            and audit["all_test_predictions_present"]
            and audit["metrics_calculated_on_original_scale"]
            and audit["orig_PCC_is_finite"]
            and audit["prediction_nonconstant"]
            and audit["orig_MAE2_le_MSE"]
            and audit["partition_audit_pass"]
            and audit["selected_epoch_valid"]
        )
        audit_rows.append(audit)

    fold_metrics = pd.DataFrame(metric_rows).sort_values(
        ["dataset", "trait", "method", "repeat", "cv_split_seed", "fold"]
    ).reset_index(drop=True)

    metric_audit = pd.DataFrame(audit_rows).sort_values(
        ["dataset", "trait", "method", "repeat", "cv_split_seed", "fold"]
    ).reset_index(drop=True)

    return fold_metrics, metric_audit


# ---------------------------------------------------------------------------
# 7. Repeated out-of-fold predictions
# ---------------------------------------------------------------------------

def build_repeated_oof_predictions(
    predictions: pd.DataFrame,
    *,
    require_complete_repeated_oof: bool,
    expected_folds_per_repeat: Optional[int],
) -> pd.DataFrame:
    """
    Average each sample's held-out predictions across repeated CV runs.

    A CV run is identified by (repeat, cv_split_seed), not by fold number.
    For repeated K-fold, every sample must appear exactly once in each run.
    """

    frame = predictions.copy()
    frame["cv_run_id"] = (
        frame["repeat"].astype(str)
        + "::"
        + frame["cv_split_seed"].astype(str)
    )

    _validate_prediction_identity(frame)
    _validate_truth_consistency(frame)

    run_keys = (
        frame[["repeat", "cv_split_seed", "cv_run_id"]]
        .drop_duplicates()
        .sort_values(["repeat", "cv_split_seed"])
    )

    if expected_folds_per_repeat is not None:
        observed_fold_counts = (
            frame.drop_duplicates(
                ["repeat", "cv_split_seed", "fold"]
            )
            .groupby(["repeat", "cv_split_seed"])
            .size()
        )
        wrong = observed_fold_counts[
            observed_fold_counts != int(expected_folds_per_repeat)
        ]
        if not wrong.empty:
            raise RuntimeError(
                "Observed fold count differs from the configured protocol: "
                f"{wrong.to_dict()}"
            )

    if require_complete_repeated_oof:
        all_sample_indices = set(
            frame["sample_index_after_y_filter"].astype(int).unique().tolist()
        )

        for run_id in run_keys["cv_run_id"]:
            run = frame.loc[frame["cv_run_id"] == run_id]
            counts = run.groupby(
                "sample_index_after_y_filter"
            ).size()

            if (counts != 1).any():
                raise RuntimeError(
                    f"Each sample must appear exactly once in repeated K-fold "
                    f"run {run_id}; duplicate/missing appearances detected."
                )

            run_sample_indices = set(
                run["sample_index_after_y_filter"].astype(int).tolist()
            )
            if run_sample_indices != all_sample_indices:
                raise RuntimeError(
                    f"Run {run_id} does not cover the same complete sample set "
                    "as the other CV runs."
                )

    final_oof = frame.groupby(
        ["sample_id", "sample_index_after_y_filter"],
        as_index=False,
        sort=True,
    ).agg(
        y_true_original=("y_true_original", "first"),
        y_pred_original_mean_across_repeats=(
            "y_pred_original",
            "mean",
        ),
        y_pred_original_sd_across_repeats=(
            "y_pred_original",
            "std",
        ),
        n_predictions=("y_pred_original", "size"),
        n_cv_runs=("cv_run_id", "nunique"),
    )

    final_oof["n_repeats"] = final_oof["n_cv_runs"]
    final_oof["residual_original"] = (
        final_oof["y_true_original"]
        - final_oof["y_pred_original_mean_across_repeats"]
    )

    return final_oof.sort_values(
        "sample_index_after_y_filter"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. Graph diagnostics
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between flattened graph vectors."""

    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)

    denominator = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denominator <= 1e-15:
        return float("nan")

    return float(np.dot(av, bv) / denominator)


def save_graph_diagnostics(
    fold_results: Sequence[Any],
    trait_dir: Path,
    *,
    make_figure: bool = True,
) -> Dict[str, Any]:
    """
    Save final fold adjacency matrices and across-fold graph stability.

    The matrices are the final frozen graphs returned by train.py, each
    constructed from the corresponding outer-training partition only.
    """

    graph_items: List[Tuple[Any, np.ndarray]] = []

    for result in fold_results:
        adjacency = _result_value(result, "final_adjacency", None)
        if adjacency is None:
            continue

        matrix = np.asarray(adjacency, dtype=np.float64)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Invalid adjacency shape in repeat={result.repeat}, "
                f"fold={result.fold}: {matrix.shape}."
            )
        if not np.isfinite(matrix).all():
            raise FloatingPointError("Adjacency matrix contains non-finite values.")
        if not np.allclose(matrix, matrix.T, atol=1e-6):
            raise ValueError("Final adjacency matrix is not symmetric.")

        graph_items.append((result, matrix))

    if not graph_items:
        return {
            "graphs_saved": 0,
            "graph_diagnostics_available": False,
        }

    shapes = {matrix.shape for _, matrix in graph_items}
    if len(shapes) != 1:
        raise ValueError(
            "Graph dimensions differ across folds; cannot aggregate graph stability."
        )

    graph_dir = trait_dir / "graph_diagnostics"
    ensure_directory(graph_dir)

    matrices = np.stack([matrix for _, matrix in graph_items], axis=0)
    mean_adjacency = np.mean(matrices, axis=0)
    sd_adjacency = (
        np.std(matrices, axis=0, ddof=1)
        if matrices.shape[0] > 1
        else np.zeros_like(mean_adjacency)
    )

    save_csv(
        pd.DataFrame(mean_adjacency).rename_axis("latent_node"),
        graph_dir / "mean_adjacency.csv",
    )
    save_csv(
        pd.DataFrame(sd_adjacency).rename_axis("latent_node"),
        graph_dir / "sd_adjacency.csv",
    )

    upper = np.triu_indices(mean_adjacency.shape[0], k=1)
    mean_vector = mean_adjacency[upper]

    stability_rows: List[Dict[str, Any]] = []

    for (result, matrix) in graph_items:
        vector = matrix[upper]
        stability_rows.append({
            "dataset": result.dataset,
            "trait": result.trait,
            "method": result.method,
            "repeat": int(result.repeat),
            "cv_split_seed": int(result.cv_seed),
            "fold": int(result.fold),
            "cosine_similarity_to_mean_graph": _cosine_similarity(
                vector,
                mean_vector,
            ),
            "frobenius_distance_to_mean_graph": float(
                np.linalg.norm(matrix - mean_adjacency, ord="fro")
            ),
            "mean_off_diagonal_edge_weight": float(np.mean(vector)),
            "sd_off_diagonal_edge_weight": float(np.std(vector, ddof=1))
            if vector.size > 1 else float("nan"),
        })

        fold_name = (
            f"final_adjacency_repeat{int(result.repeat)}_"
            f"seed{int(result.cv_seed)}_fold{int(result.fold)}.csv"
        )
        save_csv(
            pd.DataFrame(matrix).rename_axis("latent_node"),
            graph_dir / fold_name,
        )

    stability = pd.DataFrame(stability_rows).sort_values(
        ["repeat", "cv_split_seed", "fold"]
    ).reset_index(drop=True)
    save_csv(stability, graph_dir / "graph_stability.csv")

    stability_summary = pd.DataFrame([{
        "n_fold_graphs": int(len(stability)),
        "mean_cosine_similarity_to_mean_graph": float(
            stability["cosine_similarity_to_mean_graph"].mean()
        ),
        "sd_cosine_similarity_to_mean_graph": float(
            stability["cosine_similarity_to_mean_graph"].std(ddof=1)
        ),
        "mean_frobenius_distance_to_mean_graph": float(
            stability["frobenius_distance_to_mean_graph"].mean()
        ),
        "sd_frobenius_distance_to_mean_graph": float(
            stability["frobenius_distance_to_mean_graph"].std(ddof=1)
        ),
    }])
    save_csv(stability_summary, graph_dir / "graph_stability_summary.csv")

    if make_figure:
        try:
            import matplotlib.pyplot as plt

            figure_dir = trait_dir / "figures"
            ensure_directory(figure_dir)

            fig, ax = plt.subplots(figsize=(7, 6))
            image = ax.imshow(mean_adjacency, aspect="equal", origin="lower")
            ax.set_xlabel("Latent node")
            ax.set_ylabel("Latent node")
            ax.set_title("Mean frozen adjacency across outer folds")
            fig.colorbar(image, ax=ax, label="Edge weight")
            fig.tight_layout()
            fig.savefig(
                figure_dir / "mean_frozen_adjacency.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
        except ImportError:
            logging.getLogger("DVAGraph-GP.evaluate").warning(
                "Matplotlib is unavailable; graph heatmap was not generated."
            )

    return {
        "graphs_saved": int(len(graph_items)),
        "graph_shape": list(mean_adjacency.shape),
        "graph_diagnostics_available": True,
        "mean_cosine_similarity_to_mean_graph": float(
            stability["cosine_similarity_to_mean_graph"].mean()
        ),
    }


# ---------------------------------------------------------------------------
# 9. Prediction figures
# ---------------------------------------------------------------------------

def save_prediction_figures(
    final_oof: pd.DataFrame,
    trait_dir: Path,
    *,
    font_family: Optional[str] = None,
) -> List[str]:
    """Save observed-versus-predicted and residual diagnostic figures."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.getLogger("DVAGraph-GP.evaluate").warning(
            "Matplotlib is unavailable; prediction figures were skipped."
        )
        return []

    figure_dir = trait_dir / "figures"
    ensure_directory(figure_dir)

    if font_family:
        plt.rcParams["font.family"] = font_family

    truth = final_oof["y_true_original"].to_numpy(dtype=np.float64)
    prediction = final_oof[
        "y_pred_original_mean_across_repeats"
    ].to_numpy(dtype=np.float64)
    residual = truth - prediction

    generated: List[str] = []

    # Figure 1: observed versus repeat-averaged predicted values.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(truth, prediction, alpha=0.65)
    lower = float(min(np.min(truth), np.min(prediction)))
    upper = float(max(np.max(truth), np.max(prediction)))
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 1.0)
        lower -= padding
        upper += padding
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Observed phenotype")
    ax.set_ylabel("Predicted phenotype")
    ax.set_title("Observed vs. repeat-averaged predictions")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = figure_dir / "observed_vs_predicted.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path))

    # Figure 2: residuals versus predicted values.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(prediction, residual, alpha=0.65)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Predicted phenotype")
    ax.set_ylabel("Residual (observed - predicted)")
    ax.set_title("Residuals vs. predicted phenotype")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = figure_dir / "residuals_vs_predicted.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path))

    return generated


# ---------------------------------------------------------------------------
# 10. Evaluation of one dataset/method/trait
# ---------------------------------------------------------------------------

def _check_fold_identity_uniqueness(fold_results: Sequence[Any]) -> None:
    """Prevent duplicate fold result objects from inflating summaries."""

    identities = [
        (
            str(r.dataset),
            str(r.method),
            str(r.trait),
            int(r.repeat),
            int(r.cv_seed),
            int(r.fold),
        )
        for r in fold_results
    ]

    if len(identities) != len(set(identities)):
        raise RuntimeError("Duplicate dataset/method/trait/repeat/seed/fold results.")


def _build_epoch_outputs(
    fold_results: Sequence[Any],
    trait_dir: Path,
) -> Dict[str, int]:
    """Save training histories and epoch-selection records."""

    history_frames: List[pd.DataFrame] = []
    selection_rows: List[Dict[str, Any]] = []

    for result in fold_results:
        for attr in ("inner_history", "outer_history"):
            history = _result_value(result, attr, None)
            if isinstance(history, pd.DataFrame) and not history.empty:
                history_frames.append(history.copy())

        selection_rows.append({
            "dataset": result.dataset,
            "trait": result.trait,
            "method": result.method,
            "repeat": int(result.repeat),
            "cv_split_seed": int(result.cv_seed),
            "fold": int(result.fold),
            "inner_split_seed": int(result.inner_split_seed),
            "selection_seed": int(result.selection_seed),
            "selected_epoch": int(result.selected_epoch),
            "inner_epochs_completed": int(result.inner_epochs_completed),
            "inner_validation_selected_MAE_original": float(
                result.best_inner_validation_mae_original
            ),
            "inner_validation_selected_MSE_original": float(
                result.best_inner_validation_mse_original
            ),
        })

    if history_frames:
        epoch_history = pd.concat(history_frames, ignore_index=True, sort=False)
        sort_columns = [
            c for c in ("repeat", "cv_split_seed", "fold", "stage", "epoch")
            if c in epoch_history.columns
        ]
        if sort_columns:
            epoch_history = epoch_history.sort_values(sort_columns).reset_index(drop=True)
    else:
        epoch_history = pd.DataFrame()

    selection_summary = pd.DataFrame(selection_rows).sort_values(
        ["repeat", "cv_split_seed", "fold"]
    ).reset_index(drop=True)

    save_csv(epoch_history, trait_dir / "epoch_losses.csv")
    save_csv(
        epoch_history.loc[
            epoch_history["stage"].eq("inner_epoch_selection")
        ].copy()
        if "stage" in epoch_history.columns
        else pd.DataFrame(),
        trait_dir / "inner_validation_selection.csv",
    )
    save_csv(selection_summary, trait_dir / "inner_selection_summary.csv")

    loss_columns = [
        column for column in (
            "prediction_loss",
            "reconstruction_loss",
            "kl_loss",
            "weighted_prediction_loss",
            "weighted_reconstruction_loss",
            "weighted_kl_loss",
            "total_loss",
        )
        if column in epoch_history.columns
    ]

    if loss_columns and "stage" in epoch_history.columns:
        loss_summary = mean_sd_table(
            epoch_history,
            ["dataset", "trait", "method", "stage"],
            loss_columns,
        )
    else:
        loss_summary = pd.DataFrame()

    save_csv(loss_summary, trait_dir / "loss_summary.csv")

    return {
        "epoch_history_rows": int(len(epoch_history)),
        "inner_selection_rows": int(
            len(epoch_history.loc[
                epoch_history["stage"].eq("inner_epoch_selection")
            ])
        ) if "stage" in epoch_history.columns else 0,
        "selected_fold_count": int(len(selection_summary)),
    }


def _evaluate_one_group(
    fold_results: Sequence[Any],
    output_dir: Path,
    *,
    settings: Optional[Any],
    overwrite: bool,
    make_figures: bool,
    save_graphs: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Evaluate one dataset/method/trait result collection."""

    _check_fold_identity_uniqueness(fold_results)

    dataset_names = {str(r.dataset) for r in fold_results}
    method_names = {str(r.method) for r in fold_results}
    trait_names = {str(r.trait) for r in fold_results}

    if len(dataset_names) != 1 or len(method_names) != 1 or len(trait_names) != 1:
        raise ValueError("_evaluate_one_group expects exactly one dataset, method, and trait.")

    dataset = next(iter(dataset_names))
    method = next(iter(method_names))
    trait = next(iter(trait_names))

    trait_dir = output_dir / trait

    if trait_dir.exists() and any(trait_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Evaluation output already exists: {trait_dir}. "
            "Set overwrite=True only when intentionally replacing these results."
        )

    ensure_directory(trait_dir)

    # Standardize and validate fold predictions.
    frames: List[pd.DataFrame] = []
    frame_lookup: Dict[Tuple[str, str, str, int, int, int], pd.DataFrame] = {}

    for result in fold_results:
        frame = _prediction_frame_from_result(result)
        frames.append(frame)
        key = (
            str(result.dataset),
            str(result.method),
            str(result.trait),
            int(result.repeat),
            int(result.cv_seed),
            int(result.fold),
        )
        frame_lookup[key] = frame

    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions.sort_values(
        ["repeat", "cv_split_seed", "fold", "sample_index_after_y_filter"]
    ).reset_index(drop=True)

    fold_metrics, metric_audit = build_fold_metrics(
        fold_results,
        frame_lookup,
    )

    # Determine protocol expectations from the supplied TrainingSettings.
    expected_folds = (
        getattr(settings, "expected_folds_per_repeat", None)
        if settings is not None
        else None
    )
    # Do not impose a repeated-K-fold coverage assumption when no experiment
    # settings were supplied. The current Wheat2000 config explicitly sets
    # require_each_sample_once_per_repeat=True.
    require_complete_oof = (
        bool(getattr(settings, "require_each_sample_once_per_repeat", False))
        if settings is not None
        else False
    )

    final_oof = build_repeated_oof_predictions(
        predictions,
        require_complete_repeated_oof=require_complete_oof,
        expected_folds_per_repeat=expected_folds,
    )

    # Main summaries: mean +/- SD across outer folds, matching original runner.
    performance_summary = mean_sd_table(
        fold_metrics,
        ["dataset", "trait", "method"],
        ALL_METRICS,
    )

    summary_by_repeat = mean_sd_table(
        fold_metrics,
        ["dataset", "trait", "method", "repeat", "cv_split_seed"],
        ALL_METRICS,
    )

    # Supplementary metrics from repeat-averaged OOF predictions.
    repeated_metrics = compute_metrics(
        final_oof["y_true_original"].to_numpy(dtype=np.float64),
        final_oof["y_pred_original_mean_across_repeats"].to_numpy(dtype=np.float64),
        prefix="repeat_averaged_oof_",
    )
    repeated_calibration = calibration_diagnostics(
        final_oof["y_true_original"].to_numpy(dtype=np.float64),
        final_oof["y_pred_original_mean_across_repeats"].to_numpy(dtype=np.float64),
        prefix="repeat_averaged_oof_",
    )

    repeated_oof_metrics = pd.DataFrame([{
        "dataset": dataset,
        "trait": trait,
        "method": method,
        "n_samples": int(len(final_oof)),
        "n_cv_runs": int(predictions[["repeat", "cv_split_seed"]].drop_duplicates().shape[0]),
        **repeated_metrics,
        **repeated_calibration,
    }])

    # Audit the complete collection, not only individual folds.
    expected_fold_count = (
        int(expected_folds) *
        int(len(getattr(settings, "expected_cv_seeds", ()) or ()))
        if settings is not None
        and expected_folds is not None
        and getattr(settings, "expected_cv_seeds", None) is not None
        else None
    )

    observed_fold_count = int(len(fold_metrics))
    all_pcc_finite = bool(fold_metrics["orig_pcc_is_finite"].all())
    no_constant_predictions = bool(
        (~fold_metrics["predictions_effectively_constant"]).all()
    )
    all_partition_audits_pass = bool(
        fold_metrics["partition_audit_pass"].all()
    )
    all_metric_audits_pass = bool(metric_audit["pass"].all())
    all_selected_epochs_valid = bool(
        (
            fold_metrics["selected_epoch"].ge(1)
            & fold_metrics["selected_epoch"].le(
                fold_metrics["inner_epochs_completed"]
            )
        ).all()
    )

    performance_eligible = bool(
        all_pcc_finite
        and no_constant_predictions
        and all_partition_audits_pass
        and all_metric_audits_pass
        and all_selected_epochs_valid
        and (
            expected_fold_count is None
            or observed_fold_count == expected_fold_count
        )
    )

    validity = pd.DataFrame([{
        "dataset": dataset,
        "trait": trait,
        "method": method,
        "expected_fold_rows": (
            expected_fold_count
            if expected_fold_count is not None
            else observed_fold_count
        ),
        "observed_fold_rows": observed_fold_count,
        "all_folds_have_finite_PCC": all_pcc_finite,
        "constant_prediction_fold_count": int(
            fold_metrics["predictions_effectively_constant"].sum()
        ),
        "all_partition_audits_pass": all_partition_audits_pass,
        "all_metric_audits_pass": all_metric_audits_pass,
        "all_selected_epochs_valid": all_selected_epochs_valid,
        "complete_repeated_oof_required": require_complete_oof,
        "repeat_averaged_oof_sample_count": int(len(final_oof)),
        "performance_summary_eligible": performance_eligible,
    }])

    performance_summary["performance_summary_eligible"] = performance_eligible
    summary_by_repeat["performance_summary_eligible"] = performance_eligible

    # Save primary and supplementary tables.
    save_csv(fold_metrics, trait_dir / "fold_metrics.csv")
    save_csv(performance_summary, trait_dir / "summary_metrics.csv")
    save_csv(summary_by_repeat, trait_dir / "summary_by_repeat.csv")
    save_csv(predictions, trait_dir / "oof_predictions.csv")
    save_csv(final_oof, trait_dir / "final_repeated_oof_predictions.csv")
    save_csv(repeated_oof_metrics, trait_dir / "repeated_oof_metrics.csv")
    save_csv(metric_audit, trait_dir / "metric_audit.csv")
    save_csv(validity, trait_dir / "performance_validity_audit.csv")

    # Calibration diagnostics per fold.
    calibration_columns = [
        "dataset",
        "trait",
        "method",
        "repeat",
        "cv_split_seed",
        "fold",
        "orig_PCC",
        "orig_R2",
        "orig_RMSE",
        "orig_MSE",
        "orig_MAE",
        "orig_observed_mean",
        "orig_predicted_mean",
        "orig_observed_sd",
        "orig_predicted_sd",
        "orig_prediction_sd_ratio",
        "orig_mean_bias_pred_minus_obs",
        "orig_calibration_slope_pred_on_obs",
        "orig_calibration_intercept",
    ]
    save_csv(
        fold_metrics.loc[:, calibration_columns].copy(),
        trait_dir / "calibration_diagnostics.csv",
    )

    epoch_counts = _build_epoch_outputs(fold_results, trait_dir)

    graph_summary: Dict[str, Any] = {
        "graphs_saved": 0,
        "graph_diagnostics_available": False,
    }
    if save_graphs:
        graph_summary = save_graph_diagnostics(
            fold_results,
            trait_dir,
            make_figure=make_figures,
        )

    figure_paths: List[str] = []
    if make_figures:
        figure_paths = save_prediction_figures(final_oof, trait_dir)

        caption = (
            "Observed-versus-predicted and residual diagnostic plots are based "
            "on repeat-averaged outer-test predictions. These figures are "
            "descriptive; the primary performance evidence is the fold-level "
            "PCC, R2, RMSE, MSE, and MAE summary across outer folds."
        )
        (trait_dir / "figures" / "prediction_figures_caption.txt").write_text(
            caption,
            encoding="utf-8",
        )

    manifest = {
        "dataset": dataset,
        "trait": trait,
        "method": method,
        "n_outer_fold_evaluations": observed_fold_count,
        "n_prediction_rows": int(len(predictions)),
        "n_unique_samples_in_repeat_averaged_oof": int(len(final_oof)),
        "n_cv_runs": int(
            predictions[["repeat", "cv_split_seed"]].drop_duplicates().shape[0]
        ),
        "primary_metrics": [
            "Pearson correlation coefficient (PCC)",
            "Coefficient of determination (R2)",
            "Mean squared error (MSE)",
            "Mean absolute error (MAE)",
        ],
        "additional_metric": "Root mean squared error (RMSE), retained to match the original runner",
        "metric_scale": "Original phenotype units supplied to train.py; no metric-time scaling",
        "primary_aggregation": "Mean and sample SD of fold-level metrics across outer folds",
        "supplementary_aggregation": (
            "For each sample, average its held-out predictions across distinct "
            "(repeat, cv_split_seed) runs, then calculate supplementary OOF metrics"
        ),
        "pcc_undefined_if": (
            "Fewer than three observations or effectively constant observed/predicted values"
        ),
        "no_nonfinite_pairwise_deletion": True,
        "performance_summary_eligible": performance_eligible,
        "graph_diagnostics": graph_summary,
        "epoch_outputs": epoch_counts,
        "figures": figure_paths,
        "output_files": [
            "fold_metrics.csv",
            "summary_metrics.csv",
            "summary_by_repeat.csv",
            "oof_predictions.csv",
            "final_repeated_oof_predictions.csv",
            "repeated_oof_metrics.csv",
            "metric_audit.csv",
            "performance_validity_audit.csv",
            "calibration_diagnostics.csv",
            "epoch_losses.csv",
            "inner_validation_selection.csv",
            "inner_selection_summary.csv",
            "loss_summary.csv",
        ],
    }
    save_json(manifest, trait_dir / "evaluation_manifest.json")

    logger.info(
        "Evaluation complete | dataset=%s | trait=%s | method=%s | "
        "folds=%d | repeated-OOF samples=%d | eligible=%s",
        dataset,
        trait,
        method,
        observed_fold_count,
        len(final_oof),
        performance_eligible,
    )

    return {
        "trait_directory": str(trait_dir),
        "fold_metrics": fold_metrics,
        "summary_metrics": performance_summary,
        "summary_by_repeat": summary_by_repeat,
        "oof_predictions": predictions,
        "final_repeated_oof_predictions": final_oof,
        "repeated_oof_metrics": repeated_oof_metrics,
        "metric_audit": metric_audit,
        "performance_validity_audit": validity,
        "graph_summary": graph_summary,
        "performance_summary_eligible": performance_eligible,
    }


# ---------------------------------------------------------------------------
# 11. Public entry point
# ---------------------------------------------------------------------------

def evaluate_cross_validation(
    fold_results: Sequence[Any],
    output_dir: Optional[Path] = None,
    *,
    settings: Optional[Any] = None,
    overwrite: Optional[bool] = None,
    make_figures: bool = True,
    save_graphs: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all fold results returned by train.run_cross_validation().

    Parameters
    ----------
    fold_results:
        List of FoldTrainingResult objects from train.py.
    output_dir:
        Root result directory. If omitted, settings.output_dir is used.
        For the current Wheat2000 setup, this should be config.OUTPUT_ROOT.
    settings:
        Optional TrainingSettings object. Used for expected CV fold counts,
        repeated-OOF coverage policy, and overwrite behavior.
    overwrite:
        Whether existing per-trait evaluation directories may be replaced.
        If None, settings.overwrite_existing is used (False by default).
    make_figures:
        Generate prediction and graph heatmap figures.
    save_graphs:
        Save fold adjacency matrices and graph stability diagnostics.
    logger:
        Optional experiment logger.

    Returns
    -------
    Dictionary keyed by "dataset::method::trait", containing result tables.
    """

    if not fold_results:
        raise ValueError("fold_results is empty; there is nothing to evaluate.")

    logger = get_logger(logger)

    if output_dir is None:
        if settings is None or getattr(settings, "output_dir", None) is None:
            raise ValueError(
                "Provide output_dir or settings with a non-empty output_dir."
            )
        output_dir = Path(settings.output_dir)
    else:
        output_dir = Path(output_dir)

    if overwrite is None:
        overwrite = bool(
            getattr(settings, "overwrite_existing", False)
            if settings is not None
            else False
        )

    # Group by dataset/method/trait so the evaluator can support multiple traits
    # and can also be reused when a caller supplies more than one dataset.
    grouped: Dict[Tuple[str, str, str], List[Any]] = {}

    for result in fold_results:
        key = (
            str(_result_value(result, "dataset")),
            str(_result_value(result, "method")),
            str(_result_value(result, "trait")),
        )
        grouped.setdefault(key, []).append(result)

    logger.info("=" * 76)
    logger.info("DVAGraph-GP evaluation started")
    logger.info("Fold result objects: %d", len(fold_results))
    logger.info("Dataset/method/trait groups: %d", len(grouped))
    logger.info("Output root: %s", output_dir)
    logger.info("Primary metrics: PCC, R2, MSE, MAE (+ RMSE)")
    logger.info("Metric scale: original phenotype units supplied to train.py")
    logger.info("=" * 76)

    outputs: Dict[str, Dict[str, Any]] = {}

    for (dataset, method, trait), group_results in sorted(grouped.items()):
        if len(grouped) == 1:
            group_output_dir = output_dir
        else:
            group_output_dir = (
                output_dir
                / str(dataset)
                / str(method)
            )

        result = _evaluate_one_group(
            group_results,
            group_output_dir,
            settings=settings,
            overwrite=bool(overwrite),
            make_figures=bool(make_figures),
            save_graphs=bool(save_graphs),
            logger=logger,
        )
        outputs[f"{dataset}::{method}::{trait}"] = result

    logger.info(
        "DVAGraph-GP evaluation completed for %d group(s).",
        len(outputs),
    )
    return outputs


__all__ = [
    "PRIMARY_METRICS",
    "ALL_METRICS",
    "safe_corr",
    "compute_metrics",
    "calibration_diagnostics",
    "mean_sd_table",
    "build_fold_metrics",
    "build_repeated_oof_predictions",
    "save_graph_diagnostics",
    "save_prediction_figures",
    "evaluate_cross_validation",
]
