
"""
DVAGraph-GP Data Management
---------------------------
This module handles:

1. Genotype data loading and validation.
2. Phenotype data loading and sample alignment.
3. Locked outer and inner cross-validation protocols.
4. Dataset and protocol fingerprint verification.
5. Fold-index preparation for model training.

IMPORTANT:
The Wheat2000 data are already preprocessed.

Therefore, this module does NOT perform:
- Missing-value imputation.
- Genotype transformation.
- Phenotype standardization or scaling.
- Feature selection.

Original genotype and phenotype values are preserved.
"""


import hashlib
import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 1. IMPORT CONFIGURATION
# ============================================================

from config import (
    DATASET_NAME,
    GENOTYPE_PATH,
    PHENO_DIR,

    GENOTYPE_SAMPLE_ID_COLUMN,
    PHENOTYPE_SAMPLE_ID_COLUMN,
    PHENOTYPE_SAMPLE_ID_CANDIDATES,

    TRAIT_CONFIGS,
    TRAITS_TO_RUN,

    CV_SEEDS,
    N_FOLDS,

    PROTOCOL_DIR,
    OUTER_SPLIT_PATH,
    INNER_SPLIT_PATH,
    FINGERPRINT_PATH,

    OUTER_AUDIT_PATH,
    INNER_AUDIT_PATH,
    COST_POLICY_PATH,
    TRAINING_POLICY_PATH,
    PROTOCOL_MANIFEST_PATH,

    METHOD_NAME,
    FEATURE_SELECTION_METHOD,
    MARKER_PREPROCESSING,
    MARKER_ORDER_STATUS,
    MODEL_MARKER_ORDER,
)


# ============================================================
# 2. FILE HASHING AND SAMPLE ID UTILITIES
# ============================================================

def sha256_file(path: Path) -> str:
    """Calculate the SHA256 fingerprint of a file."""

    digest = hashlib.sha256()

    with open(path, "rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_strings(values) -> str:
    """Calculate SHA256 fingerprint of an ordered list."""

    content = "\n".join(str(v) for v in values)

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def stable_dataframe_sha256(
    frame: pd.DataFrame,
    columns: List[str]
) -> str:
    """Calculate a stable fingerprint of selected dataframe columns."""

    stable = (
        frame.loc[:, columns]
        .sort_values(columns, kind="mergesort")
        .reset_index(drop=True)
    )

    content = stable.to_csv(
        index=False,
        lineterminator="\n"
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def normalise_sample_ids(
    values,
    source_name: str
) -> np.ndarray:
    """
    Standardize sample identifiers.

    - Remove leading/trailing spaces.
    - Reject missing identifiers.
    - Normalize integer-like IDs such as 1001.0 to 1001.
    - Reject duplicate identifiers.
    """

    ids = pd.Series(
        values,
        dtype="string"
    ).str.strip()

    invalid = (
        ids.isna()
        | ids.eq("")
        | ids.str.lower().isin(
            ["nan", "none", "null"]
        )
    )

    if bool(invalid.any()):

        raise ValueError(
            f"Missing sample identifiers in {source_name}."
        )

    decimal_integer = ids.str.fullmatch(
        r"[+-]?[0-9]+\.0+"
    ).fillna(False)

    ids.loc[decimal_integer] = (
        ids.loc[decimal_integer]
        .str.replace(r"\.0+$", "", regex=True)
    )

    if bool(ids.duplicated().any()):

        raise ValueError(
            f"Duplicate sample identifiers in {source_name}."
        )

    return ids.astype(str).to_numpy(dtype=object)


def detect_phenotype_id_column(
    frame: pd.DataFrame,
    target_column: str
) -> Optional[str]:
    """Identify the sample ID column in a phenotype file."""

    if PHENOTYPE_SAMPLE_ID_COLUMN is not None:

        if PHENOTYPE_SAMPLE_ID_COLUMN not in frame.columns:

            raise ValueError(
                "Configured phenotype sample-ID column "
                "is absent from the phenotype file."
            )

        return PHENOTYPE_SAMPLE_ID_COLUMN

    for candidate in PHENOTYPE_SAMPLE_ID_CANDIDATES:

        if (
            candidate in frame.columns
            and candidate != target_column
        ):

            return candidate

    return None


# ============================================================
# 3. GENOTYPE DATA LOADING
# ============================================================

def load_wheat_genotypes(
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Load the already-preprocessed Wheat2000 genotype dataset.

    Returns
    -------
    X_raw:
        Original genotype matrix [individuals, markers].

    sample_ids:
        Ordered sample identifiers.

    markers:
        Ordered marker names.

    genotype_report:
        Dataset integrity information.
    """

    if not GENOTYPE_PATH.exists():

        raise FileNotFoundError(
            f"Genotype file not found: {GENOTYPE_PATH}"
        )

    frame = pd.read_csv(
        GENOTYPE_PATH,
        low_memory=False
    )

    logger.info(
        "Raw genotype dimensions: %d x %d",
        frame.shape[0],
        frame.shape[1]
    )

    # --------------------------------------------------------
    # 3.1 Validate sample identifiers
    # --------------------------------------------------------

    if GENOTYPE_SAMPLE_ID_COLUMN not in frame.columns:

        raise ValueError(
            f"Missing genotype sample-ID column: "
            f"{GENOTYPE_SAMPLE_ID_COLUMN}"
        )

    sample_ids = normalise_sample_ids(
        frame[GENOTYPE_SAMPLE_ID_COLUMN],
        "genotype sample column"
    )

    # --------------------------------------------------------
    # 3.2 Separate genotype markers from sample IDs
    # --------------------------------------------------------

    marker_frame = frame.drop(
        columns=[GENOTYPE_SAMPLE_ID_COLUMN]
    ).copy()

    if (
        marker_frame.empty
        or bool(marker_frame.columns.duplicated().any())
    ):

        raise ValueError(
            "No markers or duplicate marker names "
            "in genotype file."
        )

    # --------------------------------------------------------
    # 3.3 Convert marker columns to numeric values
    # --------------------------------------------------------

    numeric = marker_frame.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # The input data are already preprocessed.
    # No imputation is performed here.

    if numeric.isna().any().any():

        missing_count = int(
            numeric.isna().sum().sum()
        )

        raise ValueError(
            f"Unexpected missing or non-numeric genotype values: "
            f"{missing_count}. "
            "The preprocessed genotype dataset is expected "
            "to contain complete observations."
        )

    X_raw = numeric.to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X_raw).all():

        raise ValueError(
            "Genotype matrix contains non-finite values."
        )

    # --------------------------------------------------------
    # 3.4 Verify binary genotype coding
    # --------------------------------------------------------

    if not np.isin(
        X_raw,
        [0.0, 1.0]
    ).all():

        invalid = np.unique(
            X_raw[
                ~np.isin(X_raw, [0.0, 1.0])
            ]
        )[:20].tolist()

        raise ValueError(
            "Expected binary 0/1 genotype coding. "
            f"Invalid values: {invalid}"
        )

    markers = [
        str(value)
        for value in numeric.columns
    ]

    # --------------------------------------------------------
    # 3.5 Generate genotype integrity report
    # --------------------------------------------------------

    report = pd.DataFrame([{

        "dataset": DATASET_NAME,

        "genotype_file": str(GENOTYPE_PATH),

        "genotype_file_sha256": sha256_file(
            GENOTYPE_PATH
        ),

        "sample_id_column": GENOTYPE_SAMPLE_ID_COLUMN,

        "sample_column_excluded_from_model_input": True,

        "n_samples": int(X_raw.shape[0]),

        "n_markers_after_cleaning": int(X_raw.shape[1]),

        "genotype_shape": (
            f"{X_raw.shape[0]}x{X_raw.shape[1]}"
        ),

        "sample_order_sha256": sha256_strings(
            sample_ids
        ),

        "marker_order_sha256": sha256_strings(
            markers
        ),

        "missing_marker_cells": 0,

        "non_missing_min": float(X_raw.min()),

        "non_missing_max": float(X_raw.max()),

        "marker_encoding": "binary_0_1",

        "additional_preprocessing": "none",

    }])

    logger.info(
        "Genotype data loaded: %d samples, %d markers",
        X_raw.shape[0],
        X_raw.shape[1]
    )

    return X_raw, sample_ids, markers, report


# ============================================================
# 4. PHENOTYPE DATA LOADING AND SAMPLE ALIGNMENT
# ============================================================

def load_trait_phenotype(
    trait: str,
    sample_ids: np.ndarray,
    logger: logging.Logger
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load one phenotype trait and align it with genotype samples.

    Original phenotype values are preserved.

    No standardization or scaling is performed.
    """

    if trait not in TRAIT_CONFIGS:

        raise ValueError(
            f"Unknown phenotype trait: {trait}"
        )

    cfg = TRAIT_CONFIGS[trait]

    path = PHENO_DIR / cfg["file"]

    if not path.exists():

        raise FileNotFoundError(
            f"Phenotype file not found: {path}"
        )

    frame = pd.read_csv(
        path,
        low_memory=False
    )

    target = cfg["column"]

    if target not in frame.columns:

        raise ValueError(
            f"Trait column {target} not found in {path}. "
            f"Available columns: {frame.columns.tolist()}"
        )

    # --------------------------------------------------------
    # 4.1 Identify phenotype sample ID column
    # --------------------------------------------------------

    id_col = detect_phenotype_id_column(
        frame,
        target
    )

    y = pd.to_numeric(
        frame[target],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 4.2 Align phenotype and genotype observations
    # --------------------------------------------------------

    if id_col is not None:

        phenotype_ids = normalise_sample_ids(
            frame[id_col],
            f"{trait} phenotype ID column"
        )

        mapping = pd.Series(
            y.to_numpy(dtype=np.float64),
            index=phenotype_ids
        )

        missing_ids = [
            sid for sid in sample_ids
            if sid not in mapping.index
        ]

        if missing_ids:

            raise ValueError(
                f"{trait} phenotype is missing "
                f"{len(missing_ids)} genotype sample IDs."
            )

        y_aligned = mapping.loc[
            sample_ids
        ].to_numpy(dtype=np.float64)

        alignment = f"id_matched_using_{id_col}"

        extra_phenotype_samples = len(
            set(phenotype_ids) - set(sample_ids)
        )

    else:

        # Row-order alignment is accepted only when
        # both files contain exactly the same number of rows.

        if len(frame) != len(sample_ids):

            raise ValueError(
                f"{trait} phenotype has no sample ID column "
                "and its row count differs from genotype data."
            )

        y_aligned = y.to_numpy(
            dtype=np.float64
        )

        alignment = "row_order_assumed_after_equal_row_count"

        extra_phenotype_samples = 0

        logger.warning(
            "%s has no sample ID column. "
            "Genotype/phenotype row-order alignment is assumed.",
            trait
        )

    # --------------------------------------------------------
    # 4.3 Validate original phenotype values
    # --------------------------------------------------------

    if not np.isfinite(y_aligned).all():

        raise ValueError(
            f"{trait} contains missing or non-finite phenotype values. "
            "The prepared Wheat2000 phenotype data are expected "
            "to be complete."
        )

    # Preserve original phenotype values.
    # No scaling or standardization.

    # --------------------------------------------------------
    # 4.4 Generate phenotype integrity report
    # --------------------------------------------------------

    report = pd.DataFrame([{

        "dataset": DATASET_NAME,

        "trait": trait,

        "phenotype_file": str(path),

        "phenotype_file_sha256": sha256_file(path),

        "phenotype_column": target,

        "phenotype_id_column": id_col,

        "alignment_method": alignment,

        "n_phenotype_rows": int(len(frame)),

        "n_aligned": int(len(y_aligned)),

        "extra_phenotype_samples": int(
            extra_phenotype_samples
        ),

        "missing_phenotypes": 0,

        "phenotype_mean_original": float(
            np.mean(y_aligned)
        ),

        "phenotype_sd_original": float(
            np.std(y_aligned, ddof=1)
        ),

        "phenotype_min_original": float(
            np.min(y_aligned)
        ),

        "phenotype_max_original": float(
            np.max(y_aligned)
        ),

        "phenotype_scaling": "none",

    }])

    logger.info(
        "Loaded trait=%s | alignment=%s | n=%d",
        trait,
        alignment,
        len(y_aligned)
    )

    return y_aligned.astype(np.float64), report


# ============================================================
# 5. LOAD COMPLETE WHEAT DATASET
# ============================================================

def load_wheat_dataset(
    traits: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Load genotype data and all requested phenotype traits.

    Returns
    -------
    X_raw:
        Genotype matrix.

    sample_ids:
        Ordered sample IDs.

    markers:
        Ordered marker names.

    phenotype_data:
        Dictionary containing original phenotype arrays.

    genotype_report:
        Genotype integrity report.

    phenotype_report:
        Combined phenotype integrity report.
    """

    if logger is None:

        logger = logging.getLogger(__name__)

    if traits is None:

        traits = TRAITS_TO_RUN

    if not traits:

        raise ValueError(
            "No phenotype traits were specified."
        )

    X_raw, sample_ids, markers, genotype_report = (
        load_wheat_genotypes(logger)
    )

    phenotype_data = {}

    phenotype_reports = []

    for trait in traits:

        y_raw, report = load_trait_phenotype(
            trait,
            sample_ids,
            logger
        )

        phenotype_data[trait] = y_raw

        phenotype_reports.append(report)

    phenotype_report = pd.concat(
        phenotype_reports,
        ignore_index=True
    )

    logger.info(
        "Complete Wheat2000 dataset loaded: "
        "%d samples, %d markers, %d traits.",
        X_raw.shape[0],
        X_raw.shape[1],
        len(phenotype_data)
    )

    return (
        X_raw,
        sample_ids,
        markers,
        phenotype_data,
        genotype_report,
        phenotype_report
    )


# ============================================================
# 6. OUTER AND INNER CROSS-VALIDATION SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class OuterInnerFoldSpec:
    """Indices defining one nested cross-validation fold."""

    repeat: int

    cv_seed: int

    fold: int

    inner_split_seed: int

    inner_train_indices: np.ndarray

    inner_validation_indices: np.ndarray

    outer_train_indices: np.ndarray

    outer_test_indices: np.ndarray


# ============================================================
# 7. OUTER SPLIT VALIDATION
# ============================================================

def normalize_outer_split_columns(
    raw: pd.DataFrame,
    sample_ids: np.ndarray
) -> pd.DataFrame:
    """Validate and normalize the outer split table."""

    required = [
        "repeat",
        "cv_split_seed",
        "fold",
        "sample_index_after_y_filter",
        "sample_id",
        "split",
    ]

    missing = set(required) - set(raw.columns)

    if missing:

        raise ValueError(
            f"Outer split file missing columns: {sorted(missing)}"
        )

    out = raw.loc[:, required].copy()

    integer_columns = [
        "repeat",
        "cv_split_seed",
        "fold",
        "sample_index_after_y_filter",
    ]

    for column in integer_columns:

        out[column] = pd.to_numeric(
            out[column],
            errors="raise"
        ).astype(int)

    out["sample_id"] = (
        out["sample_id"]
        .astype(str)
        .str.strip()
    )

    out["split"] = (
        out["split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    if not out["split"].isin(
        ["train", "test"]
    ).all():

        raise ValueError(
            "Outer split labels must be train/test."
        )

    indices = out[
        "sample_index_after_y_filter"
    ].to_numpy(dtype=int)

    if (
        (indices < 0).any()
        or (indices >= len(sample_ids)).any()
    ):

        raise ValueError(
            "Outer split contains invalid sample indices."
        )

    expected_ids = sample_ids[indices].astype(str)

    if not np.array_equal(
        expected_ids,
        out["sample_id"].to_numpy(dtype=str)
    ):

        raise ValueError(
            "Outer split sample IDs do not match "
            "the current genotype sample order."
        )

    duplicate_key = [
        "repeat",
        "cv_split_seed",
        "fold",
        "sample_index_after_y_filter",
    ]

    if out.duplicated(duplicate_key).any():

        raise ValueError(
            "Duplicate sample records found in outer splits."
        )

    out["dataset"] = DATASET_NAME

    return out.sort_values(
        [
            "repeat",
            "cv_split_seed",
            "fold",
            "split",
            "sample_index_after_y_filter",
        ]
    ).reset_index(drop=True)


# ============================================================
# 8. INNER SPLIT VALIDATION
# ============================================================

def normalize_inner_split_columns(
    raw: pd.DataFrame,
    sample_ids: np.ndarray
) -> pd.DataFrame:
    """Validate and normalize the inner split table."""

    required = [
        "repeat",
        "cv_split_seed",
        "fold",
        "inner_split_seed",
        "sample_index_after_y_filter",
        "sample_id",
        "outer_split",
        "inner_split",
    ]

    missing = set(required) - set(raw.columns)

    if missing:

        raise ValueError(
            f"Inner split file missing columns: {sorted(missing)}"
        )

    out = raw.loc[:, required].copy()

    integer_columns = [
        "repeat",
        "cv_split_seed",
        "fold",
        "inner_split_seed",
        "sample_index_after_y_filter",
    ]

    for column in integer_columns:

        out[column] = pd.to_numeric(
            out[column],
            errors="raise"
        ).astype(int)

    out["sample_id"] = (
        out["sample_id"]
        .astype(str)
        .str.strip()
    )

    out["outer_split"] = (
        out["outer_split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    out["inner_split"] = (
        out["inner_split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    if not out["outer_split"].eq("train").all():

        raise ValueError(
            "Inner split records must belong to outer training."
        )

    if not out["inner_split"].isin(
        ["inner_train", "inner_validation"]
    ).all():

        raise ValueError(
            "Invalid inner split labels."
        )

    indices = out[
        "sample_index_after_y_filter"
    ].to_numpy(dtype=int)

    if (
        (indices < 0)
        or (indices >= len(sample_ids))
    ).any():

        raise ValueError(
            "Inner split contains invalid sample indices."
        )

    expected_ids = sample_ids[indices].astype(str)

    if not np.array_equal(
        expected_ids,
        out["sample_id"].to_numpy(dtype=str)
    ):

        raise ValueError(
            "Inner split sample IDs do not match "
            "the current genotype sample order."
        )

    duplicate_key = [
        "repeat",
        "cv_split_seed",
        "fold",
        "sample_index_after_y_filter",
    ]

    if out.duplicated(duplicate_key).any():

        raise ValueError(
            "Duplicate sample records found in inner splits."
        )

    out["dataset"] = DATASET_NAME

    return out.sort_values(
        [
            "repeat",
            "cv_split_seed",
            "fold",
            "inner_split",
            "sample_index_after_y_filter",
        ]
    ).reset_index(drop=True)


# ============================================================
# 9. CROSS-VALIDATION PROTOCOL AUDIT
# ============================================================

def audit_outer_inner_protocol(
    outer_df: pd.DataFrame,
    inner_df: pd.DataFrame,
    n_samples: int
) -> pd.DataFrame:
    """
    Independently audit the predefined nested CV protocol.

    For the current Wheat2000 design:

        Outer training:       1800
        Outer testing:         200
        Inner training:       1620
        Inner validation:      180

    The expected partition sizes are derived from the
    sample count and the configured number of folds.
    """

    rows = []

    expected_pairs = [
        (repeat, seed)
        for repeat, seed in enumerate(CV_SEEDS, start=1)
    ]

    observed_pairs = set(
        zip(
            outer_df["repeat"],
            outer_df["cv_split_seed"]
        )
    )

    if observed_pairs != set(expected_pairs):

        raise RuntimeError(
            "Outer protocol does not match configured CV repetitions and seeds."
        )

    expected_indices = set(range(n_samples))

    expected_outer_test = round(
        n_samples / N_FOLDS
    )

    expected_outer_train = (
        n_samples - expected_outer_test
    )

    expected_inner_validation = round(
        expected_outer_train / N_FOLDS
    )

    expected_inner_train = (
        expected_outer_train - expected_inner_validation
    )

    for repeat, seed in expected_pairs:

        repeated = outer_df[
            (outer_df["repeat"] == repeat)
            & (outer_df["cv_split_seed"] == seed)
        ]

        if set(repeated["fold"]) != set(
            range(1, N_FOLDS + 1)
        ):

            raise RuntimeError(
                f"Missing outer folds in repetition {repeat}."
            )

        test_counts = (
            repeated[repeated["split"] == "test"]
            .groupby("sample_index_after_y_filter")
            .size()
        )

        each_sample_once = bool(
            len(test_counts) == n_samples
            and (test_counts == 1).all()
            and set(test_counts.index) == expected_indices
        )

        for fold in range(1, N_FOLDS + 1):

            outer_fold = repeated[
                repeated["fold"] == fold
            ]

            inner_fold = inner_df[
                (inner_df["repeat"] == repeat)
                & (inner_df["cv_split_seed"] == seed)
                & (inner_df["fold"] == fold)
            ]

            outer_train_rows = outer_fold[
                outer_fold["split"] == "train"
            ]

            outer_test_rows = outer_fold[
                outer_fold["split"] == "test"
            ]

            inner_train_rows = inner_fold[
                inner_fold["inner_split"] == "inner_train"
            ]

            inner_validation_rows = inner_fold[
                inner_fold["inner_split"] == "inner_validation"
            ]

            outer_train = set(
                outer_train_rows[
                    "sample_index_after_y_filter"
                ]
            )

            outer_test = set(
                outer_test_rows[
                    "sample_index_after_y_filter"
                ]
            )

            inner_train = set(
                inner_train_rows[
                    "sample_index_after_y_filter"
                ]
            )

            inner_validation = set(
                inner_validation_rows[
                    "sample_index_after_y_filter"
                ]
            )

            no_outer_overlap = not (
                outer_train & outer_test
            )

            no_inner_overlap = not (
                inner_train & inner_validation
            )

            outer_partition_complete = (
                outer_train | outer_test
            ) == expected_indices

            inner_union_correct = (
                inner_train | inner_validation
            ) == outer_train

            no_test_leakage = not (
                (inner_train | inner_validation)
                & outer_test
            )

            inner_seeds = (
                inner_fold["inner_split_seed"].unique()
            )

            one_inner_seed = len(inner_seeds) == 1

            correct_sizes = (
                len(outer_train) == expected_outer_train
                and len(outer_test) == expected_outer_test
                and len(inner_train) == expected_inner_train
                and len(inner_validation) == expected_inner_validation
                and len(outer_train_rows) == expected_outer_train
                and len(outer_test_rows) == expected_outer_test
                and len(inner_train_rows) == expected_inner_train
                and len(inner_validation_rows) == expected_inner_validation
            )

            passed = bool(
                each_sample_once
                and no_outer_overlap
                and no_inner_overlap
                and outer_partition_complete
                and inner_union_correct
                and no_test_leakage
                and one_inner_seed
                and correct_sizes
            )

            rows.append({

                "dataset": DATASET_NAME,

                "repeat": repeat,

                "cv_split_seed": seed,

                "fold": fold,

                "n_samples": n_samples,

                "n_outer_train": len(outer_train),

                "n_outer_test": len(outer_test),

                "n_inner_train": len(inner_train),

                "n_inner_validation": len(inner_validation),

                "each_sample_once_as_outer_test_in_repeat":
                    each_sample_once,

                "no_outer_train_test_overlap":
                    no_outer_overlap,

                "no_inner_train_validation_overlap":
                    no_inner_overlap,

                "outer_partition_complete":
                    outer_partition_complete,

                "inner_union_equals_outer_train":
                    inner_union_correct,

                "no_outer_test_in_inner_selection":
                    no_test_leakage,

                "one_inner_split_seed":
                    one_inner_seed,

                "correct_partition_sizes":
                    correct_sizes,

                "pass": passed,

            })

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["repeat", "cv_split_seed", "fold"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# 10. LOAD AND VALIDATE LOCKED PROTOCOL
# ============================================================

def _audit_file_passed(path: Path) -> bool:
    """Check whether an optional protocol audit CSV passed."""

    audit = pd.read_csv(path)

    if "pass" not in audit.columns:

        raise ValueError(
            f"Audit file has no 'pass' column: {path}"
        )

    passed = (
        audit["pass"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return bool(passed.all())


def load_locked_protocol(
    sample_ids: np.ndarray,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three core protocol files:

        outer_splits.csv
        inner_splits.csv
        fingerprints.json

    Additional audit CSV files are checked when available.

    The predefined splits are loaded, not regenerated.
    """

    core_files = [
        OUTER_SPLIT_PATH,
        INNER_SPLIT_PATH,
        FINGERPRINT_PATH,
    ]

    missing = [
        str(path)
        for path in core_files
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Core cross-validation protocol files are missing:\n"
            + "\n".join(missing)
        )

    # Load original split files
    outer_raw = pd.read_csv(
        OUTER_SPLIT_PATH,
        dtype={"sample_id": str}
    )

    inner_raw = pd.read_csv(
        INNER_SPLIT_PATH,
        dtype={"sample_id": str}
    )

    # Normalize and validate
    outer = normalize_outer_split_columns(
        outer_raw,
        sample_ids
    )

    inner = normalize_inner_split_columns(
        inner_raw,
        sample_ids
    )

    # Optional pre-existing audit files
    optional_audits = [
        OUTER_AUDIT_PATH,
        INNER_AUDIT_PATH,
    ]

    for audit_path in optional_audits:

        if audit_path.exists():

            if not _audit_file_passed(audit_path):

                raise RuntimeError(
                    f"Protocol audit did not pass: {audit_path}"
                )

            logger.info(
                "Existing audit passed: %s",
                audit_path.name
            )

        else:

            logger.warning(
                "Optional audit file not found: %s. "
                "Independent local audit will be used.",
                audit_path
            )

    # Independent local audit
    local_audit = audit_outer_inner_protocol(
        outer,
        inner,
        len(sample_ids)
    )

    if not bool(local_audit["pass"].all()):

        failed = local_audit[
            ~local_audit["pass"]
        ]

        raise RuntimeError(
            "Independent cross-validation protocol audit failed:\n"
            + failed.to_string(index=False)
        )

    logger.info(
        "Locked protocol loaded and locally audited: %s",
        PROTOCOL_DIR
    )

    return outer, inner, local_audit


# ============================================================
# 11. CREATE OUTER/INNER FOLD SPECIFICATIONS
# ============================================================

def make_outer_inner_specs(
    outer: pd.DataFrame,
    inner: pd.DataFrame
) -> List[OuterInnerFoldSpec]:
    """
    Convert split tables into fold specifications.

    Each specification contains the sample indices needed
    by the training and validation procedures.
    """

    specs = []

    for repeat, seed in enumerate(CV_SEEDS, start=1):

        for fold in range(1, N_FOLDS + 1):

            outer_fold = outer[
                (outer["repeat"] == repeat)
                & (outer["cv_split_seed"] == seed)
                & (outer["fold"] == fold)
            ]

            inner_fold = inner[
                (inner["repeat"] == repeat)
                & (inner["cv_split_seed"] == seed)
                & (inner["fold"] == fold)
            ]

            if outer_fold.empty or inner_fold.empty:

                raise RuntimeError(
                    f"Missing split records for "
                    f"repeat={repeat}, fold={fold}."
                )

            inner_seeds = (
                inner_fold["inner_split_seed"].unique()
            )

            if len(inner_seeds) != 1:

                raise RuntimeError(
                    "Expected exactly one inner split seed per fold."
                )

            spec = OuterInnerFoldSpec(

                repeat=repeat,

                cv_seed=seed,

                fold=fold,

                inner_split_seed=int(inner_seeds[0]),

                inner_train_indices=np.sort(
                    inner_fold[
                        inner_fold["inner_split"] == "inner_train"
                    ]["sample_index_after_y_filter"].to_numpy(
                        dtype=int
                    )
                ),

                inner_validation_indices=np.sort(
                    inner_fold[
                        inner_fold["inner_split"] == "inner_validation"
                    ]["sample_index_after_y_filter"].to_numpy(
                        dtype=int
                    )
                ),

                outer_train_indices=np.sort(
                    outer_fold[
                        outer_fold["split"] == "train"
                    ]["sample_index_after_y_filter"].to_numpy(
                        dtype=int
                    )
                ),

                outer_test_indices=np.sort(
                    outer_fold[
                        outer_fold["split"] == "test"
                    ]["sample_index_after_y_filter"].to_numpy(
                        dtype=int
                    )
                ),

            )

            specs.append(spec)

    expected_folds = len(CV_SEEDS) * N_FOLDS

    if len(specs) != expected_folds:

        raise RuntimeError(
            f"Expected {expected_folds} folds, "
            f"but generated {len(specs)}."
        )

    return specs


# ============================================================
# 12. VERIFY DATA AND PROTOCOL FINGERPRINTS
# ============================================================

def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file and verify that its root is a dictionary."""

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as handle:

        result = json.load(handle)

    if not isinstance(result, dict):

        raise ValueError(
            f"Expected a JSON object in {path}."
        )

    return result


def validate_protocol_fingerprints(
    genotype_report: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compare current genotype and split fingerprints
    against the locked fingerprints.json file.

    Optional benchmark metadata are checked when available.
    """

    locked_data = _load_json(FINGERPRINT_PATH)

    # Support either a flat fingerprint file or a nested
    # {"fingerprints": {...}} structure.
    locked = locked_data.get(
        "fingerprints",
        locked_data
    )

    if not isinstance(locked, dict):

        raise ValueError(
            "Invalid fingerprint JSON structure."
        )

    report = genotype_report.iloc[0].to_dict()

    # --------------------------------------------------------
    # 12.1 Calculate current fingerprints
    # --------------------------------------------------------

    current = {

        "dataset": DATASET_NAME,

        "genotype_file_sha256":
            report["genotype_file_sha256"],

        "sample_id_column":
            report["sample_id_column"],

        "sample_column_excluded_from_model_input":
            bool(report["sample_column_excluded_from_model_input"]),

        "n_samples":
            int(report["n_samples"]),

        "n_markers":
            int(report["n_markers_after_cleaning"]),

        "genotype_shape":
            report["genotype_shape"],

        "marker_order_sha256":
            report["marker_order_sha256"],

        "sample_order_sha256":
            report["sample_order_sha256"],

        "canonical_outer_split_file_sha256":
            sha256_file(OUTER_SPLIT_PATH),

        "canonical_inner_split_file_sha256":
            sha256_file(INNER_SPLIT_PATH),

        "outer_split_fingerprint_sha256":
            stable_dataframe_sha256(
                outer,
                [
                    "dataset",
                    "repeat",
                    "cv_split_seed",
                    "fold",
                    "sample_index_after_y_filter",
                    "sample_id",
                    "split",
                ]
            ),

        "inner_split_fingerprint_sha256":
            stable_dataframe_sha256(
                inner,
                [
                    "dataset",
                    "repeat",
                    "cv_split_seed",
                    "fold",
                    "inner_split_seed",
                    "sample_index_after_y_filter",
                    "sample_id",
                    "outer_split",
                    "inner_split",
                ]
            ),

    }

    # --------------------------------------------------------
    # 12.2 Compare mandatory locked fingerprints
    # --------------------------------------------------------

    strict_keys = [

        "dataset",

        "genotype_file_sha256",

        "sample_id_column",

        "sample_column_excluded_from_model_input",

        "n_samples",

        "n_markers",

        "genotype_shape",

        "marker_order_sha256",

        "sample_order_sha256",

        "canonical_outer_split_file_sha256",

        "canonical_inner_split_file_sha256",

    ]

    missing_keys = [
        key for key in strict_keys
        if key not in locked
    ]

    if missing_keys:

        raise ValueError(
            "Required fingerprint fields are missing: "
            + ", ".join(missing_keys)
        )

    mismatches = [
        key for key in strict_keys
        if locked[key] != current[key]
    ]

    if mismatches:

        raise RuntimeError(
            "Current data or split files do not match "
            "the locked protocol: "
            + ", ".join(mismatches)
        )

    # --------------------------------------------------------
    # 12.3 Check split-table fingerprints when recorded
    # --------------------------------------------------------

    optional_fingerprint_keys = [
        "outer_split_fingerprint_sha256",
        "inner_split_fingerprint_sha256",
    ]

    unchecked_fingerprints = []

    for key in optional_fingerprint_keys:

        if key in locked:

            if locked[key] != current[key]:

                raise RuntimeError(
                    f"Protocol fingerprint mismatch: {key}"
                )

        else:

            unchecked_fingerprints.append(key)

    # --------------------------------------------------------
    # 12.4 Check optional benchmark metadata
    # --------------------------------------------------------

    optional_metadata = {}

    optional_files = {

        "cost_policy": COST_POLICY_PATH,

        "training_policy": TRAINING_POLICY_PATH,

        "protocol_manifest": PROTOCOL_MANIFEST_PATH,

    }

    missing_metadata = []

    for name, path in optional_files.items():

        if path.exists():

            optional_metadata[name] = _load_json(path)

        else:

            missing_metadata.append(name)

    # Cost policy verification, when available
    cost_policy_version = None

    if "cost_policy" in optional_metadata:

        cost = optional_metadata["cost_policy"]

        cost_policy_version = cost.get(
            "cost_policy_version"
        )

        expected_version = (
            "wheat_benchmark_cost_policy_v7_final_outerinner"
        )

        if cost_policy_version != expected_version:

            raise RuntimeError(
                "Unexpected benchmark cost policy version."
            )

    # Protocol manifest verification, when available
    if "protocol_manifest" in optional_metadata:

        manifest = optional_metadata["protocol_manifest"]

        if (
            not manifest.get("all_outer_audits_pass", False)
            or not manifest.get("all_inner_audits_pass", False)
        ):

            raise RuntimeError(
                "Protocol manifest does not confirm passing audits."
            )

    return {

        **current,

        "protocol_version":
            locked.get("protocol_version"),

        "protocol_dir":
            str(PROTOCOL_DIR),

        "cost_policy_version":
            cost_policy_version,

        "unchecked_split_fingerprints":
            unchecked_fingerprints,

        "missing_optional_metadata":
            missing_metadata,

        "locked_protocol_validation":
            "passed",

    }


# ============================================================
# 13. FEATURE MANIFEST
# ============================================================

def feature_manifest(
    markers: List[str],
    trait: str,
    spec: OuterInnerFoldSpec,
    stage: str,
    partition: str
) -> pd.DataFrame:
    """
    Record the marker information used in an experimental fold.

    No feature selection or marker transformation is performed.
    """

    n_markers = len(markers)

    return pd.DataFrame([{

        "dataset": DATASET_NAME,

        "trait": trait,

        "method": METHOD_NAME,

        "repeat": spec.repeat,

        "cv_split_seed": spec.cv_seed,

        "fold": spec.fold,

        "stage": stage,

        "partition": partition,

        "feature_selection_method":
            FEATURE_SELECTION_METHOD,

        "marker_preprocessing":
            MARKER_PREPROCESSING,

        "input_marker_count":
            n_markers,

        "markers_retained":
            n_markers,

        "markers_removed_by_feature_selection": 0,

        "all_input_markers_retained": True,

        "marker_order_status":
            MARKER_ORDER_STATUS,

        "model_marker_order":
            MODEL_MARKER_ORDER,

        "additional_preprocessing":
            "none",

    }])
