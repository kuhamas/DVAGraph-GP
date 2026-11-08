# -*- coding: utf-8 -*-
r"""
create_groundnut_protocol.py

Create the ONE locked, model-independent Groundnut benchmark split protocol
that must be reused by DVAGraph-GP and all benchmark methods.

Scientific design
-----------------
- Dataset: Groundnut
- Splitting unit: individual genotype ("Line")
- Outer evaluation: 3 repetitions x 10 shuffled folds
- Outer seeds: 42, 2024, 2025
- Inner validation: first fold of a deterministic shuffled 10-fold split
  performed ONLY inside each outer-training partition
- Inner seed: outer_seed * 1000 + outer_fold
- No phenotype stratification
- No pedigree/family grouping
- The four environments are NOT pooled. They are analyzed separately by the
  model runners, while the SAME genotype-level outer/inner assignments are
  reused across every environment, trait, and benchmark method.

This script DOES NOT train or evaluate any model. It only:
1. validates the Groundnut genotype file;
2. validates Line alignment for all four phenotype environment files;
3. creates the shared outer and inner split assignments;
4. audits the split integrity and outer-test isolation; and
5. saves the exact locked assignments plus reproducibility fingerprints.

Outputs
-------
E:\MoDel_1\github\protocol_splits\groundnut\
    outer_splits.csv
    inner_splits.csv
    fingerprints.json

IMPORTANT
---------
Generate this protocol ONCE before final benchmark runs. Every final Groundnut
model runner should READ these exact split files and must not independently
generate replacement folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

DATASET_NAME = "Groundnut"

DATA_ROOT = Path(r"E:\MoDel_1\Model1\data")
GENOTYPE_PATH = DATA_ROOT / "Groundnut_Markers.csv"

ENV_CONFIGS: Dict[str, Dict[str, Any]] = {
    "env1_ALIYARNAGAR_R15": {
        "label": "ALIYARNAGAR_R15",
        "file": DATA_ROOT / "env1_ALIYARNAGAR_R15_Groundnut_Pheno.csv",
    },
    "env2_ICRISAT_PR15-16": {
        "label": "ICRISAT_PR15-16",
        "file": DATA_ROOT / "env2_ICRISAT_PR15-16_Groundnut_Pheno.csv",
    },
    "env3_ICRISAT_R15": {
        "label": "ICRISAT_R15",
        "file": DATA_ROOT / "env3_ICRISAT_R15_Groundnut_Pheno.csv",
    },
    "env4_JALGOAN_R15": {
        "label": "JALGOAN_R15",
        "file": DATA_ROOT / "env4_JALGOAN_R15_Groundnut_Pheno.csv",
    },
}

TRAIT_CONFIGS = {
    "NPP": {"column": "NPP"},
    "PYPP": {"column": "PYPP"},
    "SYPP": {"column": "SYPP"},
}
TRAITS_TO_VALIDATE = ["NPP", "PYPP", "SYPP"]

# Requested GitHub/reproducibility output folder.
PROTOCOL_DIR = Path(r"E:\MoDel_1\github\protocol_splits\groundnut")
OUTER_SPLIT_PATH = PROTOCOL_DIR / "outer_splits.csv"
INNER_SPLIT_PATH = PROTOCOL_DIR / "inner_splits.csv"
FINGERPRINT_PATH = PROTOCOL_DIR / "fingerprints.json"

GENOTYPE_SAMPLE_ID_COLUMN = "Line"
PHENOTYPE_SAMPLE_ID_COLUMN: Optional[str] = "Line"
PHENOTYPE_SAMPLE_ID_CANDIDATES = (
    "Line", "line", "LINE", "Hybrid_Name", "hybrid_name",
    "sample", "Sample", "SAMPLE", "sample_id", "Sample_ID", "SAMPLE_ID",
    "id", "ID", "IID", "accession", "Accession",
)

CV_SEEDS = (42, 2024, 2025)
N_FOLDS = 10


# =============================================================================
# 2. GENERIC HELPERS
# =============================================================================

def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Sequence[object]) -> str:
    payload = "\n".join(str(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_dataframe_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    stable = (
        frame.loc[:, list(columns)]
        .sort_values(list(columns))
        .reset_index(drop=True)
    )
    payload = stable.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(payload: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def normalise_sample_ids(
    values: pd.Series,
    source_name: str,
) -> np.ndarray:
    """Return unique, non-empty Line identifiers as stable strings."""
    ids = values.astype("string").str.strip()

    invalid = (
        ids.isna()
        | ids.eq("")
        | ids.str.lower().isin({"nan", "none", "null"})
    )
    if bool(invalid.any()):
        raise ValueError(
            f"Missing/empty sample IDs in {source_name}: "
            f"{int(invalid.sum())} row(s)."
        )

    decimal_integer = ids.str.fullmatch(r"[+-]?[0-9]+\.0+")
    ids = ids.mask(
        decimal_integer,
        ids.str.replace(r"\.0+$", "", regex=True),
    )

    if bool(ids.duplicated().any()):
        examples = ids[ids.duplicated()].head(10).tolist()
        raise ValueError(
            f"Duplicate sample IDs in {source_name}; examples: {examples}"
        )

    return ids.astype(str).to_numpy(dtype=object)


def detect_phenotype_id_column(
    frame: pd.DataFrame,
    target_column: str,
) -> Optional[str]:
    if PHENOTYPE_SAMPLE_ID_COLUMN is not None:
        if PHENOTYPE_SAMPLE_ID_COLUMN not in frame.columns:
            raise ValueError(
                f"Configured phenotype sample-ID column "
                f"'{PHENOTYPE_SAMPLE_ID_COLUMN}' is absent."
            )
        return PHENOTYPE_SAMPLE_ID_COLUMN

    for candidate in PHENOTYPE_SAMPLE_ID_CANDIDATES:
        if candidate in frame.columns and candidate != target_column:
            return candidate

    return None


# =============================================================================
# 3. GENOTYPE VALIDATION
# =============================================================================

def load_and_validate_genotype(
    genotype_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    if not genotype_path.exists():
        raise FileNotFoundError(
            f"Groundnut genotype file not found: {genotype_path}"
        )

    frame = pd.read_csv(genotype_path, low_memory=False)

    if GENOTYPE_SAMPLE_ID_COLUMN not in frame.columns:
        raise ValueError(
            f"Required genotype identifier column "
            f"'{GENOTYPE_SAMPLE_ID_COLUMN}' is absent. "
            f"First columns: {frame.columns.tolist()[:12]}"
        )

    sample_ids = normalise_sample_ids(
        frame[GENOTYPE_SAMPLE_ID_COLUMN],
        "Groundnut genotype Line column",
    )

    marker_frame = frame.drop(
        columns=[GENOTYPE_SAMPLE_ID_COLUMN]
    ).copy()

    if marker_frame.shape[1] == 0:
        raise ValueError(
            "No Groundnut marker columns remain after excluding 'Line'."
        )

    if bool(marker_frame.columns.duplicated().any()):
        duplicates = (
            marker_frame.columns[marker_frame.columns.duplicated()]
            .tolist()[:20]
        )
        raise ValueError(
            f"Duplicate Groundnut marker columns detected: {duplicates}"
        )

    marker_numeric = marker_frame.apply(
        pd.to_numeric,
        errors="coerce",
    )

    all_missing = marker_numeric.columns[
        marker_numeric.isna().all()
    ].tolist()
    if all_missing:
        raise ValueError(
            "Completely non-numeric/all-missing marker column(s) detected; "
            f"examples: {all_missing[:20]}"
        )

    X_raw = marker_numeric.to_numpy(dtype=np.float32)
    finite = X_raw[np.isfinite(X_raw)]

    if finite.size == 0:
        raise ValueError(
            "No finite Groundnut marker values remain after numeric conversion."
        )

    allowed = np.asarray(
        [0.0, 1.0, 2.0],
        dtype=np.float32,
    )
    valid = np.isin(finite, allowed)

    if not bool(valid.all()):
        invalid_values = np.unique(
            finite[~valid]
        )[:20].tolist()
        raise ValueError(
            "Expected Groundnut genotype coding in {0,1,2} "
            f"(or binary {{0,1}}); invalid values: {invalid_values}"
        )

    marker_names = [
        str(value) for value in marker_numeric.columns.tolist()
    ]

    observed_unique = sorted(
        float(value)
        for value in np.unique(finite).tolist()
    )

    coding = (
        "binary_0_1"
        if set(observed_unique).issubset({0.0, 1.0})
        else "dosage_0_1_2"
    )

    report: Dict[str, Any] = {
        "dataset": DATASET_NAME,
        "genotype_file": str(genotype_path),
        "genotype_file_sha256": sha256_file(genotype_path),
        "sample_id_column": GENOTYPE_SAMPLE_ID_COLUMN,
        "sample_column_excluded_from_model_input": True,
        "n_samples": int(X_raw.shape[0]),
        "n_markers": int(X_raw.shape[1]),
        "genotype_shape": f"{X_raw.shape[0]}x{X_raw.shape[1]}",
        "marker_order_sha256": sha256_strings(marker_names),
        "sample_order_sha256": sha256_strings(
            [str(value) for value in sample_ids.tolist()]
        ),
        "marker_coding": coding,
        "observed_marker_values": observed_unique,
        "observed_marker_min": float(np.min(finite)),
        "observed_marker_max": float(np.max(finite)),
        "missing_marker_cells": int(np.isnan(X_raw).sum()),
    }

    return X_raw, sample_ids, marker_names, report


# =============================================================================
# 4. PHENOTYPE / ENVIRONMENT ALIGNMENT VALIDATION
# =============================================================================

def validate_environment_phenotypes(
    sample_ids: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Validate all four environment files and all three traits.

    Phenotype VALUES are never used to generate the CV folds. This function
    only verifies that every environment contains the same Groundnut Line IDs
    and records file/alignment fingerprints.
    """
    genotype_set = set(
        str(value) for value in sample_ids.tolist()
    )

    environment_reports: Dict[str, Dict[str, Any]] = {}

    for env_key, env_config in ENV_CONFIGS.items():
        env_label = str(env_config["label"])
        phenotype_path = Path(env_config["file"])

        if not phenotype_path.exists():
            raise FileNotFoundError(
                f"Groundnut phenotype file missing for {env_key}: "
                f"{phenotype_path}"
            )

        frame = pd.read_csv(
            phenotype_path,
            low_memory=False,
        )

        if PHENOTYPE_SAMPLE_ID_COLUMN not in frame.columns:
            raise ValueError(
                f"{env_key}: required phenotype ID column "
                f"'{PHENOTYPE_SAMPLE_ID_COLUMN}' is absent. "
                "Unsafe row-order alignment is not permitted."
            )

        phenotype_ids = normalise_sample_ids(
            frame[PHENOTYPE_SAMPLE_ID_COLUMN],
            f"{env_key} phenotype Line column",
        )

        phenotype_set = set(
            str(value) for value in phenotype_ids.tolist()
        )

        if phenotype_set != genotype_set:
            missing = sorted(
                genotype_set - phenotype_set
            )[:20]
            extra = sorted(
                phenotype_set - genotype_set
            )[:20]
            raise ValueError(
                f"{env_key}: phenotype/genotype Line mismatch. "
                f"Missing phenotype Lines: {missing}; "
                f"extra phenotype Lines: {extra}"
            )

        indexed = frame.copy()
        indexed["_normalised_sample_id"] = phenotype_ids
        indexed = indexed.set_index(
            "_normalised_sample_id",
            verify_integrity=True,
        )

        trait_reports: Dict[str, Dict[str, Any]] = {}

        for trait in TRAITS_TO_VALIDATE:
            target_column = TRAIT_CONFIGS[trait]["column"]

            if target_column not in frame.columns:
                raise ValueError(
                    f"{env_key}/{trait}: target column "
                    f"'{target_column}' is absent from {phenotype_path}."
                )

            y_raw = pd.to_numeric(
                indexed.loc[
                    sample_ids.tolist(),
                    target_column,
                ],
                errors="coerce",
            ).to_numpy(dtype=np.float32)

            if not bool(np.isfinite(y_raw).all()):
                bad = int(
                    np.sum(~np.isfinite(y_raw))
                )
                raise ValueError(
                    f"{env_key}/{trait}: {bad} missing/non-numeric "
                    "phenotype value(s) after exact Line alignment."
                )

            trait_reports[trait] = {
                "phenotype_column": target_column,
                "n_aligned": int(len(y_raw)),
                "n_valid": int(np.isfinite(y_raw).sum()),
                "mean": float(np.mean(y_raw)),
                "sd": float(np.std(y_raw, ddof=1)),
                "minimum": float(np.min(y_raw)),
                "maximum": float(np.max(y_raw)),
            }

        environment_reports[env_key] = {
            "environment_label": env_label,
            "phenotype_file": str(phenotype_path),
            "phenotype_file_sha256": sha256_file(
                phenotype_path
            ),
            "phenotype_sample_id_column": PHENOTYPE_SAMPLE_ID_COLUMN,
            "n_rows_file": int(frame.shape[0]),
            "n_genotype_lines": int(len(sample_ids)),
            "line_set_matches_genotype": True,
            "source_row_order_matches_genotype": bool(
                np.array_equal(phenotype_ids, sample_ids)
            ),
            "alignment_method": (
                "exact Line-ID matching; phenotype values are reordered "
                "to genotype sample order by model runners"
            ),
            "traits": trait_reports,
        }

    return environment_reports


# =============================================================================
# 5. OUTER + INNER PROTOCOL CREATION
# =============================================================================

def create_groundnut_protocol(
    sample_ids: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reproduce the final Groundnut environment-wise shared protocol exactly.

    Outer:
        KFold(n_splits=10, shuffle=True, random_state=seed)

    Inner:
        inner_seed = seed * 1000 + fold
        KFold(n_splits=10, shuffle=True, random_state=inner_seed)
        the FIRST inner fold is used as inner validation

    The protocol is environment-independent and trait-independent.
    """
    n_samples = int(len(sample_ids))

    if n_samples < N_FOLDS:
        raise ValueError(
            f"Groundnut requires at least {N_FOLDS} Lines; "
            f"found {n_samples}."
        )

    all_indices = np.arange(
        n_samples,
        dtype=int,
    )

    outer_rows: List[Dict[str, Any]] = []
    inner_rows: List[Dict[str, Any]] = []

    for repeat, seed in enumerate(
        CV_SEEDS,
        start=1,
    ):
        outer_cv = KFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=int(seed),
        )

        for fold, (
            outer_train_pos,
            outer_test_pos,
        ) in enumerate(
            outer_cv.split(all_indices),
            start=1,
        ):
            outer_train = np.sort(
                all_indices[outer_train_pos]
            )
            outer_test = np.sort(
                all_indices[outer_test_pos]
            )

            for index in outer_train:
                outer_rows.append({
                    "dataset": DATASET_NAME,
                    "repeat": int(repeat),
                    "cv_split_seed": int(seed),
                    "fold": int(fold),
                    "sample_index_after_y_filter": int(index),
                    "sample_id": str(sample_ids[index]),
                    "split": "train",
                })

            for index in outer_test:
                outer_rows.append({
                    "dataset": DATASET_NAME,
                    "repeat": int(repeat),
                    "cv_split_seed": int(seed),
                    "fold": int(fold),
                    "sample_index_after_y_filter": int(index),
                    "sample_id": str(sample_ids[index]),
                    "split": "test",
                })

            # Preserve the exact Groundnut inner-split algorithm.
            inner_seed = int(
                int(seed) * 1000 + int(fold)
            )

            inner_cv = KFold(
                n_splits=10,
                shuffle=True,
                random_state=inner_seed,
            )

            inner_train_pos, inner_validation_pos = next(
                inner_cv.split(outer_train)
            )

            inner_train = np.sort(
                outer_train[inner_train_pos]
            )
            inner_validation = np.sort(
                outer_train[inner_validation_pos]
            )

            for index in inner_train:
                inner_rows.append({
                    "dataset": DATASET_NAME,
                    "repeat": int(repeat),
                    "cv_split_seed": int(seed),
                    "fold": int(fold),
                    "inner_split_seed": int(inner_seed),
                    "sample_index_after_y_filter": int(index),
                    "sample_id": str(sample_ids[index]),
                    "outer_split": "train",
                    "inner_split": "inner_train",
                })

            for index in inner_validation:
                inner_rows.append({
                    "dataset": DATASET_NAME,
                    "repeat": int(repeat),
                    "cv_split_seed": int(seed),
                    "fold": int(fold),
                    "inner_split_seed": int(inner_seed),
                    "sample_index_after_y_filter": int(index),
                    "sample_id": str(sample_ids[index]),
                    "outer_split": "train",
                    "inner_split": "inner_validation",
                })

    outer = (
        pd.DataFrame(outer_rows)
        .sort_values([
            "repeat",
            "cv_split_seed",
            "fold",
            "split",
            "sample_index_after_y_filter",
        ])
        .reset_index(drop=True)
    )

    inner = (
        pd.DataFrame(inner_rows)
        .sort_values([
            "repeat",
            "cv_split_seed",
            "fold",
            "inner_split",
            "sample_index_after_y_filter",
        ])
        .reset_index(drop=True)
    )

    return outer, inner


# =============================================================================
# 6. PROTOCOL AUDIT
# =============================================================================

def audit_outer_inner_protocol(
    outer_df: pd.DataFrame,
    inner_df: pd.DataFrame,
    n_samples: int,
) -> pd.DataFrame:
    """Audit all 30 outer folds and their corresponding inner split."""
    rows: List[Dict[str, Any]] = []

    expected_pairs = [
        (repeat, int(seed))
        for repeat, seed in enumerate(
            CV_SEEDS,
            start=1,
        )
    ]

    observed_pairs = set(
        zip(
            outer_df["repeat"],
            outer_df["cv_split_seed"],
        )
    )

    if observed_pairs != set(expected_pairs):
        raise RuntimeError(
            "Groundnut outer protocol does not match the configured "
            f"repeat/seed pairs. Expected {expected_pairs}; "
            f"observed {sorted(observed_pairs)}."
        )

    for repeat, seed in expected_pairs:
        repeated = outer_df[
            (outer_df["repeat"] == repeat)
            & (outer_df["cv_split_seed"] == seed)
        ]

        test_counts = (
            repeated.loc[
                repeated["split"] == "test"
            ]
            .groupby("sample_index_after_y_filter")
            .size()
        )

        each_sample_once = bool(
            len(test_counts) == n_samples
            and (test_counts == 1).all()
        )

        for fold in range(
            1,
            N_FOLDS + 1,
        ):
            outer_fold = repeated[
                repeated["fold"] == fold
            ]

            inner_fold = inner_df[
                (inner_df["repeat"] == repeat)
                & (inner_df["cv_split_seed"] == seed)
                & (inner_df["fold"] == fold)
            ]

            outer_train = set(
                outer_fold.loc[
                    outer_fold["split"] == "train",
                    "sample_index_after_y_filter",
                ].tolist()
            )

            outer_test = set(
                outer_fold.loc[
                    outer_fold["split"] == "test",
                    "sample_index_after_y_filter",
                ].tolist()
            )

            inner_train = set(
                inner_fold.loc[
                    inner_fold["inner_split"] == "inner_train",
                    "sample_index_after_y_filter",
                ].tolist()
            )

            inner_validation = set(
                inner_fold.loc[
                    inner_fold["inner_split"] == "inner_validation",
                    "sample_index_after_y_filter",
                ].tolist()
            )

            expected_inner_seed = int(
                int(seed) * 1000 + int(fold)
            )

            observed_inner_seeds = {
                int(value)
                for value in inner_fold[
                    "inner_split_seed"
                ].tolist()
            }

            one_correct_inner_seed = (
                observed_inner_seeds
                == {expected_inner_seed}
            )

            no_outer_overlap = not bool(
                outer_train.intersection(
                    outer_test
                )
            )

            no_inner_overlap = not bool(
                inner_train.intersection(
                    inner_validation
                )
            )

            inner_union_equals_outer_train = (
                (inner_train | inner_validation)
                == outer_train
            )

            no_outer_test_in_inner_selection = not bool(
                (inner_train | inner_validation)
                .intersection(outer_test)
            )

            outer_partition_complete = (
                len(outer_train)
                + len(outer_test)
                == n_samples
            )

            inner_partition_complete = (
                len(inner_train)
                + len(inner_validation)
                == len(outer_train)
            )

            no_duplicate_inner_assignments = not bool(
                inner_fold.duplicated(
                    subset=[
                        "sample_index_after_y_filter"
                    ]
                ).any()
            )

            expected_outer_test_sizes = {
                n_samples // N_FOLDS,
                int(np.ceil(
                    n_samples / N_FOLDS
                )),
            }
            valid_outer_test_size = (
                len(outer_test)
                in expected_outer_test_sizes
            )

            sorted_outer_train = np.asarray(
                sorted(outer_train),
                dtype=int,
            )

            expected_inner_cv = KFold(
                n_splits=10,
                shuffle=True,
                random_state=expected_inner_seed,
            )

            expected_train_pos, expected_val_pos = next(
                expected_inner_cv.split(
                    sorted_outer_train
                )
            )

            expected_inner_train = set(
                sorted_outer_train[
                    expected_train_pos
                ].tolist()
            )

            expected_inner_validation = set(
                sorted_outer_train[
                    expected_val_pos
                ].tolist()
            )

            deterministic_reconstruction_matches = (
                inner_train
                == expected_inner_train
                and inner_validation
                == expected_inner_validation
            )

            passed = bool(
                each_sample_once
                and one_correct_inner_seed
                and no_outer_overlap
                and no_inner_overlap
                and inner_union_equals_outer_train
                and no_outer_test_in_inner_selection
                and outer_partition_complete
                and inner_partition_complete
                and no_duplicate_inner_assignments
                and valid_outer_test_size
                and deterministic_reconstruction_matches
            )

            rows.append({
                "dataset": DATASET_NAME,
                "repeat": int(repeat),
                "cv_split_seed": int(seed),
                "fold": int(fold),
                "n_samples": int(n_samples),
                "n_outer_train": int(
                    len(outer_train)
                ),
                "n_outer_test": int(
                    len(outer_test)
                ),
                "n_inner_train": int(
                    len(inner_train)
                ),
                "n_inner_validation": int(
                    len(inner_validation)
                ),
                "inner_split_seed": int(
                    expected_inner_seed
                ),
                "each_sample_once_as_outer_test_in_repeat": (
                    each_sample_once
                ),
                "correct_inner_split_seed": (
                    one_correct_inner_seed
                ),
                "no_outer_train_test_overlap": (
                    no_outer_overlap
                ),
                "no_inner_train_validation_overlap": (
                    no_inner_overlap
                ),
                "inner_union_equals_outer_train": (
                    inner_union_equals_outer_train
                ),
                "no_outer_test_in_inner_selection": (
                    no_outer_test_in_inner_selection
                ),
                "outer_partition_complete": (
                    outer_partition_complete
                ),
                "inner_partition_complete": (
                    inner_partition_complete
                ),
                "no_duplicate_inner_assignments": (
                    no_duplicate_inner_assignments
                ),
                "valid_outer_test_size": (
                    valid_outer_test_size
                ),
                "deterministic_reconstruction_matches": (
                    deterministic_reconstruction_matches
                ),
                "pass": passed,
            })

    audit = (
        pd.DataFrame(rows)
        .sort_values([
            "repeat",
            "cv_split_seed",
            "fold",
        ])
        .reset_index(drop=True)
    )

    if not bool(audit["pass"].all()):
        failed = audit.loc[
            ~audit["pass"]
        ]
        raise RuntimeError(
            "Groundnut protocol audit failed. "
            "No locked protocol files will be written.\n"
            + failed.to_string(index=False)
        )

    return audit


# =============================================================================
# 7. FINGERPRINT / PROTOCOL MANIFEST
# =============================================================================

def build_fingerprints(
    genotype_report: Dict[str, Any],
    marker_names: List[str],
    environment_reports: Dict[str, Dict[str, Any]],
    outer_splits: pd.DataFrame,
    inner_splits: pd.DataFrame,
    audit: pd.DataFrame,
) -> Dict[str, Any]:

    outer_fingerprint_columns = [
        "dataset",
        "repeat",
        "cv_split_seed",
        "fold",
        "sample_index_after_y_filter",
        "sample_id",
        "split",
    ]

    inner_fingerprint_columns = [
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

    first_fold = audit.iloc[0]

    return {
        "protocol_version": (
            "groundnut_envwise_outer3x10_innerval_v1"
        ),
        "dataset": DATASET_NAME,

        "protocol_purpose": (
            "One locked, model-independent genotype-level protocol reused "
            "across all four Groundnut environments, all three traits, "
            "DVAGraph-GP, and all benchmark methods."
        ),

        **genotype_report,

        "marker_names_count": int(
            len(marker_names)
        ),

        "outer_cv_protocol": (
            "3 repeated shuffled 10-fold outer CV; "
            "seeds 42, 2024, and 2025"
        ),

        "cv_split_strategy": (
            "random shuffled KFold at individual-genotype level; "
            "not phenotype-stratified; not pedigree/family grouped"
        ),

        "cv_seeds": list(
            CV_SEEDS
        ),

        "n_outer_folds": int(
            N_FOLDS
        ),

        "n_outer_evaluations_per_trait_per_environment": int(
            len(CV_SEEDS) * N_FOLDS
        ),

        "environment_design": (
            "four environments analyzed separately; identical genotype-level "
            "outer/inner split assignments reused across environments and traits"
        ),

        "environments": environment_reports,

        "traits": list(
            TRAITS_TO_VALIDATE
        ),

        "inner_validation_protocol": (
            "first fold of a deterministic shuffled 10-fold split performed "
            "inside each outer-training partition"
        ),

        "inner_seed_rule": (
            "inner_seed = outer_cv_seed * 1000 + outer_fold"
        ),

        "example_fold_sizes": {
            "n_outer_train": int(
                first_fold["n_outer_train"]
            ),
            "n_outer_test": int(
                first_fold["n_outer_test"]
            ),
            "n_inner_train": int(
                first_fold["n_inner_train"]
            ),
            "n_inner_validation": int(
                first_fold["n_inner_validation"]
            ),
        },

        "outer_split_path": str(
            OUTER_SPLIT_PATH
        ),

        "outer_split_file_sha256": sha256_file(
            OUTER_SPLIT_PATH
        ),

        "outer_split_fingerprint_sha256": (
            stable_dataframe_sha256(
                outer_splits,
                outer_fingerprint_columns,
            )
        ),

        "inner_split_path": str(
            INNER_SPLIT_PATH
        ),

        "inner_split_file_sha256": sha256_file(
            INNER_SPLIT_PATH
        ),

        "inner_split_fingerprint_sha256": (
            stable_dataframe_sha256(
                inner_splits,
                inner_fingerprint_columns,
            )
        ),

        "all_protocol_audits_pass": bool(
            audit["pass"].all()
        ),

        "n_audited_outer_folds": int(
            len(audit)
        ),

        "required_model_runner_policy": {
            "same_assignments": (
                "Every final Groundnut model runner must read these exact "
                "outer_splits.csv and inner_splits.csv files."
            ),
            "environment_handling": (
                "Each environment is modeled separately; model runners must "
                "reuse the same genotype-level fold assignments in every "
                "environment and trait analysis."
            ),
            "inner_stage": (
                "Any data-dependent preprocessing used during model selection "
                "must be fitted using inner_train only and applied unchanged "
                "to inner_validation."
            ),
            "outer_refit_stage": (
                "For final evaluation, preprocessing must be re-estimated "
                "using complete outer_train and applied unchanged to outer_test."
            ),
            "outer_test_isolation": (
                "Outer-test phenotype values must not influence preprocessing, "
                "model selection, epoch selection, hyperparameter selection, "
                "or any other model-development decision."
            ),
            "no_independent_resplitting": (
                "Final model runners must not generate replacement outer or "
                "inner split assignments."
            ),
        },
    }


# =============================================================================
# 8. COMMAND-LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the locked Groundnut environment-wise shared "
            "outer 3x10 + inner-validation benchmark protocol."
        )
    )

    parser.add_argument(
        "--genotype-path",
        type=Path,
        default=GENOTYPE_PATH,
        help=f"Default: {GENOTYPE_PATH}",
    )

    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=PROTOCOL_DIR,
        help=f"Default: {PROTOCOL_DIR}",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite an existing locked protocol. Use only before any "
            "final Groundnut benchmark run has begun."
        ),
    )

    return parser.parse_args()


# =============================================================================
# 9. MAIN
# =============================================================================

def main() -> None:
    global GENOTYPE_PATH
    global PROTOCOL_DIR
    global OUTER_SPLIT_PATH
    global INNER_SPLIT_PATH
    global FINGERPRINT_PATH

    args = parse_args()

    GENOTYPE_PATH = args.genotype_path
    PROTOCOL_DIR = args.protocol_dir

    OUTER_SPLIT_PATH = (
        PROTOCOL_DIR / "outer_splits.csv"
    )
    INNER_SPLIT_PATH = (
        PROTOCOL_DIR / "inner_splits.csv"
    )
    FINGERPRINT_PATH = (
        PROTOCOL_DIR / "fingerprints.json"
    )

    output_paths = [
        OUTER_SPLIT_PATH,
        INNER_SPLIT_PATH,
        FINGERPRINT_PATH,
    ]

    existing_outputs = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "A locked Groundnut protocol already exists. "
            "To protect benchmark reproducibility, this script will not "
            "replace it automatically.\nExisting file: "
            f"{existing_outputs[0]}\n"
            "If you are certain that no final benchmark run depends on "
            "these files, rerun with --overwrite."
        )

    X_raw, sample_ids, marker_names, genotype_report = (
        load_and_validate_genotype(
            GENOTYPE_PATH
        )
    )

    environment_reports = (
        validate_environment_phenotypes(
            sample_ids
        )
    )

    outer_splits, inner_splits = (
        create_groundnut_protocol(
            sample_ids
        )
    )

    audit = audit_outer_inner_protocol(
        outer_splits,
        inner_splits,
        n_samples=int(
            X_raw.shape[0]
        ),
    )

    ensure_directory(
        PROTOCOL_DIR
    )

    if existing_outputs and args.overwrite:
        for path in existing_outputs:
            if path.exists():
                path.unlink()

    outer_splits.to_csv(
        OUTER_SPLIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    inner_splits.to_csv(
        INNER_SPLIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    fingerprints = build_fingerprints(
        genotype_report=genotype_report,
        marker_names=marker_names,
        environment_reports=environment_reports,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        audit=audit,
    )

    write_json(
        fingerprints,
        FINGERPRINT_PATH,
    )

    first_fold = audit.iloc[0]

    print("=" * 92)
    print(
        "SUCCESS: Groundnut shared outer + inner benchmark protocol "
        "created and audited."
    )
    print("=" * 92)
    print(
        f"Dataset: {DATASET_NAME}"
    )
    print(
        f"Genotype file: {GENOTYPE_PATH}"
    )
    print(
        f"Protocol output: {PROTOCOL_DIR}"
    )
    print(
        f"Genotypes/Lines: {X_raw.shape[0]}"
    )
    print(
        f"Marker columns: {X_raw.shape[1]}"
    )
    print(
        "Environments validated: "
        + ", ".join(
            ENV_CONFIGS.keys()
        )
    )
    print(
        "Traits validated in each environment: "
        + ", ".join(
            TRAITS_TO_VALIDATE
        )
    )
    print()
    print(
        "Outer CV: 3 repeats x 10 folds = "
        f"{len(CV_SEEDS) * N_FOLDS} outer-test evaluations "
        "per model-trait combination within each environment"
    )
    print(
        "Outer seeds: "
        + ", ".join(
            str(seed)
            for seed in CV_SEEDS
        )
    )
    print(
        "Inner validation: first fold of shuffled 10-fold split "
        "inside each outer-training set"
    )
    print(
        "Inner seed rule: outer_seed * 1000 + fold"
    )
    print()
    print(
        "Example fold sizes: "
        f"outer_train={int(first_fold['n_outer_train'])}, "
        f"outer_test={int(first_fold['n_outer_test'])}, "
        f"inner_train={int(first_fold['n_inner_train'])}, "
        f"inner_validation={int(first_fold['n_inner_validation'])}"
    )
    print()
    print(
        f"All 30 protocol audits passed: "
        f"{bool(audit['pass'].all())}"
    )
    print()
    print("Created:")
    print(
        f"  {OUTER_SPLIT_PATH}"
    )
    print(
        f"  {INNER_SPLIT_PATH}"
    )
    print(
        f"  {FINGERPRINT_PATH}"
    )
    print()
    print(
        "NEXT STEP: configure every final Groundnut model runner "
        "to READ these exact two split CSV files and reuse them "
        "unchanged across all four environments and all three traits."
    )


if __name__ == "__main__":
    main()
