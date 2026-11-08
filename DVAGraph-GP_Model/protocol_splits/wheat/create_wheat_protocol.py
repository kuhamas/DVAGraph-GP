# -*- coding: utf-8 -*-
r"""
create_wheat_protocol.py

Creates the ONE FINAL, locked shared protocol for every reviewer-facing
Wheat2000 benchmark model.

IMPORTANT
---------
This script is self-contained. It does NOT train any model. It creates and
audits the canonical split assignments that every final Wheat benchmark runner
must read. It creates both:
    1) repeated shuffled outer 3 x 10-fold CV assignments; and
    2) one fixed inner-validation holdout inside each outer-training fold.

All protocol outputs are saved under:
    E:\MoDel_1\github\protocol_splits\wheat\


Final benchmark workflow for EVERY method
-----------------------------------------
inner_train
  -> fit required preprocessing / target scaling
  -> train candidate epochs/checkpoints
inner_validation
  -> select only the epoch/checkpoint using a pre-specified rule
outer_train (all 1,800 samples)
  -> refit required preprocessing from scratch
  -> refit a fresh model for the selected fixed epoch count
outer_test (200 untouched samples)
  -> evaluate exactly once; no outer-test label can influence any decision

Run ONCE before final reviewer-facing model runs:
    python create_wheat_protocol.py

Do not rerun or overwrite this folder after final model benchmarking begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

DATASET_NAME = "Wheat2000"
GENOTYPE_PATH = Path(r"D:\hamas123\MyModels\Model1\data\wheat2000\genotype_markers.csv")
PHENOTYPE_DIR = Path(r"E:\MoDel_1\Model1\data")
RESULTS_ROOT = Path(r"D:\hamas123\MyModels\KBS_Model1\Results")

# All protocol files created by this script are saved HERE.
FINAL_PROTOCOL_DIR = Path(r"E:\MoDel_1\github\protocol_splits\wheat")

GENOTYPE_SAMPLE_ID_COLUMN = "sample"
PHENOTYPE_SAMPLE_ID_COLUMN: Optional[str] = None
PHENOTYPE_SAMPLE_ID_CANDIDATES = (
    "sample", "Sample", "SAMPLE", "sample_id", "Sample_ID", "SAMPLE_ID",
    "id", "ID", "IID", "accession", "Accession",
)

OUTER_CV_SEEDS = (42, 2024, 2025)
N_OUTER_FOLDS = 10
INNER_VALIDATION_FRACTION = 0.10

TRAIT_CONFIGS = {
    "TKW": {"file": "2000_1_phe.csv", "column": "tkw"},
    "TW":  {"file": "2000_2_phe.csv", "column": "testw"},
    "GL":  {"file": "2000_3_phe.csv", "column": "length"},
    "GW":  {"file": "2000_4_phe.csv", "column": "width"},
    "GH":  {"file": "2000_5_phe.csv", "column": "Hard"},
    "GP":  {"file": "2000_6_phe.csv", "column": "Prot"},
}


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


def stable_dataframe_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    stable = frame.loc[:, list(columns)].sort_values(list(columns)).reset_index(drop=True)
    return hashlib.sha256(stable.to_csv(index=False).encode("utf-8")).hexdigest()


def write_json(payload: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def normalise_sample_ids(values: pd.Series, source_name: str) -> np.ndarray:
    """Return unique, non-empty, stable sample IDs as strings."""
    ids = values.astype("string").str.strip()
    invalid = ids.isna() | ids.eq("") | ids.str.lower().isin({"nan", "none", "null"})
    if bool(invalid.any()):
        raise ValueError(
            "Missing/empty sample identifiers in {0}: {1} row(s).".format(
                source_name, int(invalid.sum())
            )
        )

    # Match prior protocol handling: 1.0 -> 1, while preserving IDs such as 001.
    decimal_integer = ids.str.fullmatch(r"[+-]?[0-9]+\.0+")
    ids = ids.mask(decimal_integer, ids.str.replace(r"\.0+$", "", regex=True))

    if bool(ids.duplicated().any()):
        examples = ids[ids.duplicated()].head(10).tolist()
        raise ValueError(
            "Duplicate sample identifiers in {0}; examples: {1}".format(source_name, examples)
        )
    return ids.astype(str).to_numpy(dtype=object)


def detect_phenotype_id_column(frame: pd.DataFrame, target_column: str) -> Optional[str]:
    if PHENOTYPE_SAMPLE_ID_COLUMN is not None:
        if PHENOTYPE_SAMPLE_ID_COLUMN not in frame.columns:
            raise ValueError(
                "Configured phenotype ID column '{0}' is absent.".format(PHENOTYPE_SAMPLE_ID_COLUMN)
            )
        return PHENOTYPE_SAMPLE_ID_COLUMN

    for candidate in PHENOTYPE_SAMPLE_ID_CANDIDATES:
        if candidate in frame.columns and candidate != target_column:
            return candidate
    return None


def derive_inner_seed(repeat: int, outer_seed: int, fold: int) -> int:
    """Return a deterministic, portable positive 31-bit seed.

    The seed is deliberately restricted to ``1 .. 2,147,483,646``.  This
    prevents platform-dependent signed-integer overflow on Windows when the
    seed column is audited or exported, while preserving deterministic,
    fold-specific randomisation.
    """
    token = "{0}|outer={1}|repeat={2}|fold={3}|inner_validation_v5_portable_seed".format(
        DATASET_NAME, outer_seed, repeat, fold
    )
    digest_value = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
    return int(1 + (digest_value % (2**31 - 2)))


# =============================================================================
# 3. SOURCE-DATA VALIDATION
# =============================================================================

def load_and_validate_genotype(
    genotype_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, object]]:
    if not genotype_path.exists():
        raise FileNotFoundError("Genotype file not found: {0}".format(genotype_path))

    genotype_frame = pd.read_csv(genotype_path, low_memory=False)
    if GENOTYPE_SAMPLE_ID_COLUMN not in genotype_frame.columns:
        raise ValueError(
            "Required genotype sample-ID column '{0}' is absent. First columns: {1}".format(
                GENOTYPE_SAMPLE_ID_COLUMN, genotype_frame.columns.tolist()[:12]
            )
        )

    sample_ids = normalise_sample_ids(
        genotype_frame[GENOTYPE_SAMPLE_ID_COLUMN],
        "genotype column '{0}'".format(GENOTYPE_SAMPLE_ID_COLUMN),
    )

    marker_frame = genotype_frame.drop(columns=[GENOTYPE_SAMPLE_ID_COLUMN]).copy()
    if marker_frame.shape[1] == 0:
        raise ValueError("No marker columns remain after excluding the sample-ID column.")
    if bool(marker_frame.columns.duplicated().any()):
        duplicates = marker_frame.columns[marker_frame.columns.duplicated()].tolist()[:10]
        raise ValueError("Duplicate marker names detected; examples: {0}".format(duplicates))

    marker_numeric = marker_frame.apply(pd.to_numeric, errors="coerce")
    all_missing_columns = marker_numeric.columns[marker_numeric.isna().all()].tolist()
    if all_missing_columns:
        raise ValueError(
            "The following marker columns are completely non-numeric: {0}".format(
                all_missing_columns[:20]
            )
        )

    X_raw = marker_numeric.to_numpy(dtype=np.float32)
    finite_values = X_raw[np.isfinite(X_raw)]
    if finite_values.size == 0:
        raise ValueError("No finite marker values remain after numeric conversion.")

    marker_min = float(np.min(finite_values))
    marker_max = float(np.max(finite_values))
    is_binary = np.isin(finite_values, np.asarray([0.0, 1.0], dtype=np.float32))
    if not bool(is_binary.all()):
        invalid_values = np.unique(finite_values[~is_binary])[:20].tolist()
        raise ValueError(
            "Expected binary DArT marker coding 0/1 after excluding '{0}', but found: {1}".format(
                GENOTYPE_SAMPLE_ID_COLUMN, invalid_values
            )
        )

    marker_names = [str(column) for column in marker_numeric.columns.tolist()]
    report: Dict[str, object] = {
        "dataset": DATASET_NAME,
        "genotype_file": str(genotype_path),
        "genotype_file_sha256": sha256_file(genotype_path),
        "sample_id_column": GENOTYPE_SAMPLE_ID_COLUMN,
        "n_samples": int(X_raw.shape[0]),
        "n_markers": int(X_raw.shape[1]),
        "genotype_shape": "{0}x{1}".format(X_raw.shape[0], X_raw.shape[1]),
        "marker_order_sha256": sha256_strings(marker_names),
        "sample_order_sha256": sha256_strings(sample_ids.tolist()),
        "marker_coding": (
            "binary 0/1; missing marker values are permitted and must be handled "
            "inside the appropriate training subset only"
        ),
        "observed_marker_min": marker_min,
        "observed_marker_max": marker_max,
        "sample_column_excluded_from_model_input": True,
    }
    return X_raw, sample_ids, marker_names, report


def validate_phenotype_files(
    phenotype_dir: Path,
    sample_ids: np.ndarray,
) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame]:
    reports: Dict[str, Dict[str, object]] = {}
    audit_rows: List[Dict[str, object]] = []

    for trait, config in TRAIT_CONFIGS.items():
        phenotype_path = phenotype_dir / config["file"]
        target_column = config["column"]
        if not phenotype_path.exists():
            raise FileNotFoundError(
                "Phenotype file for {0} not found: {1}".format(trait, phenotype_path)
            )

        phenotype_frame = pd.read_csv(phenotype_path, low_memory=False)
        if target_column not in phenotype_frame.columns:
            raise ValueError(
                "Trait {0}: target column '{1}' is absent from {2}".format(
                    trait, target_column, phenotype_path
                )
            )

        target_values = pd.to_numeric(phenotype_frame[target_column], errors="coerce")
        if bool(target_values.isna().any()):
            raise ValueError(
                "Trait {0} contains {1} missing/non-numeric phenotype value(s). "
                "This locked protocol requires complete target values for all samples.".format(
                    trait, int(target_values.isna().sum())
                )
            )

        phenotype_id_column = detect_phenotype_id_column(phenotype_frame, target_column)
        if phenotype_id_column is not None:
            phenotype_ids = normalise_sample_ids(
                phenotype_frame[phenotype_id_column],
                "phenotype {0} ID column '{1}'".format(trait, phenotype_id_column),
            )
            if set(phenotype_ids.tolist()) != set(sample_ids.tolist()):
                missing = sorted(set(sample_ids.tolist()) - set(phenotype_ids.tolist()))[:10]
                extra = sorted(set(phenotype_ids.tolist()) - set(sample_ids.tolist()))[:10]
                raise ValueError(
                    "Trait {0}: genotype/phenotype sample-ID mismatch. "
                    "Missing phenotype IDs: {1}; extra phenotype IDs: {2}".format(
                        trait, missing, extra
                    )
                )
            order_matches_genotype = bool(np.array_equal(phenotype_ids, sample_ids))
            alignment_method = "matched by explicit phenotype sample-ID column '{0}'".format(
                phenotype_id_column
            )
            downstream_requirement = (
                "Every model runner must merge/reindex phenotype values by the validated sample ID "
                "before applying shared split indices."
            )
        else:
            if phenotype_frame.shape[0] != len(sample_ids):
                raise ValueError(
                    "Trait {0}: unsafe row-order alignment: genotype rows={1}, phenotype rows={2}.".format(
                        trait, len(sample_ids), phenotype_frame.shape[0]
                    )
                )
            order_matches_genotype = True
            alignment_method = (
                "row order assumed because no phenotype sample-ID column was available; "
                "this must be a data-source guarantee"
            )
            downstream_requirement = (
                "Every model runner must preserve this validated source row order exactly."
            )

        reports[trait] = {
            "path": str(phenotype_path),
            "sha256": sha256_file(phenotype_path),
            "target_column": target_column,
            "n_rows": int(phenotype_frame.shape[0]),
            "phenotype_sample_id_column": phenotype_id_column,
            "alignment_method": alignment_method,
            "phenotype_order_matches_genotype": order_matches_genotype,
            "mean": float(target_values.mean()),
            "sd": float(target_values.std(ddof=1)),
            "minimum": float(target_values.min()),
            "maximum": float(target_values.max()),
        }
        audit_rows.append({
            "trait": trait,
            "phenotype_file": str(phenotype_path),
            "target_column": target_column,
            "phenotype_sample_id_column": phenotype_id_column,
            "alignment_method": alignment_method,
            "phenotype_order_matches_genotype": order_matches_genotype,
            "downstream_model_requirement": downstream_requirement,
        })

    return reports, pd.DataFrame(audit_rows)


# =============================================================================
# 4. OUTER CV CREATION AND AUDIT
# =============================================================================

OUTER_SPLIT_COLUMNS = (
    "dataset", "repeat", "cv_split_seed", "fold",
    "sample_index_after_y_filter", "sample_id", "split",
)


def create_repeated_outer_splits(sample_ids: np.ndarray) -> pd.DataFrame:
    indices = np.arange(len(sample_ids), dtype=int)
    rows: List[Dict[str, object]] = []

    for repeat, seed in enumerate(OUTER_CV_SEEDS, start=1):
        splitter = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=int(seed))
        for fold, (outer_train_indices, outer_test_indices) in enumerate(splitter.split(indices), start=1):
            for index in np.sort(outer_train_indices):
                rows.append({
                    "dataset": DATASET_NAME,
                    "repeat": int(repeat),
                    "cv_split_seed": int(seed),
                    "fold": int(fold),
                    "sample_index_after_y_filter": int(index),
                    "sample_id": str(sample_ids[index]),
                    "split": "train",
                })
            for index in np.sort(outer_test_indices):
                rows.append({
                    "dataset": DATASET_NAME,
                    "repeat": int(repeat),
                    "cv_split_seed": int(seed),
                    "fold": int(fold),
                    "sample_index_after_y_filter": int(index),
                    "sample_id": str(sample_ids[index]),
                    "split": "test",
                })

    return pd.DataFrame(rows).sort_values(
        ["repeat", "cv_split_seed", "fold", "split", "sample_index_after_y_filter"]
    ).reset_index(drop=True)


def audit_outer_splits(outer_splits: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    expected_pairs = [(repeat, int(seed)) for repeat, seed in enumerate(OUTER_CV_SEEDS, start=1)]
    observed_pairs = set(zip(outer_splits["repeat"], outer_splits["cv_split_seed"]))
    if observed_pairs != set(expected_pairs):
        raise RuntimeError(
            "Unexpected repeat/seed combinations. Expected {0}; got {1}".format(
                expected_pairs, sorted(observed_pairs)
            )
        )

    audit_rows: List[Dict[str, object]] = []
    for repeat, seed in expected_pairs:
        repeat_frame = outer_splits[
            (outer_splits["repeat"] == repeat) & (outer_splits["cv_split_seed"] == seed)
        ]
        test_counts = (
            repeat_frame.loc[repeat_frame["split"] == "test"]
            .groupby("sample_index_after_y_filter")
            .size()
        )
        each_sample_once_as_test = bool(
            len(test_counts) == n_samples and (test_counts == 1).all()
        )
        all_folds_present = set(repeat_frame["fold"].unique().tolist()) == set(
            range(1, N_OUTER_FOLDS + 1)
        )

        no_overlap_all_folds = True
        valid_fold_sizes = True
        for fold in range(1, N_OUTER_FOLDS + 1):
            fold_frame = repeat_frame[repeat_frame["fold"] == fold]
            outer_train = set(
                fold_frame.loc[fold_frame["split"] == "train", "sample_index_after_y_filter"].tolist()
            )
            outer_test = set(
                fold_frame.loc[fold_frame["split"] == "test", "sample_index_after_y_filter"].tolist()
            )
            no_overlap_all_folds = no_overlap_all_folds and not bool(outer_train.intersection(outer_test))
            valid_fold_sizes = valid_fold_sizes and (
                len(outer_train) > 0 and len(outer_test) > 0 and len(outer_train) + len(outer_test) == n_samples
            )

        audit_rows.append({
            "dataset": DATASET_NAME,
            "repeat": int(repeat),
            "cv_split_seed": int(seed),
            "n_samples": int(n_samples),
            "each_sample_once_as_outer_test": bool(each_sample_once_as_test),
            "no_outer_train_test_overlap": bool(no_overlap_all_folds),
            "expected_outer_folds_present": bool(all_folds_present),
            "valid_outer_fold_sizes": bool(valid_fold_sizes),
            "pass": bool(
                each_sample_once_as_test and no_overlap_all_folds and all_folds_present and valid_fold_sizes
            ),
        })

    audit = pd.DataFrame(audit_rows)
    if not bool(audit["pass"].all()):
        raise RuntimeError("Outer CV audit failed. No protocol files were written.")
    return audit


# =============================================================================
# 5. INNER-VALIDATION CREATION AND AUDIT
# =============================================================================

def inner_validation_count(n_outer_train: int, validation_fraction: float) -> int:
    """Return the deterministic number of held-out inner-validation samples.

    We explicitly use ceil(n * fraction), rather than relying on a library's
    float-to-integer behavior.  With 2,000 samples and a 10-fold outer CV this
    gives 180 of 1,800 outer-training samples (10%) for inner validation.
    """
    if n_outer_train < 2:
        raise ValueError("Each outer-training fold must contain at least two samples.")
    if not 0.0 < float(validation_fraction) < 0.5:
        raise ValueError(
            "INNER_VALIDATION_FRACTION must be >0 and <0.5; received {0}".format(
                validation_fraction
            )
        )
    n_validation = int(np.ceil(float(n_outer_train) * float(validation_fraction)))
    return max(1, min(n_validation, n_outer_train - 1))


def deterministic_inner_partition(
    outer_train_indices: np.ndarray,
    inner_seed: int,
    validation_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a stable, reproducible inner train/validation partition.

    It intentionally does not use ShuffleSplit.  This avoids any dependency on
    scikit-learn's internal float rounding rules across versions.  The selected
    validation indices are based solely on the *outer-training* sample indices
    and the saved fold-specific seed.
    """
    outer_train = np.sort(np.asarray(outer_train_indices, dtype=int))
    n_validation = inner_validation_count(len(outer_train), validation_fraction)
    rng = np.random.RandomState(int(inner_seed) % (2**32 - 1))
    permutation = rng.permutation(len(outer_train))
    validation_positions = np.sort(permutation[:n_validation])
    train_positions = np.sort(permutation[n_validation:])
    inner_validation = np.sort(outer_train[validation_positions])
    inner_train = np.sort(outer_train[train_positions])
    return inner_train, inner_validation


def create_inner_validation_splits(
    outer_splits: pd.DataFrame,
    sample_ids: np.ndarray,
    validation_fraction: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_columns = ["repeat", "cv_split_seed", "fold"]

    for (repeat, outer_seed, fold), fold_frame in outer_splits.groupby(group_columns, sort=True):
        outer_train = np.sort(
            fold_frame.loc[
                fold_frame["split"] == "train", "sample_index_after_y_filter"
            ].to_numpy(dtype=int)
        )
        outer_test = np.sort(
            fold_frame.loc[
                fold_frame["split"] == "test", "sample_index_after_y_filter"
            ].to_numpy(dtype=int)
        )
        if outer_train.size == 0 or outer_test.size == 0:
            raise RuntimeError(
                "Empty outer train/test set in repeat={0}, seed={1}, fold={2}".format(
                    repeat, outer_seed, fold
                )
            )
        if bool(set(outer_train.tolist()).intersection(set(outer_test.tolist()))):
            raise RuntimeError(
                "Outer train/test overlap in repeat={0}, seed={1}, fold={2}".format(
                    repeat, outer_seed, fold
                )
            )

        inner_seed = derive_inner_seed(int(repeat), int(outer_seed), int(fold))
        inner_train, inner_validation = deterministic_inner_partition(
            outer_train,
            inner_seed=inner_seed,
            validation_fraction=float(validation_fraction),
        )

        for index in inner_train:
            rows.append({
                "dataset": DATASET_NAME,
                "repeat": int(repeat),
                "cv_split_seed": int(outer_seed),
                "fold": int(fold),
                "inner_split_seed": int(inner_seed),
                "sample_index_after_y_filter": int(index),
                "sample_id": str(sample_ids[index]),
                "outer_split": "train",
                "inner_split": "inner_train",
            })
        for index in inner_validation:
            rows.append({
                "dataset": DATASET_NAME,
                "repeat": int(repeat),
                "cv_split_seed": int(outer_seed),
                "fold": int(fold),
                "inner_split_seed": int(inner_seed),
                "sample_index_after_y_filter": int(index),
                "sample_id": str(sample_ids[index]),
                "outer_split": "train",
                "inner_split": "inner_validation",
            })

    if not rows:
        raise RuntimeError("No inner-validation split rows were created.")
    return pd.DataFrame(rows).sort_values(
        ["repeat", "cv_split_seed", "fold", "inner_split", "sample_index_after_y_filter"]
    ).reset_index(drop=True)


def audit_inner_validation_splits(
    outer_splits: pd.DataFrame,
    inner_splits: pd.DataFrame,
    n_samples: int,
    validation_fraction: float,
) -> pd.DataFrame:
    group_columns = ["repeat", "cv_split_seed", "fold"]
    expected_groups = set(
        tuple(row) for row in outer_splits.loc[:, group_columns].drop_duplicates().to_numpy()
    )
    observed_groups = set(
        tuple(row) for row in inner_splits.loc[:, group_columns].drop_duplicates().to_numpy()
    )
    if observed_groups != expected_groups:
        raise RuntimeError("Inner-validation fold groups do not exactly match the outer CV fold groups.")

    audit_rows: List[Dict[str, object]] = []
    for repeat, outer_seed, fold in sorted(expected_groups):
        outer_fold = outer_splits[
            (outer_splits["repeat"] == repeat)
            & (outer_splits["cv_split_seed"] == outer_seed)
            & (outer_splits["fold"] == fold)
        ]
        inner_fold = inner_splits[
            (inner_splits["repeat"] == repeat)
            & (inner_splits["cv_split_seed"] == outer_seed)
            & (inner_splits["fold"] == fold)
        ]

        outer_train = set(
            outer_fold.loc[outer_fold["split"] == "train", "sample_index_after_y_filter"].tolist()
        )
        outer_test = set(
            outer_fold.loc[outer_fold["split"] == "test", "sample_index_after_y_filter"].tolist()
        )
        inner_train = set(
            inner_fold.loc[
                inner_fold["inner_split"] == "inner_train", "sample_index_after_y_filter"
            ].tolist()
        )
        inner_validation = set(
            inner_fold.loc[
                inner_fold["inner_split"] == "inner_validation", "sample_index_after_y_filter"
            ].tolist()
        )

        expected_inner_seed = derive_inner_seed(int(repeat), int(outer_seed), int(fold))
        # Use Python integers rather than NumPy's platform-dependent ``int``.
        # On Windows, ``astype(int)`` can use signed 32-bit integers and can
        # overflow for large unsigned 32-bit seed values.
        observed_seed_values = {int(value) for value in inner_fold["inner_split_seed"].tolist()}
        seed_correct = observed_seed_values == {int(expected_inner_seed)}
        seed_within_portable_range = all(1 <= int(value) <= (2**31 - 2) for value in observed_seed_values)
        no_inner_train_validation_overlap = not bool(inner_train.intersection(inner_validation))
        no_outer_test_in_inner = not bool((inner_train | inner_validation).intersection(outer_test))
        inner_union_equals_outer_train = (inner_train | inner_validation) == outer_train
        no_duplicate_inner_assignments = not bool(
            inner_fold.duplicated(subset=["sample_index_after_y_filter"]).any()
        )
        expected_validation_count = inner_validation_count(
            len(outer_train), float(validation_fraction)
        )
        validation_size_correct = len(inner_validation) == expected_validation_count
        counts_add_up = len(inner_train) + len(inner_validation) == len(outer_train)

        # Audit exact deterministic reconstruction, not merely set properties.
        expected_inner_train, expected_inner_validation = deterministic_inner_partition(
            np.asarray(sorted(outer_train), dtype=int),
            expected_inner_seed,
            float(validation_fraction),
        )
        deterministic_reconstruction_matches = (
            inner_train == set(expected_inner_train.tolist())
            and inner_validation == set(expected_inner_validation.tolist())
        )

        audit_rows.append({
            "dataset": DATASET_NAME,
            "repeat": int(repeat),
            "cv_split_seed": int(outer_seed),
            "fold": int(fold),
            "n_all_samples": int(n_samples),
            "n_outer_train": int(len(outer_train)),
            "n_outer_test": int(len(outer_test)),
            "n_inner_train": int(len(inner_train)),
            "n_inner_validation": int(len(inner_validation)),
            "inner_validation_fraction": float(validation_fraction),
            "inner_split_seed": int(expected_inner_seed),
            "inner_split_seed_correct": bool(seed_correct),
            "inner_split_seed_within_portable_range": bool(seed_within_portable_range),
            "no_inner_train_validation_overlap": bool(no_inner_train_validation_overlap),
            "no_outer_test_in_inner_selection": bool(no_outer_test_in_inner),
            "inner_union_equals_outer_train": bool(inner_union_equals_outer_train),
            "no_duplicate_inner_assignments": bool(no_duplicate_inner_assignments),
            "inner_validation_size_expected": bool(validation_size_correct),
            "inner_train_plus_validation_equals_outer_train": bool(counts_add_up),
            "deterministic_reconstruction_matches": bool(deterministic_reconstruction_matches),
            "pass": bool(
                seed_correct
                and seed_within_portable_range
                and no_inner_train_validation_overlap
                and no_outer_test_in_inner
                and inner_union_equals_outer_train
                and no_duplicate_inner_assignments
                and validation_size_correct
                and counts_add_up
                and deterministic_reconstruction_matches
            ),
        })

    return pd.DataFrame(audit_rows).sort_values(
        ["repeat", "cv_split_seed", "fold"]
    ).reset_index(drop=True)


# =============================================================================
# 6. ARGUMENTS AND MAIN
# =============================================================================



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the final locked Wheat2000 shared outer+inner CV protocol."
    )
    parser.add_argument("--genotype-path", type=Path, default=GENOTYPE_PATH)
    parser.add_argument("--phenotype-dir", type=Path, default=PHENOTYPE_DIR)
    parser.add_argument("--protocol-dir", type=Path, default=FINAL_PROTOCOL_DIR)
    parser.add_argument(
        "--inner-validation-fraction",
        type=float,
        default=INNER_VALIDATION_FRACTION,
        help="Fraction of each outer-training fold reserved for shared inner validation (default: 0.10).",
    )
    parser.add_argument(
        "--overwrite-protocol",
        action="store_true",
        help=(
            "Replace an existing protocol directory. Use ONLY before any final model run has started. "
            "Never use this after benchmark outputs exist."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_dir = args.protocol_dir

    output_paths = {
        "outer_splits": protocol_dir / "outer_splits.csv",
        "inner_splits": protocol_dir / "inner_splits.csv",
        "fingerprints": protocol_dir / "fingerprints.json",
    }

    existing_outputs = [path for path in output_paths.values() if path.exists()]
    if existing_outputs and not args.overwrite_protocol:
        raise FileExistsError(
            "Final protocol already exists. Do not recreate it after final model runs have begun. "
            "Existing file: {0}".format(existing_outputs[0])
        )
    if existing_outputs and args.overwrite_protocol:
        for path in existing_outputs:
            path.unlink()

    # Validate everything before writing any protocol file.
    X_raw, sample_ids, marker_names, genotype_report = load_and_validate_genotype(args.genotype_path)
    phenotype_reports, alignment_audit = validate_phenotype_files(args.phenotype_dir, sample_ids)

    outer_splits = create_repeated_outer_splits(sample_ids)
    outer_audit = audit_outer_splits(outer_splits, n_samples=X_raw.shape[0])
    inner_splits = create_inner_validation_splits(
        outer_splits,
        sample_ids,
        validation_fraction=float(args.inner_validation_fraction),
    )
    inner_audit = audit_inner_validation_splits(
        outer_splits,
        inner_splits,
        n_samples=X_raw.shape[0],
        validation_fraction=float(args.inner_validation_fraction),
    )
    if not bool(inner_audit["pass"].all()):
        failed = inner_audit.loc[~inner_audit["pass"]]
        diagnostic_columns = [
            "repeat", "cv_split_seed", "fold", "n_outer_train", "n_outer_test",
            "n_inner_train", "n_inner_validation", "inner_split_seed_correct",
            "inner_split_seed_within_portable_range",
            "no_inner_train_validation_overlap", "no_outer_test_in_inner_selection",
            "inner_union_equals_outer_train", "no_duplicate_inner_assignments",
            "inner_validation_size_expected", "inner_train_plus_validation_equals_outer_train",
            "deterministic_reconstruction_matches",
        ]
        raise RuntimeError(
            "Inner-validation audit failed before writing protocol files. Failed rows:\n{0}".format(
                failed.loc[:, diagnostic_columns].to_string(index=False)
            )
        )

    # All validation succeeded. Now write the locked protocol files.
    ensure_directory(protocol_dir)
    outer_splits.to_csv(output_paths["outer_splits"], index=False, encoding="utf-8-sig")
    inner_splits.to_csv(output_paths["inner_splits"], index=False, encoding="utf-8-sig")

    outer_fingerprint_columns = list(OUTER_SPLIT_COLUMNS)
    inner_fingerprint_columns = [
        "repeat", "cv_split_seed", "fold", "inner_split_seed",
        "sample_index_after_y_filter", "sample_id", "outer_split", "inner_split",
    ]
    fingerprints: Dict[str, object] = {
        "protocol_version": "Wheat2000_Outer3x10_InnerVal_v5",
        "dataset": DATASET_NAME,
        **genotype_report,
        "marker_names_count": int(len(marker_names)),
        "outer_cv_protocol": "newly created repeated shuffled 10-fold outer CV; seeds 42, 2024, 2025",
        "cv_split_strategy": "random shuffled KFold; not stratified; not pedigree/family based",
        "cv_seeds": list(OUTER_CV_SEEDS),
        "n_outer_folds": int(N_OUTER_FOLDS),
        "n_outer_evaluations_per_trait": int(len(OUTER_CV_SEEDS) * N_OUTER_FOLDS),
        "inner_validation_protocol": "one deterministic holdout inside every outer-training fold",
        "inner_validation_fraction": float(args.inner_validation_fraction),
        "canonical_outer_split_path": str(output_paths["outer_splits"]),
        "canonical_outer_split_file_sha256": sha256_file(output_paths["outer_splits"]),
        "outer_split_fingerprint_sha256": stable_dataframe_sha256(
            outer_splits, outer_fingerprint_columns
        ),
        "canonical_inner_split_path": str(output_paths["inner_splits"]),
        "canonical_inner_split_file_sha256": sha256_file(output_paths["inner_splits"]),
        "inner_split_fingerprint_sha256": stable_dataframe_sha256(
            inner_splits, inner_fingerprint_columns
        ),
        "phenotypes": phenotype_reports,
    }
    # Store audit status inside the fingerprint file so GitHub needs only the
    # three canonical public protocol artifacts requested for this dataset.
    fingerprints["all_outer_audits_pass"] = bool(outer_audit["pass"].all())
    fingerprints["all_inner_audits_pass"] = bool(inner_audit["pass"].all())
    fingerprints["public_protocol_files"] = {
        "outer_splits": str(output_paths["outer_splits"]),
        "inner_splits": str(output_paths["inner_splits"]),
        "fingerprints": str(output_paths["fingerprints"]),
    }
    write_json(fingerprints, output_paths["fingerprints"])

    first_audit_row = inner_audit.iloc[0]
    print("=" * 88)
    print("SUCCESS: Final Wheat2000 outer + inner validation protocol is locked.")
    print("Protocol folder:")
    print("  {0}".format(protocol_dir))
    print("Samples: {0}; markers as model inputs: {1}".format(X_raw.shape[0], X_raw.shape[1]))
    print("Outer evaluations per trait: {0} (3 repeats x 10 folds)".format(
        len(OUTER_CV_SEEDS) * N_OUTER_FOLDS
    ))
    print("Per outer fold: outer train={0}, outer test={1}, inner train={2}, inner validation={3}".format(
        int(first_audit_row["n_outer_train"]),
        int(first_audit_row["n_outer_test"]),
        int(first_audit_row["n_inner_train"]),
        int(first_audit_row["n_inner_validation"]),
    ))
    print("All outer audits passed: {0}".format(bool(outer_audit["pass"].all())))
    print("All inner audits passed: {0}".format(bool(inner_audit["pass"].all())))
    print("Next step: update every final model runner to read BOTH split CSV files from this folder.")


if __name__ == "__main__":
    main()

