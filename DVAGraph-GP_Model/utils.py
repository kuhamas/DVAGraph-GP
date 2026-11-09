"""
DVAGraph-GP shared utilities.

Seed protocol (preserved from the original OurModel_1.py):
- Outer CV repeat seeds are configured in config.py (Wheat2000: 42, 2024, 2025).
- The locked inner_split_seed is read from the existing split file by data.py.
- For each fold:
      base = cv_seed * 10000 + fold * 101
      selection_seed = base + 17
      refit_seed = base + 37
- Each training stage seeds Python, NumPy, PyTorch CPU/CUDA, and its
  DataLoader generator with that stage's own seed.

This module never creates or changes CV partitions.
"""

from __future__ import annotations

import os

# Must be set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import json
import logging
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# 1. Random-seed control
# ---------------------------------------------------------------------------

def set_global_seed(
    seed: int,
    settings: Optional[Any] = None,
    *,
    deterministic_torch: Optional[bool] = None,
) -> None:
    """
    Seed Python, NumPy, PyTorch CPU and CUDA for one training stage.

    The settings argument preserves train.py's call style:
        set_global_seed(stage_seed, settings)

    It does not generate CV splits or derive the stage seed.
    """
    seed = int(seed)

    if deterministic_torch is None:
        deterministic_torch = bool(
            getattr(settings, "deterministic_torch", True)
            if settings is not None else True
        )
    else:
        deterministic_torch = bool(deterministic_torch)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_torch

    # Same warning-only deterministic-algorithm policy as the original runner.
    try:
        torch.use_deterministic_algorithms(
            deterministic_torch,
            warn_only=True,
        )
    except Exception:
        pass


def derive_stage_seeds(
    spec: Any,
    settings: Optional[Any] = None,
) -> Tuple[int, int]:
    """
    Return (selection_seed, refit_seed) using the original formula.

    Uses spec.cv_seed and spec.fold only.
    It deliberately does not use spec.repeat or spec.inner_split_seed:
    the latter belongs to the locked split protocol, not model initialization.
    """
    cv_multiplier = int(
        getattr(settings, "cv_seed_multiplier", 10000)
        if settings is not None else 10000
    )
    fold_multiplier = int(
        getattr(settings, "fold_seed_multiplier", 101)
        if settings is not None else 101
    )
    selection_offset = int(
        getattr(settings, "selection_seed_offset", 17)
        if settings is not None else 17
    )
    refit_offset = int(
        getattr(settings, "refit_seed_offset", 37)
        if settings is not None else 37
    )

    base = int(spec.cv_seed) * cv_multiplier + int(spec.fold) * fold_multiplier
    selection_seed = int(base + selection_offset)
    refit_seed = int(base + refit_offset)

    if selection_seed == refit_seed:
        raise ValueError("Selection and refit seeds must be different.")

    return selection_seed, refit_seed


def make_torch_generator(seed: int) -> torch.Generator:
    """Create the CPU generator used for deterministic DataLoader shuffling."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def make_numpy_generator(seed: int) -> np.random.Generator:
    """
    Create an independent NumPy generator for a separately seeded analysis.

    Do not use it to regenerate the locked CV splits.
    """
    return np.random.default_rng(int(seed))


# ---------------------------------------------------------------------------
# 2. File and directory utilities
# ---------------------------------------------------------------------------

def ensure_directory(path: Union[str, Path]) -> Path:
    """Create a directory and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _atomic_write(path: Union[str, Path], writer: Any, encoding: str) -> None:
    """Write a file in its destination directory, then atomically replace it."""
    destination = Path(path)
    ensure_directory(destination.parent)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer(handle)

        os.replace(temporary_path, destination)

    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def save_csv(
    frame: pd.DataFrame,
    path: Union[str, Path],
    *,
    encoding: str = "utf-8-sig",
) -> None:
    """
    Save a DataFrame atomically.

    UTF-8 with BOM is the default to remain compatible with the original
    runner's CSV outputs and spreadsheet applications.
    """
    _atomic_write(
        path,
        lambda handle: frame.to_csv(handle, index=False),
        encoding,
    )


def _json_safe(value: Any) -> Any:
    """Convert NumPy, Pandas, Path, and non-finite values to valid JSON data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def save_json(payload: Mapping[str, Any], path: Union[str, Path]) -> None:
    """Save a JSON mapping atomically, including NumPy/Pandas values."""
    safe_payload = _json_safe(dict(payload))

    def writer(handle: Any) -> None:
        json.dump(
            safe_payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    _atomic_write(path, writer, "utf-8")


def safe_filename(value: Any, fallback: str = "unnamed") -> str:
    """Convert a label to a filesystem-safe filename component."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return name or fallback


# ---------------------------------------------------------------------------
# 3. SHA-256 fingerprints (same DataFrame algorithm as original runner)
# ---------------------------------------------------------------------------

def sha256_file(
    path: Union[str, Path],
    chunk_size: int = 1024 * 1024,
) -> str:
    """Hash the exact bytes of a file using streaming reads."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Sequence[Any]) -> str:
    """Hash newline-joined string representations, matching the original."""
    text = "\n".join(str(value) for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_dataframe_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    """
    Reproduce the original runner's stable DataFrame fingerprint exactly:
    select columns, sort by those columns, reset index, to_csv(index=False),
    then SHA-256 hash the UTF-8 serialization.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing fingerprint columns: {missing}")

    stable = (
        frame.loc[:, list(columns)]
        .copy()
        .sort_values(list(columns))
        .reset_index(drop=True)
    )
    serialized = stable.to_csv(index=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 4. Logging and device synchronization
# ---------------------------------------------------------------------------

def get_logger(
    name: str = "DVAGraph-GP",
    *,
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
) -> logging.Logger:
    """Create/retrieve a logger without attaching duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    if console and not any(
        getattr(handler, "_dvagraph_kind", None) == "console"
        for handler in logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler._dvagraph_kind = "console"
        logger.addHandler(handler)

    if log_file is not None:
        file_path = Path(log_file)
        ensure_directory(file_path.parent)
        resolved = str(file_path.resolve())

        if not any(
            getattr(handler, "_dvagraph_log_path", None) == resolved
            for handler in logger.handlers
        ):
            handler = logging.FileHandler(
                file_path,
                mode="a",
                encoding="utf-8",
            )
            handler.setFormatter(formatter)
            handler._dvagraph_log_path = resolved
            logger.addHandler(handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """Flush, close, and detach all handlers from a logger."""
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)


def sync_device(
    device: Optional[Union[str, torch.device]] = None,
) -> None:
    """Synchronize CUDA before timing; this is a no-op for CPU execution."""
    if not torch.cuda.is_available():
        return

    if device is None:
        torch.cuda.synchronize()
    else:
        resolved = torch.device(device)
        if resolved.type == "cuda":
            torch.cuda.synchronize(resolved)


__all__ = [
    "set_global_seed",
    "derive_stage_seeds",
    "make_torch_generator",
    "make_numpy_generator",
    "ensure_directory",
    "save_csv",
    "save_json",
    "safe_filename",
    "sha256_file",
    "sha256_strings",
    "stable_dataframe_sha256",
    "get_logger",
    "close_logger",
    "sync_device",
]
