"""
Dataset loading utilities for the CDLI Rwandan NSS ASR sprint.

Handles local data (TSV/CSV splits + speaker_metadata.csv) or HuggingFace
gated datasets.  If the local data directory does not exist or the expected
split files are absent, raises ``DataNotReadyError`` with a message describing
blocker I01 (data access pending).

Usage::

    from src.utils.data import load_split, validate_data_root
    validate_data_root("/data/user_data/jgibson2/asr_sprint/data")
    df = load_split("dev", "ky", "/data/user_data/jgibson2/asr_sprint/data")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Blocker error
# ---------------------------------------------------------------------------

class DataNotReadyError(Exception):
    """Raised when the local data directory is missing or incomplete.

    Blocker I01: The CDLI Rwandan NSS datasets are gated on HuggingFace.
    Accept terms at:
      https://huggingface.co/datasets/cdli/rwandan_kinyarwanda_nonstandard_speech_v1.0
      https://huggingface.co/datasets/cdli/rwandan_english_nonstandard_speech_v1.0
    Then set HF_TOKEN and run the download script, or place pre-downloaded
    data under the configured data_root directory.
    """


# ---------------------------------------------------------------------------
# Expected split files
# ---------------------------------------------------------------------------

# Map (language, split_name) → filename stem candidates
_SPLIT_STEMS = {
    ("ky", "train"): ["ky_train", "kinyarwanda_train"],
    ("ky", "dev"):   ["ky_dev", "kinyarwanda_dev", "ky_validation"],
    ("ky", "test"):  ["ky_test", "kinyarwanda_test"],
    ("en", "train"): ["en_train", "english_train"],
    ("en", "dev"):   ["en_dev", "english_dev", "en_validation"],
    ("en", "test"):  ["en_test", "english_test"],
}

_EXTENSIONS = [".tsv", ".csv"]

SPEAKER_METADATA_FILE = "speaker_metadata.csv"


def validate_data_root(data_root: str) -> None:
    """Assert that *data_root* exists and contains the expected split files.

    Raises
    ------
    DataNotReadyError
        If the directory is absent or no recognised split files are found.
    """
    root = Path(data_root)
    if not root.exists():
        raise DataNotReadyError(
            f"Data directory not found: {root}\n"
            "Blocker I01: accept CDLI HuggingFace dataset terms and set HF_TOKEN "
            "before downloading data."
        )

    found: list[str] = []
    for stems in _SPLIT_STEMS.values():
        for stem in stems:
            for ext in _EXTENSIONS:
                if (root / f"{stem}{ext}").exists():
                    found.append(f"{stem}{ext}")
                    break

    if not found:
        raise DataNotReadyError(
            f"No recognised split files found under {root}.\n"
            "Expected files like ky_train.tsv, en_dev.tsv, etc.\n"
            "Blocker I01: data access pending."
        )


def _find_split_file(language: str, split_name: str, data_root: Path) -> Path:
    """Locate the split file for *(language, split_name)* under *data_root*."""
    key = (language, split_name)
    stems = _SPLIT_STEMS.get(key, [f"{language}_{split_name}"])
    for stem in stems:
        for ext in _EXTENSIONS:
            candidate = data_root / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    raise DataNotReadyError(
        f"Split file for language='{language}', split='{split_name}' not found "
        f"under {data_root}.\n"
        "Blocker I01: data access pending."
    )


def load_split(split_name: str, language: str, data_root: str):
    """Load a dataset split and join speaker metadata.

    Parameters
    ----------
    split_name:
        One of ``'train'``, ``'dev'``, ``'test'``, ``'extra'``.
    language:
        ``'ky'`` (Kinyarwanda) or ``'en'`` (English).
    data_root:
        Path to the local data directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with all per-utterance fields plus speaker metadata columns
        (severity, etiology, gender, age_band) joined on ``speaker_id``.
    """
    import pandas as pd

    root = Path(data_root)
    split_file = _find_split_file(language, split_name, root)

    sep = "\t" if split_file.suffix == ".tsv" else ","
    df = pd.read_csv(split_file, sep=sep, dtype=str)

    # Normalise column names to lowercase and replace spaces with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Join speaker metadata if available
    meta_file = root / SPEAKER_METADATA_FILE
    if meta_file.exists():
        meta = pd.read_csv(meta_file, dtype=str)
        meta.columns = [c.strip().lower().replace(" ", "_") for c in meta.columns]
        if "speaker_id" in df.columns and "speaker_id" in meta.columns:
            df = df.merge(meta, on="speaker_id", how="left")

    # Add convenience columns
    df["language"] = language
    df["split"] = split_name

    return df


# ---------------------------------------------------------------------------
# Batch iteration for HuggingFace datasets (used when data is on HF)
# ---------------------------------------------------------------------------

def iter_batches(dataset, batch_size: int) -> Iterator[list[dict]]:
    """Yield lists of row dicts from a HuggingFace ``datasets.Dataset``.

    Each dict contains at least: ``utt_id``, ``audio_array``,
    ``sampling_rate``, ``reference``, ``speaker_id``, ``recording_env``.

    Parameters
    ----------
    dataset:
        A ``datasets.Dataset`` instance.
    batch_size:
        Number of examples per batch.
    """
    batch: list[dict] = []
    for i, row in enumerate(dataset):
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def utt_id_from_row(row: dict, model_name: str, index: int) -> str:
    """Construct a stable unique utterance ID.

    Uses the row's native ``id`` field if present, otherwise falls back to
    ``{model_name}_{index:06d}``.

    Parameters
    ----------
    row:
        A dataset row dict.
    model_name:
        Model alias (e.g. ``'ky'`` or ``'en'``).
    index:
        Dataset row index (used as fallback).
    """
    native_id = row.get("id") or row.get("utt_id") or row.get("utterance_id")
    if native_id:
        return f"{model_name}_{native_id}"
    return f"{model_name}_{index:06d}"


# ---------------------------------------------------------------------------
# HuggingFace dataset download (invoked only with --download flag)
# ---------------------------------------------------------------------------

def download_dataset(
    language: str,
    data_root: str,
    cfg=None,
    hf_token: str | None = None,
) -> None:
    """Download a CDLI Rwandan NSS dataset from HuggingFace to local storage.

    This function is **not** called automatically.  Invoke it explicitly with
    the ``--download`` flag in ``baseline.py`` once HF access has been granted
    and ``HF_TOKEN`` is set.

    Parameters
    ----------
    language:
        ``'ky'`` or ``'en'``.
    data_root:
        Local directory where split files will be written.
    cfg:
        Loaded config object.  If None, loaded automatically.
    hf_token:
        HuggingFace token string.  Defaults to the ``HF_TOKEN`` environment
        variable, then ``~/.cache/huggingface/token``.

    Raises
    ------
    RuntimeError
        If the HF token is missing or the dataset cannot be downloaded.
    """
    import os

    if cfg is None:
        from src.utils.config import load_config
        cfg = load_config()

    # Resolve token
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN not found. Set the HF_TOKEN environment variable or place "
            "your token in ~/.cache/huggingface/token before running --download."
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for downloading. "
            "Install it with: pip install datasets"
        ) from exc

    hf_id = cfg.datasets[language].hf_id
    split_map = cfg.datasets[language].split_map  # e.g. {dev: validation, test: test}
    audio_col = cfg.datasets[language].audio_column
    text_col = cfg.datasets[language].reference_column

    out_dir = Path(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_split_names = list(split_map.__dict__.items()) if hasattr(split_map, "__dict__") else split_map.items()

    for local_name, hf_split in hf_split_names:
        print(f"Downloading {hf_id} / {hf_split} → {local_name} ...")
        ds = load_dataset(hf_id, split=hf_split, token=token)

        out_path = out_dir / f"{language}_{local_name}.tsv"
        import pandas as pd
        rows = []
        for i, row in enumerate(ds):
            audio_info = row.get(audio_col, {})
            if isinstance(audio_info, dict):
                audio_path = audio_info.get("path", "")
            else:
                audio_path = str(audio_info)
            rows.append({
                "utt_id": row.get("id", f"{language}_{local_name}_{i:06d}"),
                "audio_path": audio_path,
                "transcription": row.get(text_col, ""),
                "speaker_id": row.get("speaker_id", ""),
                "prompt_type": row.get("prompt_type", ""),
                "recording_environment": row.get("recording_environment", ""),
                "recording_device": row.get("recording_device", ""),
                "audio_length": row.get("audio_length", ""),
                "transcript_length": row.get("transcript_length", ""),
                "transcriber_confidence": row.get("transcriber_confidence", ""),
            })
        pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
        print(f"  Saved {len(rows)} rows to {out_path}")

    print(f"\nDownload complete for language='{language}'. Data in {out_dir}")
