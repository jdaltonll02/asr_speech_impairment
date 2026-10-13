"""
Core decode-and-measure harness for the CDLI Rwandan NSS ASR sprint.

This is the *only* module that loads Whisper models and calls ``model.generate()``.
All other analysis modules (``failure.py``, ``regression.py``, ``interventions.py``)
read the Parquet files produced here and never load the model themselves.

Incremental checkpointing: results are flushed to disk every
``checkpoint_every`` utterances so Slurm preemption does not lose work.  On
resume the harness reads the checkpoint file, identifies the last processed
utterance, and skips forward in the dataset.

Key public functions
--------------------
load_model(model_id, cache_dir)
    Load a Whisper model + processor onto GPU (fp16) or CPU (fp32).

transcribe_dataset(model, processor, dataset_df, language, decode_kwargs,
                   batch_size, results_root)
    Iterate a dataset DataFrame, decode each utterance, compute per-utterance
    metrics, and return the results DataFrame.

decode_single(waveform, sr, model, processor, generate_kwargs)
    Decode one waveform; returns a dict with ``hypothesis``,
    ``compression_ratio``, and ``avg_log_prob``.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

# Heavy imports (numpy, pandas, torch, audio) are deferred to function bodies
# so this module can be imported for inspection/testing without GPU or full
# scientific stack installed.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, cache_dir: str):
    """Load a WhisperForConditionalGeneration model and processor.

    The model is loaded in float16 on CUDA if available, otherwise float32 on
    CPU.

    Parameters
    ----------
    model_id:
        HuggingFace model ID, e.g.
        ``'cdli/whisper-large-v3_finetuned_rwandan_kinyarwanda_...'``.
    cache_dir:
        Local directory to cache downloaded model files.

    Returns
    -------
    tuple[WhisperForConditionalGeneration, WhisperProcessor]
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("Loading model %s on %s (%s)...", model_id, device, dtype)
    processor = WhisperProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=cache_dir
    )
    model = model.to(device)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Single-utterance decode
# ---------------------------------------------------------------------------

def decode_single(
    waveform,
    sr: int,
    model,
    processor,
    generate_kwargs: dict,
) -> dict:
    """Decode a single audio waveform with a Whisper model.

    Parameters
    ----------
    waveform:
        Mono float32 audio at ``sr`` Hz.
    sr:
        Sample rate.  Resampled to 16 kHz internally if needed.
    model:
        WhisperForConditionalGeneration model.
    processor:
        WhisperProcessor (feature extractor + tokenizer).
    generate_kwargs:
        Keyword arguments forwarded to ``model.generate()``.

    Returns
    -------
    dict
        Keys: ``hypothesis`` (str), ``compression_ratio`` (float),
        ``avg_log_prob`` (float or None).
    """
    import torch
    import numpy as np
    from src.utils import audio as audio_mod

    # Ensure 16 kHz
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Compute compression ratio before generate()
    compression_ratio = audio_mod.compute_compression_ratio(waveform, sr)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    inputs = processor.feature_extractor(
        waveform, sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs.input_features.to(device=device, dtype=dtype)

    with torch.no_grad():
        outputs = model.generate(
            input_features,
            return_dict_in_generate=True,
            output_scores=True,
            **generate_kwargs,
        )

    # Decode tokens
    hypothesis = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

    # Extract avg_log_prob from sequences_scores
    avg_log_prob: float | None = None
    if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
        seq_score = outputs.sequences_scores[0].item()
        # sequences_scores is the sum of log-probs; normalise by output length
        output_len = outputs.sequences[0].shape[-1] - 1  # exclude initial BOS
        avg_log_prob = seq_score / max(output_len, 1)

    return {
        "hypothesis": hypothesis,
        "compression_ratio": compression_ratio,
        "avg_log_prob": avg_log_prob,
    }


# ---------------------------------------------------------------------------
# Batch transcription
# ---------------------------------------------------------------------------

def transcribe_dataset(
    model,
    processor,
    dataset_df,
    language: str,
    decode_kwargs: dict,
    batch_size: int,
    results_root: str,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 50,
    prior_rows: list[dict] | None = None,
):
    """Transcribe all utterances in *dataset_df* and return a results DataFrame.

    Iterates *dataset_df* in batches, calls ``decode_single`` for each
    utterance, computes per-utterance metrics, and writes incremental Parquet
    checkpoints to survive Slurm preemption.

    Parameters
    ----------
    model:
        Loaded WhisperForConditionalGeneration model.
    processor:
        WhisperProcessor.
    dataset_df:
        DataFrame with columns including ``audio_path``, ``transcription``
        (or ``sentence``), and all per-utterance metadata fields.
    language:
        Language code (``'ky'`` or ``'en'``).
    decode_kwargs:
        Keyword arguments forwarded to ``model.generate()`` (e.g. language
        token, beam size, temperature).
    batch_size:
        Number of utterances processed per batch.
    results_root:
        Root directory for results; checkpoint files are written here.
    checkpoint_path:
        Explicit path for the checkpoint Parquet.  If None, derived from
        *results_root*.
    checkpoint_every:
        Flush checkpoint to disk after every this many utterances.
    prior_rows:
        Rows already completed in a previous run (from a checkpoint).  These
        are prepended to the output so the returned DataFrame is complete.

    Returns
    -------
    pd.DataFrame
        Per-utterance results with all columns from ``metrics.SCHEMA``.
    """
    import numpy as np
    import pandas as pd
    from src.eval import metrics as metrics_mod
    from src.eval import normalise as norm_mod
    from src.utils import audio as audio_mod

    rows: list[dict] = list(prior_rows) if prior_rows else []
    last_saved_index = -1

    # Determine reference text column
    ref_col = "transcription" if "transcription" in dataset_df.columns else "sentence"

    # Determine utterance ID column
    id_col = None
    for c in ("utt_id", "utterance_id", "id"):
        if c in dataset_df.columns:
            id_col = c
            break

    # Set up SIGUSR1 handler for preemption-safe checkpointing
    _flush_requested = [False]

    def _handle_sigusr1(signum, frame):
        _flush_requested[0] = True
        logger.warning("Received SIGUSR1 — will flush checkpoint at next opportunity.")

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handle_sigusr1)

    total = len(dataset_df)
    logger.info("Transcribing %d utterances (batch_size=%d)...", total, batch_size)

    for idx, row_data in enumerate(dataset_df.itertuples(index=False)):
        row_dict = row_data._asdict()

        utt_id = (
            str(row_dict.get(id_col, idx)) if id_col else f"{language}_{idx:06d}"
        )
        audio_path = str(row_dict.get("audio_path", ""))
        reference = str(row_dict.get(ref_col, ""))

        # Load and decode audio
        try:
            waveform, sr = audio_mod.load_audio(audio_path)
        except Exception as exc:
            logger.warning("Failed to load audio %s: %s", audio_path, exc)
            import numpy as np
            waveform = np.zeros(16000, dtype=np.float32)
            sr = 16000

        audio_length_sec = len(waveform) / sr

        try:
            decode_result = decode_single(waveform, sr, model, processor, decode_kwargs)
        except Exception as exc:
            logger.warning("Decode failed for %s: %s", utt_id, exc)
            decode_result = {
                "hypothesis": "",
                "compression_ratio": audio_mod.compute_compression_ratio(waveform, sr),
                "avg_log_prob": None,
            }

        hypothesis = decode_result["hypothesis"]
        compression_ratio = decode_result["compression_ratio"]
        avg_log_prob = decode_result["avg_log_prob"]

        # Normalise
        ref_norm = norm_mod.normalize(reference, language)
        hyp_norm = norm_mod.normalize(hypothesis, language)

        # Metrics
        errs = metrics_mod.word_errors(ref_norm, hyp_norm)
        wer_cap = metrics_mod.per_utterance_wer(ref_norm, hyp_norm)
        wer_uncap = metrics_mod.uncapped_wer(ref_norm, hyp_norm)
        cer_val = metrics_mod.cer(ref_norm, hyp_norm)
        rep_rate = metrics_mod.repeated_ngram_rate(hyp_norm, n=4)

        # Audio features
        try:
            leading_s, trailing_s = audio_mod.leading_trailing_silence(audio_path)
        except Exception:
            leading_s, trailing_s = 0.0, 0.0

        try:
            snr_db = audio_mod.estimate_snr(audio_path)
        except Exception:
            snr_db = 0.0

        result_row = {
            "utt_id": utt_id,
            "split": str(row_dict.get("split", "")),
            "model": language.upper(),
            "audio_path": audio_path,
            "audio_length_sec": audio_length_sec,
            "reference": reference,
            "hypothesis": hypothesis,
            "reference_normalised": ref_norm,
            "hypothesis_normalised": hyp_norm,
            "n_ref_words": errs["total_ref_words"],
            "n_hyp_words": errs["total_hyp_words"],
            "substitutions": errs["substitutions"],
            "deletions": errs["deletions"],
            "insertions": errs["insertions"],
            "wer_capped": wer_cap,
            "wer_uncapped": wer_uncap,
            "cer": cer_val,
            "length_ratio": errs["total_hyp_words"] / max(errs["total_ref_words"], 1),
            "compression_ratio": compression_ratio,
            "avg_log_prob": avg_log_prob if avg_log_prob is not None else float("nan"),
            "repeated_4gram_rate": rep_rate,
            "is_catastrophic": False,  # set by failure.py
            "prompt_type": str(row_dict.get("prompt_type", "")),
            "recording_environment": str(row_dict.get("recording_environment", "")),
            "speaker_id": str(row_dict.get("speaker_id", "")),
            "leading_silence_s": leading_s,
            "trailing_silence_s": trailing_s,
            "snr_db": snr_db,
        }
        rows.append(result_row)

        # Incremental checkpoint
        should_flush = (
            (idx + 1) % checkpoint_every == 0
            or idx == total - 1
            or _flush_requested[0]
        )
        if should_flush and checkpoint_path and rows:
            _write_checkpoint(rows, checkpoint_path)
            last_saved_index = idx
            _flush_requested[0] = False
            logger.info("Checkpoint written at utterance %d/%d", idx + 1, total)

        if (idx + 1) % 50 == 0:
            logger.info("Progress: %d/%d utterances", idx + 1, total)

    import pandas as pd
    return pd.DataFrame(rows)


def _write_checkpoint(rows: list[dict], path: str) -> None:
    """Write a list of result dicts to a Parquet checkpoint file."""
    import pandas as pd

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
