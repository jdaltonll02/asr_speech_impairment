"""
Per-utterance and corpus-level ASR metrics for the CDLI Rwandan NSS sprint.

Functions receive *already-normalised* strings (``normalise.py`` handles
normalisation before these are called).

Public API
----------
word_errors(ref, hyp)        → dict with substitutions/deletions/insertions counts
compute_edit_ops(ref, hyp)   → alias for word_errors (plan Step 4 name)
corpus_wer(refs, hyps)       → float (aggregate WER across all utterances)
per_utterance_wer(ref, hyp)  → float (capped at ``cap``, default 1.0)
uncapped_wer(ref, hyp)       → float
cer(ref, hyp)                → float (character error rate)
repeated_ngram_rate(text, n) → float (fraction of repeated n-grams)
build_row(utt_id, ...)       → dict matching SCHEMA (plan Step 4 canonical row builder)
SCHEMA                       → dict[str, str] dtype mapping for the canonical DataFrame
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import jiwer


# ---------------------------------------------------------------------------
# Canonical per-utterance DataFrame dtype mapping
# ---------------------------------------------------------------------------
SCHEMA: dict[str, str] = {
    "utt_id": "object",
    "split": "object",
    "model": "object",
    "audio_path": "object",
    "audio_length_sec": "float64",
    "reference": "object",
    "hypothesis": "object",
    "reference_normalised": "object",
    "hypothesis_normalised": "object",
    "n_ref_words": "int32",
    "n_hyp_words": "int32",
    "substitutions": "int32",
    "deletions": "int32",
    "insertions": "int32",
    "wer_capped": "float64",
    "wer_uncapped": "float64",
    "cer": "float64",
    "length_ratio": "float64",
    "compression_ratio": "float64",
    "avg_log_prob": "float64",
    "repeated_4gram_rate": "float64",
    "is_catastrophic": "bool",
    "prompt_type": "object",
    "recording_environment": "object",
    "speaker_id": "object",
    "leading_silence_s": "float64",
    "trailing_silence_s": "float64",
    "snr_db": "float64",
}

# (No module-level transform pipelines needed — process_words/process_characters
#  use their own defaults.)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def word_errors(ref: str, hyp: str) -> dict:
    """Compute word-level edit operations between *ref* and *hyp*.

    Parameters
    ----------
    ref:
        Normalised reference string.
    hyp:
        Normalised hypothesis string.

    Returns
    -------
    dict
        Keys: ``substitutions``, ``deletions``, ``insertions``,
        ``total_ref_words``, ``total_hyp_words``.
    """
    # Use non-empty strings so jiwer doesn't choke on empty input
    ref_safe = ref.strip() if ref.strip() else " "
    hyp_safe = hyp.strip() if hyp.strip() else " "
    result = jiwer.process_words(reference=ref_safe, hypothesis=hyp_safe)
    ref_words = ref.split() if ref.strip() else []
    hyp_words = hyp.split() if hyp.strip() else []
    return {
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
        "total_ref_words": len(ref_words),
        "total_hyp_words": len(hyp_words),
    }


def corpus_wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Compute corpus-level WER as total errors / total reference words.

    Parameters
    ----------
    refs:
        Sequence of normalised reference strings.
    hyps:
        Sequence of normalised hypothesis strings (same length as *refs*).

    Returns
    -------
    float
        Corpus WER.  Returns 0.0 if the reference corpus has no words.
    """
    total_errors = 0
    total_ref_words = 0
    for ref, hyp in zip(refs, hyps):
        errs = word_errors(ref, hyp)
        total_errors += errs["substitutions"] + errs["deletions"] + errs["insertions"]
        total_ref_words += errs["total_ref_words"]
    if total_ref_words == 0:
        return 0.0
    return total_errors / total_ref_words


def uncapped_wer(ref: str, hyp: str) -> float:
    """Return uncapped per-utterance WER: (S+D+I) / max(|ref|, 1)."""
    errs = word_errors(ref, hyp)
    total_errors = errs["substitutions"] + errs["deletions"] + errs["insertions"]
    denom = max(errs["total_ref_words"], 1)
    return total_errors / denom


def per_utterance_wer(ref: str, hyp: str, cap: float = 1.0) -> float:
    """Return per-utterance WER capped at *cap* (default 1.0)."""
    return min(uncapped_wer(ref, hyp), cap)


def cer(ref: str, hyp: str) -> float:
    """Return character error rate between *ref* and *hyp*.

    Uses character-level edit distance.  Returns 0.0 for empty reference.
    """
    if not ref.strip():
        return 0.0
    result = jiwer.process_characters(
        reference=ref if ref else " ",
        hypothesis=hyp if hyp else " ",
    )
    return result.cer


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    """Fraction of n-gram positions occupied by a repeated n-gram.

    A 4-gram is considered repeated if it appears more than once in *text*.
    Returns 0.0 for text with fewer than *n* words.

    Parameters
    ----------
    text:
        Whitespace-tokenised hypothesis text.
    n:
        N-gram size (default 4).
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated_positions = sum(count for count in counts.values() if count > 1)
    return repeated_positions / len(ngrams)


# Alias required by plan Step 4 public API
def compute_edit_ops(ref: str, hyp: str) -> dict:
    """Alias for ``word_errors`` using the plan's specified function name.

    Parameters
    ----------
    ref:
        Normalised reference string.
    hyp:
        Normalised hypothesis string.

    Returns
    -------
    dict
        Keys: ``substitutions``, ``deletions``, ``insertions``,
        ``total_ref_words``, ``total_hyp_words``.
    """
    return word_errors(ref, hyp)


def build_row(
    utt_id: str,
    split: str,
    model_name: str,
    audio_path: str,
    duration_s: float,
    ref: str,
    hyp: str,
    compression_ratio: float | None = None,
    avg_log_prob: float | None = None,
    prompt_type: str = "",
    recording_env: str = "",
    speaker_id: str = "",
    leading_silence_s: float = 0.0,
    trailing_silence_s: float = 0.0,
    snr_db: float = 0.0,
) -> dict:
    """Build a canonical per-utterance result row (matches ``SCHEMA``).

    Computes all metric fields from *ref* and *hyp* and assembles a flat dict
    whose keys match ``SCHEMA``.  Callers are expected to have already
    normalised *ref* and *hyp* via ``normalise.py``.

    Parameters
    ----------
    utt_id:
        Unique utterance identifier.
    split:
        Dataset split name (``'dev'``, ``'test'``, etc.).
    model_name:
        Model alias (``'KY'`` or ``'EN'``).
    audio_path:
        Absolute path to the audio file.
    duration_s:
        Audio duration in seconds.
    ref:
        Normalised reference string.
    hyp:
        Normalised hypothesis string.
    compression_ratio:
        Whisper compression ratio (from ``audio.compute_compression_ratio``).
    avg_log_prob:
        Mean token log-probability (from ``model.generate`` sequences_scores).
    prompt_type:
        Prompt type metadata field.
    recording_env:
        Recording environment metadata field.
    speaker_id:
        Speaker identifier.
    leading_silence_s:
        Leading silence duration in seconds.
    trailing_silence_s:
        Trailing silence duration in seconds.
    snr_db:
        Estimated signal-to-noise ratio in dB.

    Returns
    -------
    dict
        Flat dict with all keys from ``SCHEMA``.
    """
    import math

    errs = word_errors(ref, hyp)
    wer_cap = per_utterance_wer(ref, hyp)
    wer_uncap = uncapped_wer(ref, hyp)
    cer_val = cer(ref, hyp)
    rep_rate = repeated_ngram_rate(hyp, n=4)
    length_rat = errs["total_hyp_words"] / max(errs["total_ref_words"], 1)

    return {
        "utt_id": utt_id,
        "split": split,
        "model": model_name,
        "audio_path": audio_path,
        "audio_length_sec": float(duration_s),
        "reference": ref,
        "hypothesis": hyp,
        "reference_normalised": ref,
        "hypothesis_normalised": hyp,
        "n_ref_words": int(errs["total_ref_words"]),
        "n_hyp_words": int(errs["total_hyp_words"]),
        "substitutions": int(errs["substitutions"]),
        "deletions": int(errs["deletions"]),
        "insertions": int(errs["insertions"]),
        "wer_capped": float(wer_cap),
        "wer_uncapped": float(wer_uncap),
        "cer": float(cer_val),
        "length_ratio": float(length_rat),
        "compression_ratio": float(compression_ratio) if compression_ratio is not None else float("nan"),
        "avg_log_prob": float(avg_log_prob) if avg_log_prob is not None else float("nan"),
        "repeated_4gram_rate": float(rep_rate),
        "is_catastrophic": False,  # set downstream by failure.py
        "prompt_type": str(prompt_type),
        "recording_environment": str(recording_env),
        "speaker_id": str(speaker_id),
        "leading_silence_s": float(leading_silence_s),
        "trailing_silence_s": float(trailing_silence_s),
        "snr_db": float(snr_db),
    }
