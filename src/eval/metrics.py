"""
Per-utterance and corpus-level ASR metrics for the CDLI Rwandan NSS sprint.

Functions receive *already-normalised* strings (``normalise.py`` handles
normalisation before these are called).

Public API
----------
word_errors(ref, hyp)       → dict with substitutions/deletions/insertions counts
corpus_wer(refs, hyps)      → float (aggregate WER across all utterances)
per_utterance_wer(ref, hyp) → float (capped at ``cap``, default 1.0)
uncapped_wer(ref, hyp)      → float
cer(ref, hyp)               → float (character error rate)
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
