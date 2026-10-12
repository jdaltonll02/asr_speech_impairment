"""
Text normalisation for CDLI Rwandan NSS ASR evaluation.

Applies normalisation identically to reference transcripts and model hypotheses
before any metric computation, following CDLI published rules:

- English (``language='en'``): Apply Whisper BasicTextNormalizer (lowercases,
  removes punctuation, collapses whitespace).
- Kinyarwanda (``language='ky'`` or ``'rw'``): Apply BasicTextNormalizer, then
  unify apostrophe glyphs (U+2019 RIGHT SINGLE QUOTATION MARK, U+02BC MODIFIER
  LETTER APOSTROPHE → ASCII apostrophe), then remove elision apostrophes between
  word characters so that ``n'abantu`` becomes ``nabantu``.

Usage::

    from src.eval.normalise import normalize
    ref_norm = normalize(reference, language='ky')
    hyp_norm = normalize(hypothesis, language='ky')
"""

from __future__ import annotations

import re
import string
import unicodedata

# Try to use Whisper's BasicTextNormalizer; fall back to a pure-Python
# equivalent if the openai-whisper package or its dependencies are unavailable.
try:
    from whisper.normalizers import BasicTextNormalizer as _WhisperNormalizer

    _normalizer = _WhisperNormalizer()

    def _basic_normalize(text: str) -> str:
        return _normalizer(text)

except Exception:
    # Pure-Python fallback: lowercase + remove punctuation + collapse whitespace
    _PUNCT_TABLE = str.maketrans("", "", string.punctuation)

    def _basic_normalize(text: str) -> str:  # type: ignore[misc]
        text = text.lower()
        text = text.translate(_PUNCT_TABLE)
        text = " ".join(text.split())
        return text


def normalize(text: str | None, language: str) -> str:
    """Normalise *text* for the given *language*.

    Parameters
    ----------
    text:
        Raw transcript string.  ``None`` or empty string returns ``""``.
    language:
        Language code: ``'en'`` for English, ``'ky'`` or ``'rw'`` for
        Kinyarwanda (both accepted).

    Returns
    -------
    str
        Normalised text ready for WER/CER computation.
    """
    if not text:
        return ""

    # Apply Whisper BasicTextNormalizer (shared for all languages)
    text = _basic_normalize(text)

    if language in ("ky", "rw"):
        # Unify apostrophe glyphs to ASCII apostrophe
        text = text.replace("\u2019", "'").replace("\u02bc", "'")
        # Remove elision apostrophe between word characters: n'abantu → nabantu
        text = re.sub(r"(\w)'(\w)", r"\1\2", text)

    # Final whitespace collapse (BasicTextNormalizer should already do this,
    # but apply again after Kinyarwanda transforms)
    text = " ".join(text.split())
    return text
