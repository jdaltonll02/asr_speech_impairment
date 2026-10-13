"""
RQ1 Interventions — re-decode catastrophic utterances with six strategies.

Each intervention is a standalone function that modifies ``generate()`` kwargs
or applies pre/post-processing around ``harness.decode_single``.  Interventions
run independently (not stacked) on the subset of utterances flagged as
catastrophic by ``failure.py``.

Interventions
-------------
A  repetition_penalty    no_repeat_ngram_size=4
B  max_length_from_audio max_new_tokens ∝ audio duration (≈6 tokens/sec)
C  temperature_fallback  Whisper's built-in compression_ratio_threshold trigger
D  beam_search           num_beams=5
E  vad_trim              silero-VAD silence trimming before decode
F  ctc_fallback          TODO placeholder — CTC model not yet available for KY

Per-utterance results (with wer_before, wer_after, delta_wer) are saved as
    {results_root}/rq1/interventions/intervention_{name}_{lang}_{split}.parquet

An intervention summary CSV is appended to:
    {results_root}/rq1/intervention_summary.csv

Usage::

    python -m src.rq1.interventions
    python -m src.rq1.interventions --intervention beam_search --model ky --split test
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Callable

from src.utils.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intervention functions
# ---------------------------------------------------------------------------

def decode_repetition_penalty(
    model, processor, row: dict, language: str, config
) -> str:
    """Intervention A: add no_repeat_ngram_size=4 to suppress repetitions."""
    from src.eval.harness import decode_single
    from src.utils import audio as audio_mod
    waveform, sr = _load_waveform(row, audio_mod)
    result = decode_single(
        waveform, sr, model, processor,
        generate_kwargs={
            "language": _lang_token(language, config),
            "task": "transcribe",
            "no_repeat_ngram_size": 4,
            "do_sample": False,
        },
    )
    return result["hypothesis"]


def decode_max_length_from_audio(
    model, processor, row: dict, language: str, config
) -> str:
    """Intervention B: cap max_new_tokens ∝ audio duration."""
    from src.eval.harness import decode_single
    from src.utils import audio as audio_mod
    waveform, sr = _load_waveform(row, audio_mod)
    audio_len = len(waveform) / max(sr, 1)
    max_new_tokens = int(audio_len * config.interventions.max_length_multiplier) + 10

    result = decode_single(
        waveform, sr, model, processor,
        generate_kwargs={
            "language": _lang_token(language, config),
            "task": "transcribe",
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        },
    )
    return result["hypothesis"]


def decode_temperature_fallback(
    model, processor, row: dict, language: str, config
) -> str:
    """Intervention C: pass compression_ratio_threshold to enable Whisper's fallback."""
    from src.eval.harness import decode_single
    from src.utils import audio as audio_mod
    waveform, sr = _load_waveform(row, audio_mod)
    result = decode_single(
        waveform, sr, model, processor,
        generate_kwargs={
            "language": _lang_token(language, config),
            "task": "transcribe",
            "compression_ratio_threshold": config.decode.compression_ratio_threshold,
            "do_sample": False,
        },
    )
    return result["hypothesis"]


def decode_beam_search(
    model, processor, row: dict, language: str, config
) -> str:
    """Intervention D: beam search with num_beams=5."""
    from src.eval.harness import decode_single
    from src.utils import audio as audio_mod
    waveform, sr = _load_waveform(row, audio_mod)
    result = decode_single(
        waveform, sr, model, processor,
        generate_kwargs={
            "language": _lang_token(language, config),
            "task": "transcribe",
            "num_beams": config.interventions.beam_size,
            "do_sample": False,
        },
    )
    return result["hypothesis"]


def decode_vad_trim(
    model, processor, row: dict, language: str, config
) -> str:
    """Intervention E: trim silence with silero-VAD before decoding."""
    from src.eval.harness import decode_single
    from src.utils import audio as audio_mod

    audio_path = str(row.get("audio_path", ""))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        trimmed_path = audio_mod.vad_trim(
            audio_path, tmp_path, aggressiveness=config.interventions.vad_aggressiveness
        )
        waveform, sr = audio_mod.load_audio(trimmed_path)
    except Exception as exc:
        logger.warning("VAD trim failed for %s: %s — using original audio.", audio_path, exc)
        waveform, sr = _load_waveform(row, audio_mod)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    result = decode_single(
        waveform, sr, model, processor,
        generate_kwargs={
            "language": _lang_token(language, config),
            "task": "transcribe",
            "do_sample": False,
        },
    )
    return result["hypothesis"]


def decode_ctc_fallback(
    model, processor, row: dict, language: str, config
) -> str:
    """Intervention F: CTC fallback for catastrophic utterances (placeholder).

    TODO: Replace 'TODO_CTC_MODEL_ID' in base.yaml with a real CTC model for
    Kinyarwanda when one becomes available.  For now returns the Whisper
    hypothesis unchanged and logs a warning.
    """
    ctc_model_id = config.models[language].ctc_fallback_id
    warnings.warn(
        f"CTC fallback for language='{language}' uses placeholder model '{ctc_model_id}'. "
        "Results are unchanged from baseline Whisper output. "
        "Update models.{language}.ctc_fallback_id in base.yaml when a suitable "
        "CTC model is available.",
        stacklevel=2,
    )
    # Return the baseline Whisper hypothesis unchanged
    return str(row.get("hypothesis", ""))


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

INTERVENTION_FUNCTIONS: dict[str, Callable] = {
    "repetition_penalty": decode_repetition_penalty,
    "max_length_from_audio": decode_max_length_from_audio,
    "temperature_fallback": decode_temperature_fallback,
    "beam_search": decode_beam_search,
    "vad_trim": decode_vad_trim,
    "ctc_fallback": decode_ctc_fallback,
}


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_interventions(
    model_alias: str,
    split: str,
    flagged_parquet: str,
    cfg,
    all_utterances: bool = False,
):
    """Run all six interventions and return per-utterance results.

    Parameters
    ----------
    all_utterances:
        If True, run each intervention on the full split (needed for corpus
        and per-utterance WER comparison as specified by the research plan).
        If False, run only on catastrophic utterances.
    """
    import pandas as pd
    import torch
    from src.eval.harness import load_model
    from src.eval.metrics import uncapped_wer, per_utterance_wer, corpus_wer
    from src.eval.normalise import normalize

    df = pd.read_parquet(flagged_parquet)
    catastro = df[df["is_catastrophic"]].copy()
    target = df.copy() if all_utterances else catastro

    logger.info(
        "[%s/%s] %d/%d utterances flagged; running on %d (%s)",
        model_alias, split, len(catastro), len(df),
        len(target), "all" if all_utterances else "catastrophic only",
    )

    if target.empty:
        return pd.DataFrame(columns=["utt_id", "intervention", "wer_before",
                                     "wer_after", "delta_wer"])

    model, processor = load_model(
        model_id=cfg.models[model_alias].model_id,
        cache_dir=cfg.cache_dir,
    )

    all_rows: list[dict] = []

    for intervention_name, fn in INTERVENTION_FUNCTIONS.items():
        logger.info(
            "Running intervention '%s' on %d utterances [%s/%s]",
            intervention_name, len(target), model_alias, split,
        )
        n = len(target)
        for i, (_, row) in enumerate(target.iterrows()):
            row_dict = row.to_dict()
            ref_norm = str(row_dict.get("reference_normalised", ""))
            hyp_before_norm = str(row_dict.get("hypothesis_normalised", ""))
            wer_before_cap = float(row_dict.get("wer_capped", 1.0))
            wer_before_uncap = uncapped_wer(ref_norm, hyp_before_norm)

            try:
                hyp_after = fn(model, processor, row_dict, model_alias, cfg)
                hyp_after_norm = normalize(hyp_after, model_alias)
                wer_after_cap = per_utterance_wer(ref_norm, hyp_after_norm)
                wer_after_uncap = uncapped_wer(ref_norm, hyp_after_norm)
            except Exception as exc:
                logger.warning(
                    "Intervention %s failed for utt %s: %s",
                    intervention_name, row_dict.get("utt_id", "?"), exc,
                )
                hyp_after_norm = hyp_before_norm
                wer_after_cap = wer_before_cap
                wer_after_uncap = wer_before_uncap

            all_rows.append({
                "utt_id": row_dict.get("utt_id", ""),
                "intervention": intervention_name,
                "reference_normalised": ref_norm,
                "hypothesis_before_norm": hyp_before_norm,
                "hypothesis_after_norm": hyp_after_norm,
                "wer_before_capped": wer_before_cap,
                "wer_after_capped": wer_after_cap,
                "delta_wer_capped": wer_after_cap - wer_before_cap,
                "wer_before_uncapped": wer_before_uncap,
                "wer_after_uncapped": wer_after_uncap,
                "delta_wer_uncapped": wer_after_uncap - wer_before_uncap,
                "is_catastrophic": bool(row_dict.get("is_catastrophic", False)),
                "model": model_alias.upper(),
                "split": split,
            })

            if (i + 1) % 100 == 0:
                logger.info(
                    "  %s: %d/%d utterances done", intervention_name, i + 1, n
                )

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg=None):
    """Run all interventions for all language/split combinations."""
    import pandas as pd

    if cfg is None:
        cfg = load_config()

    rq1_dir = Path(cfg.results_root) / "rq1"
    failure_dir = rq1_dir / "failure"
    interv_dir = rq1_dir / "interventions"
    interv_dir.mkdir(parents=True, exist_ok=True)

    import argparse
    parser = argparse.ArgumentParser(description="RQ1 Interventions")
    parser.add_argument("--model", choices=["ky", "en", "both"], default="both")
    parser.add_argument("--split", choices=["dev", "test", "both"], default="both")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--intervention", default=None, help="Run a single named intervention")
    parser.add_argument("--all-utterances", action="store_true",
                        help="Run on full split (required for corpus WER comparison)")
    args = parser.parse_args()

    languages = ["ky", "en"] if args.model == "both" else [args.model]
    splits = ["dev", "test"] if args.split == "both" else [args.split]

    summary_rows: list[dict] = []

    for lang in languages:
        for split in splits:
            flagged_path = failure_dir / f"failures_{lang}_{split}.parquet"
            if not flagged_path.exists():
                logger.warning("Flagged parquet not found: %s — skipping.", flagged_path)
                continue

            out_path = interv_dir / f"interventions_{lang}_{split}.parquet"
            if args.resume and out_path.exists():
                logger.info("Skipping %s/%s — already exists.", lang, split)
                results_df = pd.read_parquet(out_path)
            else:
                results_df = run_all_interventions(
                    lang, split, str(flagged_path), cfg,
                    all_utterances=args.all_utterances,
                )
                results_df.to_parquet(out_path, index=False)
                logger.info("Saved %s", out_path)

            # Compute summary per intervention — corpus WER if all_utterances
            from src.eval.metrics import corpus_wer, per_utterance_wer
            for interv_name, grp in results_df.groupby("intervention"):
                row = {
                    "language": lang,
                    "split": split,
                    "intervention": interv_name,
                    "n_utterances": len(grp),
                    "mean_per_utt_wer_before": grp["wer_before_capped"].mean()
                        if "wer_before_capped" in grp.columns else grp.get("wer_before", pd.Series()).mean(),
                    "mean_per_utt_wer_after": grp["wer_after_capped"].mean()
                        if "wer_after_capped" in grp.columns else grp.get("wer_after", pd.Series()).mean(),
                }
                # Corpus WER only meaningful when running on full split
                if "reference_normalised" in grp.columns and "hypothesis_after_norm" in grp.columns:
                    row["corpus_wer_before"] = corpus_wer(
                        grp["reference_normalised"].tolist(),
                        grp["hypothesis_before_norm"].tolist(),
                    )
                    row["corpus_wer_after"] = corpus_wer(
                        grp["reference_normalised"].tolist(),
                        grp["hypothesis_after_norm"].tolist(),
                    )
                    row["corpus_wer_delta"] = row["corpus_wer_after"] - row["corpus_wer_before"]
                summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = rq1_dir / "intervention_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Intervention summary saved to %s", summary_path)

        print("\n=== Intervention Summary ===")
        print(summary_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_waveform(row: dict, audio_mod=None):
    """Load the waveform for a DataFrame row dict."""
    if audio_mod is None:
        from src.utils import audio as audio_mod
    import numpy as np
    audio_path = str(row.get("audio_path", ""))
    try:
        return audio_mod.load_audio(audio_path)
    except Exception as exc:
        logger.warning("Cannot load audio %s: %s — using silence.", audio_path, exc)
        return np.zeros(16000, dtype=np.float32), 16000


def _lang_token(language: str, config) -> str:
    """Look up the language token for *language* from config."""
    return config.models[language].language_token


if __name__ == "__main__":
    main()
