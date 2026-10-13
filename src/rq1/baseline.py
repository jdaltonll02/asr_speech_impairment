"""
RQ1 Baseline — reproduce CDLI published WER numbers.

Runs the CDLI KY and EN Whisper models on dev and test splits with published
decoding settings (greedy, beam_size=1, temperature=0, language='sw' for KY).

Per-utterance results are saved as Parquet files under
    {results_root}/rq1/baseline_{lang}_{split}.parquet

A summary CSV of corpus WER / CER is saved to:
    {results_root}/rq1/baseline_summary.csv

The script checks whether the reproduced WER falls within ±0.02 of the CDLI
published numbers (KY test: 0.72, EN test: 0.24) and logs the comparison.

Usage::

    python -m src.rq1.baseline
    python -m src.rq1.baseline --model ky --split dev
    python -m src.rq1.baseline --resume   # skip already-written parquets
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.data import DataNotReadyError, validate_data_root, load_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLITS = ["dev", "test"]
LANGUAGES = ["ky", "en"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_kwargs(language: str, cfg) -> dict:
    """Build generate() kwargs for the published baseline settings."""
    return {
        "language": cfg.models[language].language_token,
        "task": cfg.models[language].task,
        "num_beams": cfg.decode.beam_size,
        "do_sample": False,
    }


def _results_path(results_root: str, lang: str, split: str) -> Path:
    path = Path(results_root) / "rq1" / f"baseline_{lang}_{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_baseline(
    cfg,
    languages: list[str],
    splits: list[str],
    resume: bool,
):
    """Run baseline decoding for all specified language/split combinations.

    Returns a summary DataFrame with corpus WER and CER per combination.
    """
    import pandas as pd
    from src.eval.harness import load_model, transcribe_dataset
    from src.eval.metrics import corpus_wer

    summary_rows = []

    for lang in languages:
        model = None
        processor = None

        for split in splits:
            out_path = _results_path(cfg.results_root, lang, split)
            checkpoint_path = str(out_path).replace(".parquet", "_checkpoint.parquet")

            if resume and out_path.exists():
                logger.info("Skipping %s/%s — final parquet already exists.", lang, split)
                df = pd.read_parquet(out_path)

            else:
                # Lazy-load the model (once per language)
                if model is None:
                    logger.info("Loading model for language=%s", lang)
                    model, processor = load_model(
                        model_id=cfg.models[lang].model_id,
                        cache_dir=cfg.cache_dir,
                    )

                logger.info("Loading split: language=%s, split=%s", lang, split)
                split_df = load_split(split, lang, cfg.data_root)

                # Resume from partial checkpoint if it exists
                already_done_ids: set[str] = set()
                existing_rows: list[dict] = []
                ckpt = Path(checkpoint_path)
                if ckpt.exists():
                    try:
                        ckpt_df = pd.read_parquet(ckpt)
                        existing_rows = ckpt_df.to_dict("records")
                        already_done_ids = set(ckpt_df["utt_id"].tolist())
                        logger.info(
                            "Resuming %s/%s from checkpoint: %d/%d utterances already done.",
                            lang, split, len(already_done_ids), len(split_df),
                        )
                        # Filter out already-done rows from the split
                        split_df = split_df[
                            ~split_df["utt_id"].isin(already_done_ids)
                        ].reset_index(drop=True)
                    except Exception as exc:
                        logger.warning(
                            "Could not read checkpoint %s: %s — restarting from scratch.",
                            ckpt, exc,
                        )
                        existing_rows = []

                decode_kwargs = _decode_kwargs(lang, cfg)

                logger.info(
                    "Transcribing %d utterances (lang=%s, split=%s)...",
                    len(split_df), lang, split,
                )

                new_df = transcribe_dataset(
                    model=model,
                    processor=processor,
                    dataset_df=split_df,
                    language=lang,
                    decode_kwargs=decode_kwargs,
                    batch_size=cfg.decode.batch_size,
                    results_root=str(Path(cfg.results_root) / "rq1"),
                    checkpoint_path=checkpoint_path,
                    checkpoint_every=cfg.decode.get("checkpoint_every", 50),
                    prior_rows=existing_rows,
                )

                df = new_df
                df.to_parquet(out_path, index=False)
                logger.info("Saved final results: %s", out_path)

                # Clean up checkpoint now that we have the final file
                if ckpt.exists():
                    ckpt.unlink()
                    logger.info("Removed checkpoint %s", ckpt)

            # Compute corpus WER / CER
            w = corpus_wer(
                df["reference_normalised"].tolist(),
                df["hypothesis_normalised"].tolist(),
            )
            cer_val = df["cer"].mean() if "cer" in df.columns else float("nan")

            summary_rows.append(
                {
                    "language": lang,
                    "split": split,
                    "n_utterances": len(df),
                    "corpus_wer": round(w, 4),
                    "mean_cer": round(cer_val, 4),
                }
            )

            logger.info(
                "[%s/%s] corpus WER = %.4f  mean CER = %.4f",
                lang, split, w, cer_val,
            )

        # Free GPU memory after processing all splits for this language
        if model is not None:
            import torch
            del model, processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def _check_against_published(summary_df: pd.DataFrame, cfg) -> None:
    """Log whether reproduced WER matches published numbers (±0.02)."""
    published = {
        ("ky", "test"): cfg.published_wer.ky_test_wer,
        ("en", "test"): cfg.published_wer.en_test_wer,
    }
    tolerance = 0.02
    for (lang, split), pub_wer in published.items():
        row = summary_df[
            (summary_df["language"] == lang) & (summary_df["split"] == split)
        ]
        if row.empty:
            continue
        repro_wer = row["corpus_wer"].iloc[0]
        diff = abs(repro_wer - pub_wer)
        status = "✓ WITHIN" if diff <= tolerance else "✗ OUTSIDE"
        logger.info(
            "[%s/%s] published WER=%.4f  reproduced WER=%.4f  diff=%.4f  %s ±%.2f tolerance",
            lang, split, pub_wer, repro_wer, diff, status, tolerance,
        )


def main(cfg=None):
    """Entry point: parse CLI args and run baseline."""
    if cfg is None:
        cfg = load_config()

    parser = argparse.ArgumentParser(description="RQ1 Baseline decoder")
    parser.add_argument(
        "--model",
        choices=["ky", "en", "both"],
        default="both",
        help="Which model/language to run (default: both)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "both"],
        default="both",
        help="Which split to run (default: both)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip language/split combinations that already have output parquets",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download the CDLI dataset from HuggingFace before running inference. "
            "Requires HF_TOKEN to be set. Does NOT run inference — just downloads."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (default: config/base.yaml)",
    )
    args = parser.parse_args()

    if args.config:
        from src.utils.config import reset_config
        reset_config()
        cfg = load_config(args.config)

    languages = LANGUAGES if args.model == "both" else [args.model]
    splits = SPLITS if args.split == "both" else [args.split]

    # Handle --download: fetch dataset to local data root, then exit
    if args.download:
        from src.utils.data import download_dataset
        for lang in languages:
            print(f"\nDownloading language='{lang}' ...")
            download_dataset(language=lang, data_root=cfg.data_root, cfg=cfg)
        print("\nDownload finished. Run without --download to start inference.")
        sys.exit(0)

    # Validate data availability before loading models
    try:
        validate_data_root(cfg.data_root)
    except DataNotReadyError as exc:
        print(f"\n[INFO] Data not yet available — skipping baseline run.\n{exc}\n")
        sys.exit(0)

    summary_df = run_baseline(cfg, languages, splits, resume=args.resume)

    # Save summary
    summary_path = Path(cfg.results_root) / "rq1" / "baseline_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary saved to %s", summary_path)

    _check_against_published(summary_df, cfg)

    print("\n=== Baseline Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
