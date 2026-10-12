"""
RQ1 Failure Diagnostics — identify and characterise catastrophic ASR failures.

Reads baseline Parquet files written by ``baseline.py``, adds diagnostic
columns, flags catastrophic utterances, and prints Table 1.

Thresholds for ``is_catastrophic``:
    - ``wer_uncapped > catastrophic.uncapped_wer_threshold`` (default 1.0)
    - ``length_ratio > catastrophic.length_ratio_threshold`` (default 2.0)
    - ``repeated_4gram_rate > threshold`` (threshold data-driven from dev)

The 4-gram threshold is determined on dev by maximising Youden's J statistic
(TPR - FPR) treating ``wer_uncapped > 1.0`` as the ground-truth proxy for a
catastrophic utterance.  The chosen threshold is saved to
    {results_root}/rq1/thresholds.json

Usage::

    python -m src.rq1.failure
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from src.utils.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic columns
# ---------------------------------------------------------------------------

def add_failure_diagnostics(df, config):
    """Add derived diagnostic columns to a baseline results DataFrame.

    Parameters
    ----------
    df:
        Per-utterance DataFrame produced by ``harness.transcribe_dataset``.
    config:
        Loaded configuration object (from ``load_config()``).

    Returns
    -------
    pd.DataFrame (or equivalent)
        Copy of *df* with additional columns.
    """
    import numpy as np
    import pandas as pd
    from src.eval.metrics import repeated_ngram_rate

    df = df.copy()

    # Length ratio: |hypothesis| / max(|reference|, 1)
    if "length_ratio" not in df.columns:
        df["length_ratio"] = df["n_hyp_words"] / df["n_ref_words"].clip(lower=1)

    # Compression ratio: use harness value if present, else approximate
    if "compression_ratio" not in df.columns or df["compression_ratio"].isna().all():
        # Approximate: len(hypothesis_chars) / (audio_length_sec * 14)
        # 14 chars/sec is a rough speech-rate proxy
        df["compression_ratio"] = df["hypothesis_normalised"].str.len() / (
            df["audio_length_sec"].clip(lower=0.1) * 14
        )
    else:
        # Fill nulls with approximation
        mask = df["compression_ratio"].isna()
        df.loc[mask, "compression_ratio"] = (
            df.loc[mask, "hypothesis_normalised"].str.len()
            / (df.loc[mask, "audio_length_sec"].clip(lower=0.1) * 14)
        )

    # avg_log_prob: keep as-is (None/NaN stays NaN — not used for flagging)

    # Repeated 4-gram rate
    if "repeated_4gram_rate" not in df.columns or df["repeated_4gram_rate"].isna().all():
        df["repeated_4gram_rate"] = df["hypothesis_normalised"].apply(
            lambda t: repeated_ngram_rate(t if isinstance(t, str) else "", n=4)
        )

    # Catastrophic flag (initial: use config threshold; overridden after
    # set_repeated_4gram_threshold_on_dev runs)
    threshold_4gram = config.catastrophic.repeated_4gram_threshold
    df["is_catastrophic"] = (
        (df["wer_uncapped"] > config.catastrophic.uncapped_wer_threshold)
        | (df["length_ratio"] > config.catastrophic.length_ratio_threshold)
        | (df["repeated_4gram_rate"] > threshold_4gram)
    )

    return df


def set_repeated_4gram_threshold_on_dev(dev_df) -> float:
    """Find the repeated 4-gram threshold that maximises Youden's J on dev.

    Uses ``wer_uncapped > 1.0`` as the ground-truth proxy for catastrophic
    utterances (independent of the 4-gram criterion).

    Parameters
    ----------
    dev_df:
        Per-utterance dev split DataFrame.

    Returns
    -------
    float
        Chosen threshold value.
    """
    import numpy as np

    y_true = (dev_df["wer_uncapped"] > 1.0).astype(int).values
    scores = dev_df["repeated_4gram_rate"].fillna(0).values

    # Candidate thresholds: all unique values in the score array
    thresholds = np.unique(scores)

    best_j = -np.inf
    best_threshold = thresholds[0] if len(thresholds) > 0 else 0.1

    for t in thresholds:
        y_pred = (scores > t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr

        if j > best_j:
            best_j = j
            best_threshold = float(t)

    logger.info(
        "dev 4-gram threshold = %.4f (Youden J = %.4f)", best_threshold, best_j
    )
    return best_threshold


def flag_catastrophic(df, threshold_4gram: float, config):
    """Apply final catastrophic flags using the data-driven 4-gram threshold.

    Parameters
    ----------
    df:
        DataFrame with ``wer_uncapped``, ``length_ratio``, and
        ``repeated_4gram_rate`` columns.
    threshold_4gram:
        Threshold determined by ``set_repeated_4gram_threshold_on_dev``.
    config:
        Config object.

    Returns
    -------
    pd.DataFrame
        *df* with ``is_catastrophic`` column set.
    """
    df = df.copy()
    df["is_catastrophic"] = (
        (df["wer_uncapped"] > config.catastrophic.uncapped_wer_threshold)
        | (df["length_ratio"] > config.catastrophic.length_ratio_threshold)
        | (df["repeated_4gram_rate"] > threshold_4gram)
    )
    return df


def compute_summary(df) -> dict:
    """Compute the share of catastrophic utterances and word errors.

    Parameters
    ----------
    df:
        Per-utterance DataFrame with ``is_catastrophic``, ``substitutions``,
        ``deletions``, and ``insertions`` columns.

    Returns
    -------
    dict
        Keys: ``n_total``, ``n_catastrophic``, ``share_utterances``,
        ``total_word_errors``, ``catastrophic_word_errors``,
        ``share_word_errors``.
    """
    n_total = len(df)
    n_catastrophic = int(df["is_catastrophic"].sum())
    share_utterances = n_catastrophic / max(n_total, 1)

    total_errors = int(
        (df["substitutions"] + df["deletions"] + df["insertions"]).sum()
    )
    catast_errors = int(
        (
            df.loc[df["is_catastrophic"], "substitutions"]
            + df.loc[df["is_catastrophic"], "deletions"]
            + df.loc[df["is_catastrophic"], "insertions"]
        ).sum()
    )
    share_word_errors = catast_errors / max(total_errors, 1)

    return {
        "n_total": n_total,
        "n_catastrophic": n_catastrophic,
        "share_utterances": round(share_utterances, 4),
        "total_word_errors": total_errors,
        "catastrophic_word_errors": catast_errors,
        "share_word_errors": round(share_word_errors, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg=None):
    """Load baseline parquets, add diagnostics, and save failure parquets."""
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser(description="RQ1 Failure diagnostics")
    parser.add_argument("--lang", choices=["ky", "en", "both"], default="both",
                        help="Which language to process (default: both)")
    parser.add_argument("--split", choices=["dev", "test", "both"], default="both",
                        help="Which split to process (default: both)")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    if args.config:
        from src.utils.config import reset_config
        reset_config()
        cfg = load_config(args.config)

    if cfg is None:
        cfg = load_config()

    languages = ["ky", "en"] if args.lang == "both" else [args.lang]
    splits = ["dev", "test"] if args.split == "both" else [args.split]

    rq1_dir = Path(cfg.results_root) / "rq1"
    failure_dir = rq1_dir / "failure"
    failure_dir.mkdir(parents=True, exist_ok=True)

    # Load all baseline parquets for the requested language/split combinations
    dfs: dict[tuple[str, str], pd.DataFrame] = {}
    for lang in languages:
        for split in splits:
            p = rq1_dir / f"baseline_{lang}_{split}.parquet"
            if not p.exists():
                logger.warning("Baseline parquet not found: %s — skipping.", p)
                continue
            dfs[(lang, split)] = pd.read_parquet(p)

    if not dfs:
        logger.error(
            "No baseline parquets found under %s. Run baseline.py first.", rq1_dir
        )
        sys.exit(1)

    # Add diagnostics to all splits
    for key in list(dfs.keys()):
        dfs[key] = add_failure_diagnostics(dfs[key], cfg)

    # Determine 4-gram threshold on dev (use KY dev if available, else EN dev)
    dev_key = ("ky", "dev") if ("ky", "dev") in dfs else ("en", "dev")
    if dev_key in dfs:
        threshold_4gram = set_repeated_4gram_threshold_on_dev(dfs[dev_key])
    else:
        threshold_4gram = cfg.catastrophic.repeated_4gram_threshold
        logger.warning(
            "Dev split not available; using config threshold %.4f", threshold_4gram
        )

    # Save thresholds
    thresholds = {
        "uncapped_wer_threshold": cfg.catastrophic.uncapped_wer_threshold,
        "length_ratio_threshold": cfg.catastrophic.length_ratio_threshold,
        "repeated_4gram_threshold": threshold_4gram,
    }
    thresh_path = rq1_dir / "thresholds.json"
    thresh_path.write_text(json.dumps(thresholds, indent=2))
    logger.info("Thresholds saved to %s", thresh_path)

    # Apply final flags and save failure parquets
    print("\n=== Table 1: Catastrophic Failure Summary ===")
    print(f"{'Language':<10} {'Split':<8} {'N':<8} {'Catastro.':<12} {'%Utts':<10} {'%WordErr':<10}")
    print("-" * 60)

    for (lang, split), df in dfs.items():
        df = flag_catastrophic(df, threshold_4gram, cfg)
        dfs[(lang, split)] = df

        out_path = failure_dir / f"failures_{lang}_{split}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved %s", out_path)

        summary = compute_summary(df)
        print(
            f"{lang:<10} {split:<8} {summary['n_total']:<8} "
            f"{summary['n_catastrophic']:<12} "
            f"{summary['share_utterances']:.1%}    "
            f"{summary['share_word_errors']:.1%}"
        )

    print()


if __name__ == "__main__":
    main()
