"""
RQ1 Report — aggregate results and produce publication-ready tables and plots.

Loads all Parquet files from ``baseline/``, ``failure/``, and
``interventions/`` subdirectories and produces:

1. ``table_baseline_wer.csv``          — WER by model × split
2. ``fig_wer_distribution.png``        — per-utterance WER histogram
3. ``fig_catastrophic_share.png``      — % utterances and % word errors that
                                         are catastrophic, per language
4. ``regression_coefs.png``            — forest plot (written by regression.py;
                                         symlinked/copied here)
5. ``table_interventions.csv``         — mean ∆WER per intervention per model
6. ``fig_interventions.png``           — grouped bar chart of ∆WER

All outputs are saved to ``{results_root}/rq1/``.

Usage::

    python -m src.rq1.report
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from src.utils.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual report generators
# ---------------------------------------------------------------------------

def generate_baseline_table(rq1_dir: Path) -> None:
    """Load baseline_summary.csv and save a clean copy as a CSV table."""
    import pandas as pd

    src = rq1_dir / "baseline_summary.csv"
    if not src.exists():
        logger.warning("baseline_summary.csv not found — skipping baseline table.")
        return

    df = pd.read_csv(src)
    out = rq1_dir / "table_baseline_wer.csv"
    df.to_csv(out, index=False)
    logger.info("Baseline table saved to %s", out)


def generate_catastrophic_table(rq1_dir: Path) -> None:
    """Compute and print share of catastrophic utterances and word errors."""
    import pandas as pd

    failure_dir = rq1_dir / "failure"
    rows = []

    for lang in ("ky", "en"):
        for split in ("dev", "test"):
            p = failure_dir / f"failures_{lang}_{split}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            n_total = len(df)
            n_catastro = int(df["is_catastrophic"].sum())
            total_errors = int(
                (df["substitutions"] + df["deletions"] + df["insertions"]).sum()
            )
            catastro_errors = int(
                (
                    df.loc[df["is_catastrophic"], "substitutions"]
                    + df.loc[df["is_catastrophic"], "deletions"]
                    + df.loc[df["is_catastrophic"], "insertions"]
                ).sum()
            )
            rows.append({
                "language": lang,
                "split": split,
                "n_total": n_total,
                "n_catastrophic": n_catastro,
                "share_utterances": n_catastro / max(n_total, 1),
                "total_word_errors": total_errors,
                "catastrophic_word_errors": catastro_errors,
                "share_word_errors": catastro_errors / max(total_errors, 1),
            })

    if not rows:
        logger.warning("No failure parquets found — skipping catastrophic table.")
        return

    summary_df = pd.DataFrame(rows)
    out = rq1_dir / "table_catastrophic_share.csv"
    summary_df.to_csv(out, index=False)
    logger.info("Catastrophic share table saved to %s", out)


def plot_wer_distribution(rq1_dir: Path) -> None:
    """Plot per-utterance WER histogram for KY and EN (catastrophic in red)."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    failure_dir = rq1_dir / "failure"
    dfs = []
    for lang in ("ky", "en"):
        for split in ("dev", "test"):
            p = failure_dir / f"failures_{lang}_{split}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df["language"] = lang
                dfs.append(df)

    if not dfs:
        logger.warning("No failure parquets for WER distribution plot — skipping.")
        return

    all_df = pd.concat(dfs, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, lang in zip(axes, ("ky", "en")):
        sub = all_df[all_df["language"] == lang]
        if sub.empty:
            ax.set_title(lang.upper())
            continue
        normal = sub[~sub["is_catastrophic"]]["wer_uncapped"]
        catastro = sub[sub["is_catastrophic"]]["wer_uncapped"]
        ax.hist(normal, bins=40, alpha=0.6, color="steelblue", label="Normal")
        ax.hist(catastro, bins=40, alpha=0.6, color="red", label="Catastrophic")
        ax.axvline(x=1.0, linestyle="--", color="black", linewidth=1, label="WER=1.0")
        ax.set_xlabel("Uncapped WER")
        ax.set_ylabel("Count")
        ax.set_title(f"{lang.upper()} — WER Distribution")
        ax.legend()

    plt.tight_layout()
    out = rq1_dir / "fig_wer_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("WER distribution plot saved to %s", out)


def plot_intervention_comparison(rq1_dir: Path) -> None:
    """Grouped bar chart: mean ∆WER for baseline + 6 interventions per language."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary_path = rq1_dir / "intervention_summary.csv"
    if not summary_path.exists():
        logger.warning("intervention_summary.csv not found — skipping intervention plot.")
        return

    df = pd.read_csv(summary_path)
    test_df = df[df["split"] == "test"]
    if test_df.empty:
        test_df = df  # fall back to all splits

    interventions = sorted(test_df["intervention"].unique())
    languages = sorted(test_df["language"].unique())

    x = np.arange(len(interventions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, lang in enumerate(languages):
        lang_df = test_df[test_df["language"] == lang]
        means = [
            lang_df[lang_df["intervention"] == iv]["mean_delta_wer"].mean()
            if iv in lang_df["intervention"].values else 0.0
            for iv in interventions
        ]
        offset = (i - len(languages) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=lang.upper())

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Intervention")
    ax.set_ylabel("Mean ∆WER (negative = improvement)")
    ax.set_title("Intervention Effect on Catastrophic Utterances")
    ax.set_xticks(x)
    ax.set_xticklabels(interventions, rotation=20, ha="right")
    ax.legend()
    plt.tight_layout()

    out = rq1_dir / "fig_interventions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Intervention comparison plot saved to %s", out)


def plot_regression_coefs(rq1_dir: Path) -> None:
    """Copy regression forest plot into the report directory (if not already there)."""
    src = rq1_dir / "regression_coefs.png"
    if not src.exists():
        logger.warning("regression_coefs.png not found — skipping.")
        return
    dst = rq1_dir / "fig_logit_coef.png"
    if not dst.exists():
        shutil.copy2(src, dst)
        logger.info("Copied regression coef plot to %s", dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(cfg=None):
    """Run all report generators."""
    import argparse

    parser = argparse.ArgumentParser(description="RQ1 Report generator")
    parser.add_argument("--lang", choices=["ky", "en", "both"], default="both")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.config:
        from src.utils.config import reset_config
        reset_config()
        cfg = load_config(args.config)

    if cfg is None:
        cfg = load_config()

    rq1_dir = Path(cfg.results_root) / "rq1"
    rq1_dir.mkdir(parents=True, exist_ok=True)

    generate_baseline_table(rq1_dir)
    generate_catastrophic_table(rq1_dir)
    plot_wer_distribution(rq1_dir)
    plot_intervention_comparison(rq1_dir)
    plot_regression_coefs(rq1_dir)

    logger.info("Report generation complete. Outputs in %s", rq1_dir)
    print(f"\nReport outputs saved to: {rq1_dir}")


def main(cfg=None):
    generate_report(cfg)


if __name__ == "__main__":
    main()
