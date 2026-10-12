"""
RQ1 Logistic Regression — explain catastrophic failure with interpretable features.

Fits a balanced logistic regression on the dev split (``is_catastrophic`` as
target), evaluates on test, and produces:
- ``results_root/rq1/regression_results.json`` — coefficients, odds ratios,
  bootstrap 95 % CIs, AUC-ROC
- ``results_root/rq1/regression_coefs.png`` — coefficient forest plot

Usage::

    python -m src.rq1.regression
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from src.utils.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_feature_matrix(df) -> "tuple":
    """Build feature matrix X, target y, and feature names from *df*.

    Categorical features are one-hot encoded via ``pd.get_dummies``.
    Numeric features are kept as-is (caller should standardise before fitting).

    Features used:
    - ``prompt_type`` (dummy-encoded)
    - ``audio_length_sec`` (numeric)
    - ``severity`` (dummy-encoded, from speaker metadata join)
    - ``recording_environment`` (dummy-encoded)
    - ``leading_silence_s`` (numeric)
    - ``trailing_silence_s`` (numeric)
    - ``snr_db`` (numeric)

    Parameters
    ----------
    df:
        Per-utterance DataFrame with all diagnostic columns.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        X (n_samples × n_features), y (n_samples,), feature_names.
    """
    import numpy as np
    import pandas as pd
    numeric_cols = [
        "audio_length_sec",
        "leading_silence_s",
        "trailing_silence_s",
        "snr_db",
    ]
    categorical_cols = ["prompt_type", "recording_environment"]
    # severity may be present after speaker_metadata join
    if "severity" in df.columns:
        categorical_cols.append("severity")

    # Fill missing values
    df_feat = df.copy()
    for col in numeric_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0.0
        df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce").fillna(0.0)

    for col in categorical_cols:
        if col not in df_feat.columns:
            df_feat[col] = "unknown"
        df_feat[col] = df_feat[col].fillna("unknown").astype(str)

    # Dummy-encode categoricals
    dummies = pd.get_dummies(df_feat[categorical_cols], drop_first=True, dtype=float)
    numeric = df_feat[numeric_cols].astype(float)

    X_df = pd.concat([numeric, dummies], axis=1)
    feature_names = list(X_df.columns)
    X = X_df.values

    y = df_feat["is_catastrophic"].astype(int).values

    return X, y, feature_names


def fit_regression(X_train: np.ndarray, y_train: np.ndarray):
    """Fit a balanced logistic regression and return the fitted model.

    Parameters
    ----------
    X_train:
        Feature matrix (standardised).
    y_train:
        Binary target array.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Fitted model.
    """
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def compute_bootstrap_cis(
    model,
    X,
    y,
    feature_names: list[str],
    n_bootstrap: int = 1000,
    random_seed: int = 42,
):
    """Compute bootstrap 95 % confidence intervals for logistic regression coefficients.

    Parameters
    ----------
    model:
        Fitted LogisticRegression model.
    X:
        Feature matrix used for CIs.
    y:
        Target array.
    feature_names:
        Names of feature columns (in X column order).
    n_bootstrap:
        Number of bootstrap resamples.
    random_seed:
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``coef``, ``odds_ratio``, ``ci_lower``,
        ``ci_upper``.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(random_seed)
    n_samples = len(y)
    coef_samples = np.zeros((n_bootstrap, len(feature_names)))

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        X_boot, y_boot = X[idx], y[idx]

        # Skip if only one class present in resample
        if len(np.unique(y_boot)) < 2:
            coef_samples[i] = model.coef_[0]
            continue

        boot_model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            random_state=int(rng.integers(0, 2**31)),
            solver="lbfgs",
        )
        try:
            boot_model.fit(X_boot, y_boot)
            coef_samples[i] = boot_model.coef_[0]
        except Exception:
            coef_samples[i] = model.coef_[0]

    coefs = model.coef_[0]
    ci_lower = np.percentile(coef_samples, 2.5, axis=0)
    ci_upper = np.percentile(coef_samples, 97.5, axis=0)

    result_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coefs,
            "odds_ratio": np.exp(coefs),
            "ci_lower": np.exp(ci_lower),
            "ci_upper": np.exp(ci_upper),
        }
    )
    return result_df.sort_values("coef", key=abs, ascending=False)


def _plot_coefs(coef_df, output_path: str) -> None:
    """Save a coefficient forest plot (odds ratios with 95 % CIs)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, max(4, len(coef_df) * 0.5)))

    y_pos = np.arange(len(coef_df))
    ax.errorbar(
        coef_df["odds_ratio"],
        y_pos,
        xerr=[
            coef_df["odds_ratio"] - coef_df["ci_lower"],
            coef_df["ci_upper"] - coef_df["odds_ratio"],
        ],
        fmt="o",
        color="steelblue",
        ecolor="gray",
        capsize=4,
    )
    ax.axvline(x=1.0, linestyle="--", color="red", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df["feature"])
    ax.set_xlabel("Odds Ratio (95 % CI)")
    ax.set_title("Logistic Regression Coefficients — Catastrophic Failure")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def stratified_sample(df, n: int = 50, random_seed: int = 42):
    """Return a stratified sample of catastrophic utterances for manual listening.

    Stratifies on ``(model, recording_environment, prompt_type)``.

    Parameters
    ----------
    df:
        Per-utterance DataFrame (all splits combined).
    n:
        Target sample size.
    random_seed:
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Sample of *n* catastrophic utterances (or all if fewer than *n*).
    """
    import pandas as pd

    catastro = df[df["is_catastrophic"]].copy()
    if len(catastro) <= n:
        return catastro

    strat_cols = [c for c in ["model", "recording_environment", "prompt_type"] if c in catastro.columns]
    if not strat_cols:
        return catastro.sample(n, random_state=random_seed)

    catastro["_strat"] = catastro[strat_cols].astype(str).apply("_".join, axis=1)
    groups = catastro["_strat"].value_counts(normalize=True)

    sample_rows = []
    for group, frac in groups.items():
        k = max(1, round(frac * n))
        subset = catastro[catastro["_strat"] == group]
        sample_rows.append(subset.sample(min(k, len(subset)), random_state=random_seed))

    result = pd.concat(sample_rows).head(n)
    return result.drop(columns=["_strat"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg=None):
    """Fit logistic regression, evaluate, save results and forest plot."""
    import numpy as np
    import pandas as pd
    import argparse
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, classification_report

    parser = argparse.ArgumentParser(description="RQ1 Logistic Regression")
    parser.add_argument("--lang", choices=["ky", "en", "both"], default="both")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.config:
        from src.utils.config import reset_config
        reset_config()
        cfg = load_config(args.config)

    if cfg is None:
        cfg = load_config()

    languages = ["ky", "en"] if args.lang == "both" else [args.lang]

    rq1_dir = Path(cfg.results_root) / "rq1"
    failure_dir = rq1_dir / "failure"

    # Load failure parquets for requested languages
    dev_dfs, test_dfs = [], []
    for lang in languages:
        dev_p = failure_dir / f"failures_{lang}_dev.parquet"
        test_p = failure_dir / f"failures_{lang}_test.parquet"
        if dev_p.exists():
            dev_dfs.append(pd.read_parquet(dev_p))
        if test_p.exists():
            test_dfs.append(pd.read_parquet(test_p))

    if not dev_dfs:
        logger.error("No dev failure parquets found. Run failure.py first.")
        sys.exit(1)

    dev_df = pd.concat(dev_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else dev_df.copy()

    # Build feature matrices
    X_train, y_train, feature_names = build_feature_matrix(dev_df)
    X_test, y_test, _ = build_feature_matrix(test_df)

    # Standardise numeric features (fit on train, apply to test)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Fit
    lr_model = fit_regression(X_train_sc, y_train)

    # Evaluate on test
    y_pred = lr_model.predict(X_test_sc)
    try:
        y_prob = lr_model.predict_proba(X_test_sc)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = float("nan")

    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("Test AUC-ROC = %.4f", auc)

    # Bootstrap CIs
    coef_df = compute_bootstrap_cis(lr_model, X_train_sc, y_train, feature_names)

    # Save results
    results = {
        "feature_names": feature_names,
        "coef": lr_model.coef_[0].tolist(),
        "odds_ratio": np.exp(lr_model.coef_[0]).tolist(),
        "auc_roc": auc,
        "classification_report": report,
        "bootstrap_cis": coef_df.to_dict(orient="records"),
    }
    out_path = rq1_dir / "regression_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Regression results saved to %s", out_path)

    # Forest plot
    plot_path = rq1_dir / "regression_coefs.png"
    _plot_coefs(coef_df, str(plot_path))
    logger.info("Forest plot saved to %s", plot_path)

    # Stratified sample for manual listening
    all_df = pd.concat(dev_dfs + test_dfs, ignore_index=True)
    sample = stratified_sample(all_df)
    sample_path = rq1_dir / "catastrophic_sample_50.csv"
    sample.to_csv(sample_path, index=False)
    logger.info("Stratified sample saved to %s", sample_path)

    print(f"\nTest AUC-ROC: {auc:.4f}")
    print(coef_df[["feature", "coef", "odds_ratio", "ci_lower", "ci_upper"]].to_string(index=False))


if __name__ == "__main__":
    main()
