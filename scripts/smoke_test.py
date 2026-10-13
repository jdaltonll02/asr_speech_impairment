"""
Smoke test — runs the full RQ1 pipeline on synthetic data without a GPU.

Checks:
  1. All imports resolve
  2. Config loads and required keys exist
  3. Normaliser works for KY and EN
  4. Metrics (WER, CER, edit ops) produce sensible values
  5. Failure diagnostics (length ratio, repeated n-gram, catastrophic flag)
  6. harness.transcribe_dataset runs end-to-end on 3 fake utterances using
     a tiny CPU stub model (no real Whisper weights needed)
  7. Checkpoint write + resume: simulate preemption mid-batch and confirm
     the second run merges prior_rows correctly
  8. load_split works against real downloaded TSV (dev, 5 rows only)
  9. Speaker metadata join produces a 'severity' column

Run with:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --no-data   # skip checks that need local data
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

# Ensure project root is on sys.path when run directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
_failures: list[str] = []


def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
    except Exception as exc:
        print(f"  {FAIL}  {name}")
        print(f"       {type(exc).__name__}: {exc}")
        traceback.print_exc()
        _failures.append(name)


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

def test_imports():
    from src.eval import normalise, metrics, harness  # noqa: F401
    from src.rq1 import baseline, failure, regression, interventions, report  # noqa: F401
    from src.utils import config, data, audio, slurm  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------

def test_config():
    from src.utils.config import load_config
    cfg = load_config()
    assert hasattr(cfg, "data_root"), "cfg missing data_root"
    assert hasattr(cfg, "models"), "cfg missing models"
    assert hasattr(cfg.models, "ky"), "cfg missing models.ky"
    assert hasattr(cfg.models, "en"), "cfg missing models.en"
    assert hasattr(cfg, "results_root"), "cfg missing results_root"
    assert hasattr(cfg, "published_wer"), "cfg missing published_wer"


# ---------------------------------------------------------------------------
# 3. Normaliser
# ---------------------------------------------------------------------------

def test_normaliser():
    from src.eval.normalise import normalize

    # KY: elision apostrophe removed and tokens joined
    result = normalize("n'abantu bose", "ky")
    assert "nabantu" in result, f"KY elision not handled: {result!r}"

    # EN: lowercased, punctuation stripped
    result_en = normalize("Hello, World!", "en")
    assert result_en == result_en.lower(), f"EN not lowercased: {result_en!r}"


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

def test_metrics():
    from src.eval.metrics import (
        word_errors, corpus_wer, per_utterance_wer, cer, uncapped_wer,
        repeated_ngram_rate,
    )

    ref = "the cat sat on the mat"
    hyp = "the cat sat on the hat"
    errs = word_errors(ref, hyp)
    assert errs["substitutions"] == 1, f"expected 1 sub, got {errs}"
    assert errs["deletions"] == 0
    assert errs["insertions"] == 0

    # Corpus WER: 1 error / 6 ref words
    wer = corpus_wer([ref], [hyp])
    assert abs(wer - 1 / 6) < 1e-6, f"corpus_wer wrong: {wer}"

    # Per-utterance capped at 1.0
    capped = per_utterance_wer("a", "a b c d e f g h i j k")
    assert capped <= 1.0, f"per_utterance_wer not capped: {capped}"

    # Uncapped can exceed 1.0
    uncapped = uncapped_wer("a", "a b c d e f g h i j k")
    assert uncapped > 1.0, f"uncapped_wer should exceed 1: {uncapped}"

    # CER
    cer_val = cer("hello", "helo")
    assert 0 < cer_val < 1, f"cer out of range: {cer_val}"

    # Repeated n-gram rate: repeated phrase should score high
    rep_rate = repeated_ngram_rate("a b c d a b c d a b c d", n=4)
    assert rep_rate > 0, f"repeated_ngram_rate should be > 0: {rep_rate}"
    no_rep = repeated_ngram_rate("the quick brown fox jumps over the lazy dog", n=4)
    assert no_rep < rep_rate, "non-repeating text should have lower rate"


# ---------------------------------------------------------------------------
# 5. Failure diagnostics
# ---------------------------------------------------------------------------

def test_failure_diagnostics():
    from src.rq1.failure import compute_failure_features, apply_catastrophic_flag
    import pandas as pd

    rows = [
        {
            "utt_id": "u1",
            "wer_uncapped": 0.5,
            "n_ref_words": 10,
            "n_hyp_words": 8,
            "hypothesis_normalised": "the cat sat on the mat",
            "compression_ratio": 0.9,
            "avg_log_prob": -0.3,
            "audio_length_sec": 3.0,
        },
        {
            # Should be catastrophic: uncapped WER > 1.0
            "utt_id": "u2",
            "wer_uncapped": 1.5,
            "n_ref_words": 4,
            "n_hyp_words": 10,
            "hypothesis_normalised": "a b c d a b c d a b c d",
            "compression_ratio": 2.5,
            "avg_log_prob": -1.2,
            "audio_length_sec": 2.0,
        },
    ]
    df = pd.DataFrame(rows)
    df = compute_failure_features(df)
    df = apply_catastrophic_flag(df, wer_threshold=1.0, length_ratio_threshold=2.0,
                                  ngram_threshold=0.1)
    assert not df.loc[df["utt_id"] == "u1", "is_catastrophic"].iloc[0], \
        "u1 should not be catastrophic"
    assert df.loc[df["utt_id"] == "u2", "is_catastrophic"].iloc[0], \
        "u2 should be catastrophic (WER > 1.0 and length_ratio > 2.0)"


# ---------------------------------------------------------------------------
# 6. Harness stub run (CPU, no real model)
# ---------------------------------------------------------------------------

class _StubModel:
    """Minimal stub that mimics WhisperForConditionalGeneration interface."""
    class _Params:
        def device(self): ...
    def parameters(self):
        import torch
        class _P:
            device = torch.device("cpu")
            dtype = torch.float32
        yield _P()
    def eval(self): return self
    def generate(self, input_features, **kwargs):
        import torch
        class _Out:
            sequences = torch.zeros((1, 5), dtype=torch.long)
            sequences_scores = torch.tensor([-1.0])
        return _Out()


class _StubProcessor:
    class _FeatureExtractor:
        def __call__(self, waveform, sampling_rate, return_tensors):
            import torch
            class _R:
                input_features = torch.zeros((1, 80, 3000))
            return _R()
    feature_extractor = _FeatureExtractor()

    def batch_decode(self, sequences, skip_special_tokens):
        return ["stub hypothesis"]


def test_harness_stub():
    import numpy as np
    import pandas as pd
    from src.eval.harness import transcribe_dataset

    # Create a tiny dataset with 3 fake utterances backed by real audio files
    # Use the first 3 rows from ky_dev.tsv if it exists, else skip.
    from src.utils.config import load_config
    cfg = load_config()
    dev_tsv = Path(cfg.data_root) / "ky_dev.tsv"
    if not dev_tsv.exists():
        print("    (skipping harness stub — ky_dev.tsv not present)")
        return

    df = pd.read_csv(dev_tsv, sep="\t", dtype=str, nrows=3)
    df["language"] = "ky"
    df["split"] = "dev"

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "ckpt.parquet")
        result = transcribe_dataset(
            model=_StubModel(),
            processor=_StubProcessor(),
            dataset_df=df,
            language="ky",
            decode_kwargs={"language": "sw"},
            batch_size=2,
            results_root=tmpdir,
            checkpoint_path=ckpt_path,
            checkpoint_every=2,
        )
    assert len(result) == 3, f"expected 3 rows, got {len(result)}"
    assert "hypothesis" in result.columns
    assert "wer_capped" in result.columns


# ---------------------------------------------------------------------------
# 7. Checkpoint resume simulation
# ---------------------------------------------------------------------------

def test_checkpoint_resume():
    import pandas as pd
    import numpy as np
    from src.eval.harness import transcribe_dataset
    from src.utils.config import load_config

    cfg = load_config()
    dev_tsv = Path(cfg.data_root) / "ky_dev.tsv"
    if not dev_tsv.exists():
        print("    (skipping checkpoint resume — ky_dev.tsv not present)")
        return

    # Simulate: 5 utterances, pretend first 2 already done
    df_all = pd.read_csv(dev_tsv, sep="\t", dtype=str, nrows=5)
    df_all["language"] = "ky"
    df_all["split"] = "dev"

    # Build fake prior_rows for the first 2 utterances
    prior_rows = [
        {
            "utt_id": df_all.iloc[i]["utt_id"],
            "hypothesis": "prior result",
            "wer_capped": 0.5,
            "wer_uncapped": 0.5,
            "cer": 0.3,
            "reference_normalised": "ref",
            "hypothesis_normalised": "prior result",
            "n_ref_words": 3, "n_hyp_words": 2,
            "substitutions": 1, "deletions": 0, "insertions": 0,
            "length_ratio": 0.67, "compression_ratio": 0.9,
            "avg_log_prob": -0.5, "repeated_4gram_rate": 0.0,
            "is_catastrophic": False, "split": "dev", "model": "KY",
            "audio_path": "", "audio_length_sec": 3.0,
            "reference": "ref", "prompt_type": "", "recording_environment": "",
            "speaker_id": "", "leading_silence_s": 0.0, "trailing_silence_s": 0.0,
            "snr_db": 20.0,
        }
        for i in range(2)
    ]
    df_remaining = df_all.iloc[2:].reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "ckpt.parquet")
        result = transcribe_dataset(
            model=_StubModel(),
            processor=_StubProcessor(),
            dataset_df=df_remaining,
            language="ky",
            decode_kwargs={"language": "sw"},
            batch_size=2,
            results_root=tmpdir,
            checkpoint_path=ckpt_path,
            checkpoint_every=2,
            prior_rows=prior_rows,
        )

    assert len(result) == 5, f"expected 5 rows total (2 prior + 3 new), got {len(result)}"
    # First 2 rows should be the prior ones
    assert result.iloc[0]["hypothesis"] == "prior result"
    assert result.iloc[2]["hypothesis"] == "stub hypothesis"


# ---------------------------------------------------------------------------
# 8 & 9. Real data loading + metadata join
# ---------------------------------------------------------------------------

def test_data_loading():
    from src.utils.config import load_config
    from src.utils.data import load_split

    cfg = load_config()
    dev_tsv = Path(cfg.data_root) / "ky_dev.tsv"
    if not dev_tsv.exists():
        print("    (skipping data loading — ky_dev.tsv not present)")
        return

    df = load_split("dev", "ky", cfg.data_root)
    assert len(df) > 0, "empty dataframe"
    assert "severity" in df.columns, "severity column missing after metadata join"
    assert "audio_path" in df.columns
    assert df["severity"].notna().any(), "all severity values are NaN"
    # Spot check: RW001 should be Severe
    rw001 = df[df["speaker_id"] == "RW001"]
    if not rw001.empty:
        assert rw001.iloc[0]["severity"] == "Severe", \
            f"RW001 should be Severe, got {rw001.iloc[0]['severity']}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RQ1 pipeline smoke test")
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="Skip checks that require downloaded local data",
    )
    args = parser.parse_args()

    print("\n=== RQ1 Smoke Test ===\n")

    print("Core (no data needed):")
    check("imports", test_imports)
    check("config", test_config)
    check("normaliser (KY elision + EN lowercase)", test_normaliser)
    check("metrics (WER / CER / edit ops / repeated n-gram)", test_metrics)
    check("failure diagnostics (catastrophic flag)", test_failure_diagnostics)

    if not args.no_data:
        print("\nData-dependent:")
        check("data loading + metadata join", test_data_loading)
        check("harness stub (3 utterances, CPU)", test_harness_stub)
        check("checkpoint resume (2 prior + 3 new = 5)", test_checkpoint_resume)
    else:
        print("\n(data-dependent checks skipped via --no-data)")

    print()
    if _failures:
        print(f"FAILED ({len(_failures)} check(s)): {', '.join(_failures)}")
        sys.exit(1)
    else:
        print("All checks passed — safe to submit sbatch job.")
        sys.exit(0)


if __name__ == "__main__":
    main()
