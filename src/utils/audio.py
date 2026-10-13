"""
Audio loading and analysis utilities for the CDLI Rwandan NSS ASR sprint.

Provides:
- ``leading_trailing_silence``: energy-based leading/trailing silence detection
- ``estimate_snr``: simple power-ratio SNR estimator
- ``vad_trim``: silence trimming using silero-VAD with energy fallback
- ``compute_compression_ratio``: Whisper mel-frame budget ratio

All functions operate on 16 kHz mono float32 waveforms unless otherwise noted.
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError as _np_err:
    raise ImportError(
        "numpy is required for audio utilities. "
        "Install it with: pip install numpy"
    ) from _np_err

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16000) -> "Tuple[Any, int]":
    """Load an audio file and resample to *target_sr* Hz mono float32.

    Parameters
    ----------
    path:
        Path to the audio file (WAV, FLAC, MP3, etc.).
    target_sr:
        Target sample rate in Hz.  Default 16 000.

    Returns
    -------
    tuple[np.ndarray, int]
        Mono float32 waveform and the actual sample rate (*target_sr*).
    """
    import soundfile as sf

    try:
        waveform, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import librosa
        waveform, sr = librosa.load(path, sr=None, mono=True, dtype=np.float32)

    # Mix down to mono
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    return waveform.astype(np.float32), target_sr


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------

def leading_trailing_silence(
    audio_path: str,
    threshold_db: float = -40.0,
    frame_ms: int = 25,
) -> Tuple[float, float]:
    """Measure leading and trailing silence durations via energy-based VAD.

    Parameters
    ----------
    audio_path:
        Path to the audio file.
    threshold_db:
        Energy threshold in dBFS below which a frame is considered silent.
        Default -40 dB.
    frame_ms:
        Frame length in milliseconds.  Default 25 ms.

    Returns
    -------
    tuple[float, float]
        (leading_silence_seconds, trailing_silence_seconds).
    """
    waveform, sr = load_audio(audio_path)
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(waveform) // frame_len

    if n_frames == 0:
        return 0.0, 0.0

    frames = waveform[: n_frames * frame_len].reshape(n_frames, frame_len)
    # Energy per frame in dBFS (add tiny constant to avoid log(0))
    energy_db = 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-10)

    active = energy_db > threshold_db
    active_indices = np.where(active)[0]

    if len(active_indices) == 0:
        # Whole signal is silent
        total_duration = len(waveform) / sr
        return total_duration / 2.0, total_duration / 2.0

    first_active = active_indices[0]
    last_active = active_indices[-1]

    leading = first_active * frame_len / sr
    trailing = (n_frames - 1 - last_active) * frame_len / sr
    return float(leading), float(trailing)


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------

def estimate_snr(audio_path: str) -> float:
    """Estimate SNR in dB using a simple power-ratio approach.

    Signal energy is estimated from the loudest 50 % of frames; noise energy
    from the quietest 10 % of frames.  SNR = 10·log10(signal / noise).

    Parameters
    ----------
    audio_path:
        Path to the audio file.

    Returns
    -------
    float
        Estimated SNR in dB.  Returns 0.0 if estimation fails.
    """
    waveform, sr = load_audio(audio_path)
    frame_len = int(sr * 0.025)  # 25 ms frames
    n_frames = len(waveform) // frame_len
    if n_frames < 2:
        return 0.0

    frames = waveform[: n_frames * frame_len].reshape(n_frames, frame_len)
    power = (frames ** 2).mean(axis=1)

    sorted_power = np.sort(power)
    noise_end = max(1, int(0.10 * n_frames))
    signal_start = int(0.50 * n_frames)

    noise_energy = sorted_power[:noise_end].mean()
    signal_energy = sorted_power[signal_start:].mean()

    if noise_energy <= 0:
        return 0.0

    return float(10.0 * math.log10(max(signal_energy / noise_energy, 1e-10)))


# ---------------------------------------------------------------------------
# VAD trimming
# ---------------------------------------------------------------------------

def vad_trim(audio_path: str, out_path: str, aggressiveness: int = 2) -> str:
    """Trim leading/trailing silence using silero-VAD or energy fallback.

    Attempts to use silero-VAD (torch hub).  If silero-VAD is unavailable or
    raises an error, falls back to energy-threshold-based trimming.

    Parameters
    ----------
    audio_path:
        Path to input audio file.
    out_path:
        Path where trimmed audio will be written (WAV).
    aggressiveness:
        silero-VAD sensitivity (0–3, higher = more aggressive trimming).
        Unused in the energy fallback path.

    Returns
    -------
    str
        Path to the trimmed audio file (*out_path*).
    """
    import soundfile as sf

    waveform, sr = load_audio(audio_path)

    trimmed = _silero_trim(waveform, sr, aggressiveness)
    if trimmed is None:
        trimmed = _energy_trim(waveform, sr)

    sf.write(out_path, trimmed, sr)
    return out_path


def _silero_trim(
    waveform: np.ndarray, sr: int, aggressiveness: int
) -> np.ndarray | None:
    """Try silero-VAD trimming; return None on any failure."""
    try:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
            trust_repo=True,
        )
        (get_speech_timestamps, _, read_audio, *_) = utils

        audio_tensor = torch.tensor(waveform)
        # silero-vad expects 16 kHz
        if sr != 16000:
            warnings.warn("silero-VAD requires 16 kHz; resampling.")
            import librosa
            waveform_16k = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            audio_tensor = torch.tensor(waveform_16k)

        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            threshold=0.3 + aggressiveness * 0.1,
            sampling_rate=16000,
        )
        if not speech_timestamps:
            return waveform  # nothing to trim

        start = speech_timestamps[0]["start"]
        end = speech_timestamps[-1]["end"]
        return audio_tensor[start:end].numpy()

    except Exception as exc:
        logger.debug("silero-VAD unavailable (%s); using energy fallback.", exc)
        return None


def _energy_trim(waveform: np.ndarray, sr: int, threshold_db: float = -40.0) -> np.ndarray:
    """Simple energy-based trimming fallback."""
    frame_len = int(sr * 0.025)
    n_frames = len(waveform) // frame_len
    if n_frames == 0:
        return waveform

    frames = waveform[: n_frames * frame_len].reshape(n_frames, frame_len)
    energy_db = 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-10)
    active = np.where(energy_db > threshold_db)[0]

    if len(active) == 0:
        return waveform

    start_sample = active[0] * frame_len
    end_sample = (active[-1] + 1) * frame_len
    return waveform[start_sample:end_sample]


# ---------------------------------------------------------------------------
# Whisper compression ratio
# ---------------------------------------------------------------------------

def compute_compression_ratio(
    waveform: np.ndarray,
    sr: int,
    hop_length: int = 160,
    n_frames: int = 3000,
) -> float:
    """Compute Whisper mel-frame compression ratio.

    Returns ``len(waveform) / (n_frames * hop_length)`` — i.e. the ratio of
    the actual audio length to the Whisper 30-second mel-frame budget.
    A ratio > 1.0 means the audio is longer than one Whisper chunk.

    Parameters
    ----------
    waveform:
        Audio waveform array.
    sr:
        Sample rate (should be 16 000 Hz).
    hop_length:
        Mel spectrogram hop length in samples.  Default 160 (10 ms at 16 kHz).
    n_frames:
        Number of mel frames in Whisper's 30-second context.  Default 3000.

    Returns
    -------
    float
        Compression ratio.
    """
    budget_samples = n_frames * hop_length
    return len(waveform) / max(budget_samples, 1)
