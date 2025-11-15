from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio.functional as F


def load_audio(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load audio as float32 in shape (channels, samples)."""
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.astype(np.float32).T  # (channels, samples)
    return audio, sr


def save_audio(path: str | Path, audio: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim == 1:
        data = audio
    else:
        data = audio.T
    sf.write(str(path), data, sr, subtype="PCM_24")


def loudness_normalize(
    audio: np.ndarray,
    sr: int,
    target_lufs: float,
    true_peak_db: float = -1.0,
    block_size: float = 0.400,
) -> Tuple[np.ndarray, float]:
    """Normalize integrated loudness to the target LUFS."""
    meter = pyln.Meter(sr, block_size=block_size)
    if audio.ndim == 1:
        reference = audio
    else:
        reference = audio.T
    loudness = meter.integrated_loudness(reference)
    gain_db = target_lufs - float(loudness)
    gain = 10 ** (gain_db / 20.0)
    normalized = audio * gain
    peak = float(np.max(np.abs(normalized))) + 1e-12
    peak_dbfs = 20 * math.log10(peak)
    max_lin = 10 ** (true_peak_db / 20.0)
    if peak > max_lin:
        normalized = normalized * (max_lin / peak)
    return normalized.astype(np.float32), float(loudness)


def highpass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    if cutoff <= 0:
        return audio
    tensor = torch.from_numpy(audio.copy())
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    filtered = F.highpass_biquad(tensor, sr, cutoff)
    return filtered.squeeze(0).numpy() if audio.ndim == 1 else filtered.numpy()


def high_shelf(audio: np.ndarray, sr: int, freq: float, gain_db: float, q: float = 0.707) -> np.ndarray:
    if abs(gain_db) < 1e-3:
        return audio
    tensor = torch.from_numpy(audio.copy())
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    filtered = F.equalizer_biquad(tensor, sr, freq, gain_db, q)
    return filtered.squeeze(0).numpy() if audio.ndim == 1 else filtered.numpy()


def rms_db(audio: np.ndarray) -> float:
    value = float(np.sqrt(np.mean(np.square(audio)))) + 1e-12
    return 20.0 * math.log10(value)


def ensure_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
