"""Shared helpers for beat-based image generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Generator

import librosa
import numpy as np
from PIL import Image

COLORMAP_POSITIONS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
COLORMAP_RED = np.array([0.268, 0.286, 0.001, 0.121, 0.445, 0.993], dtype=np.float32)
COLORMAP_GREEN = np.array([0.004, 0.139, 0.399, 0.518, 0.698, 0.906], dtype=np.float32)
COLORMAP_BLUE = np.array([0.329, 0.458, 0.818, 0.877, 0.713, 0.144], dtype=np.float32)


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale matrix into [0, 1] for consistent colormap lookups."""
    min_val = float(np.min(matrix))
    max_val = float(np.max(matrix))
    span = max_val - min_val
    if span <= 0:
        return np.zeros_like(matrix, dtype=np.float32)
    return ((matrix - min_val) / span).astype(np.float32)


def apply_colormap(normalized: np.ndarray) -> np.ndarray:
    """Map normalized grayscale data into an RGB uint8 tensor."""
    flat = np.clip(normalized.ravel(), 0.0, 1.0)
    r = np.interp(flat, COLORMAP_POSITIONS, COLORMAP_RED)
    g = np.interp(flat, COLORMAP_POSITIONS, COLORMAP_GREEN)
    b = np.interp(flat, COLORMAP_POSITIONS, COLORMAP_BLUE)
    stacked = np.stack((r, g, b), axis=-1)
    return (np.reshape(stacked, (*normalized.shape, 3)) * 255.0).astype(np.uint8)


def feature_to_image(window: np.ndarray, sr: int, feature: str, size: tuple[int, int]) -> Image.Image:
    """Render one audio window to an RGB image for the requested feature."""
    if feature == "mel":
        spec = librosa.feature.melspectrogram(
            y=window, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=sr / 2
        )
        db = librosa.power_to_db(spec, ref=np.max)
    else:
        chroma = librosa.feature.chroma_stft(
            y=window, sr=sr, n_chroma=128, n_fft=2048, hop_length=512
        )
        db = librosa.amplitude_to_db(chroma, ref=np.max)
    normalized = normalize(db)
    colorized = apply_colormap(normalized)
    image = Image.fromarray(colorized, mode="RGB")
    return image.resize(size, resample=Image.BILINEAR)


def sliding_windows(
    audio: np.ndarray, window_samples: int, hop_samples: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield indices and padded chunks from the audio stream."""
    total = len(audio)
    if total == 0:
        return
    pos = 0
    while pos < total:
        end = pos + window_samples
        chunk = audio[pos:end]
        if chunk.shape[0] < window_samples:
            padding = window_samples - chunk.shape[0]
            chunk = np.pad(chunk, (0, padding), mode="constant")
        yield pos, chunk
        pos += hop_samples


def samples_for_duration(duration: float, sr: int) -> int:
    """Ensure we never produce zero-length windows."""
    return max(1, int(math.ceil(duration * sr)))


def format_run_filename(run_number: int, frame_idx: int, feature: str) -> str:
    """Consistent naming used by both test-beats runs and the GUI."""
    base = f"run{run_number:03d}_frame{frame_idx:05d}"
    return f"{base}_{feature}.png"
