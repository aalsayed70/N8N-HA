# Generate small synthetic noise and music clips for negative raw data.
# This keeps the repo self-contained without downloading large datasets.
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
DURATION = 3.0  # seconds per clip
SAMPLES = int(SR * DURATION)

RAW_NOISE_DIR = Path("data/raw/negative/noise")
RAW_MUSIC_DIR = Path("data/raw/negative/music")

os.makedirs(RAW_NOISE_DIR, exist_ok=True)
os.makedirs(RAW_MUSIC_DIR, exist_ok=True)


def white_noise():
    return np.random.normal(0, 0.2, SAMPLES)


def pink_noise():
    # Voss-McCartney style pink-ish noise
    num_rows = 16
    array = np.random.randn(num_rows, SAMPLES)
    cum = np.cumsum(array, axis=0)
    pink = cum[-1] / num_rows
    pink = pink / (np.max(np.abs(pink)) + 1e-9) * 0.25
    return pink


def band_limited_noise(low_hz=300, high_hz=3000):
    freqs = np.fft.rfftfreq(SAMPLES, 1 / SR)
    spectrum = np.zeros_like(freqs)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum[mask] = np.random.rand(mask.sum())
    # random phase
    phase = np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
    spectrum = spectrum * phase
    noise = np.fft.irfft(spectrum, n=SAMPLES)
    noise = noise / (np.max(np.abs(noise)) + 1e-9) * 0.25
    return noise.real


def synth_music():
    t = np.linspace(0, DURATION, SAMPLES, endpoint=False)
    # Random chord from a small pool
    base_freqs = random.choice([
        [220, 277, 330],  # A major-ish
        [196, 247, 294],  # G-ish
        [262, 330, 392],  # C-ish
        [247, 311, 370],  # Bbm-ish
    ])
    signal = sum(np.sin(2 * np.pi * f * t) for f in base_freqs)
    # Add simple amplitude modulation for texture
    trem = 0.5 * (1 + 0.5 * np.sin(2 * np.pi * random.uniform(3, 6) * t))
    signal = signal * trem
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.3
    # Add a bit of noise
    signal += 0.02 * np.random.randn(SAMPLES)
    return np.clip(signal, -0.99, 0.99)


def save_wavs(out_dir: Path, generator_fn, count: int):
    for i in range(count):
        y = generator_fn()
        y = np.clip(y, -0.99, 0.99).astype(np.float32)
        sf.write(out_dir / f"{generator_fn.__name__}_{i:03d}.wav", y, SR)


def main():
    # A handful of variants for each type
    save_wavs(RAW_NOISE_DIR, white_noise, 5)
    save_wavs(RAW_NOISE_DIR, pink_noise, 5)
    save_wavs(RAW_NOISE_DIR, band_limited_noise, 5)

    save_wavs(RAW_MUSIC_DIR, synth_music, 10)
    print(f"Wrote synthetic noise to {RAW_NOISE_DIR} and music to {RAW_MUSIC_DIR}")


if __name__ == "__main__":
    main()
