# src/augment_offline.py
import os
import glob
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

TARGET_SR = 16000
TARGET_LEN = TARGET_SR

POS_TRAIN_DIR = "data/processed16k/positive/train"
OUT_DIR = "data/processed16k/positive/train_augmented"
RIR_DIR = "data/raw/rir"     # put some room impulse responses here (wav)
NOISE_DIR = "data/raw/negative/noise"  # reuse noise

os.makedirs(OUT_DIR, exist_ok=True)

def pad_or_trim(y):
    if len(y) < TARGET_LEN:
        pad = TARGET_LEN - len(y)
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > TARGET_LEN:
        start = (len(y) - TARGET_LEN) // 2
        y = y[start:start + TARGET_LEN]
    return y

def load_random_noise():
    noise_files = glob.glob(os.path.join(NOISE_DIR, "*.wav"))
    if not noise_files:
        return None
    path = random.choice(noise_files)
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return y

def add_noise(y, snr_db_range=(5, 20)):
    noise = load_random_noise()
    if noise is None:
        return y
    if len(noise) < len(y):
        # tile noise
        repeats = int(np.ceil(len(y) / len(noise)))
        noise = np.tile(noise, repeats)[:len(y)]
    else:
        start = random.randint(0, len(noise) - len(y))
        noise = noise[start:start + len(y)]

    signal_power = np.mean(y ** 2) + 1e-9
    noise_power = np.mean(noise ** 2) + 1e-9
    snr_db = random.uniform(*snr_db_range)
    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(desired_noise_power / noise_power)
    y_noisy = y + noise * scale
    return y_noisy

def pitch_shift(y, sr, max_steps=2.0):
    steps = random.uniform(-max_steps, max_steps)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def speed_change(y, max_rate=0.15):
    rate = 1.0 + random.uniform(-max_rate, max_rate)
    y = librosa.effects.time_stretch(y, rate)
    y = pad_or_trim(y)
    return y

def random_gain(y, db_range=(-6, 6)):
    gain_db = random.uniform(*db_range)
    gain = 10 ** (gain_db / 20)
    return np.clip(y * gain, -0.99, 0.99)

def load_random_rir():
    rir_files = glob.glob(os.path.join(RIR_DIR, "*.wav"))
    if not rir_files:
        return None
    path = random.choice(rir_files)
    rir, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return rir / np.max(np.abs(rir) + 1e-9)

def apply_rir(y):
    rir = load_random_rir()
    if rir is None:
        return y
    y_conv = fftconvolve(y, rir, mode="full")
    y_conv = y_conv[:len(y)]
    return y_conv

def main():
    wavs = glob.glob(os.path.join(POS_TRAIN_DIR, "*.wav"))
    for wav_path in wavs:
        fname = os.path.basename(wav_path)
        y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

        for i in range(3):  # generate 3 augmented versions per file
            y_aug = y.copy()
            if random.random() < 0.7:
                y_aug = pitch_shift(y_aug, sr)
            if random.random() < 0.7:
                y_aug = speed_change(y_aug)
            if random.random() < 0.7:
                y_aug = apply_rir(y_aug)
            if random.random() < 0.7:
                y_aug = add_noise(y_aug)
            if random.random() < 0.7:
                y_aug = random_gain(y_aug)

            y_aug = pad_or_trim(y_aug)
            out_name = fname.replace(".wav", f"_aug{i}.wav")
            out_path = os.path.join(OUT_DIR, out_name)
            sf.write(out_path, y_aug, TARGET_SR)

if __name__ == "__main__":
    main()
