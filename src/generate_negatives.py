# src/generate_negatives.py
import os
import glob
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000
TARGET_LEN = TARGET_SR  # 1 second

RAW_NEG_DIR = "data/raw/negative"
OUT_DIR = "data/processed16k/negative/train"  # we'll later split some to val if needed

def rms_normalize(y, target_dbfs=-25.0):
    eps = 1e-9
    rms = np.sqrt(np.mean(y ** 2) + eps)
    if rms < eps:
        return y
    current_dbfs = 20 * np.log10(rms)
    gain = 10 ** ((target_dbfs - current_dbfs) / 20)
    y = y * gain
    y = np.clip(y, -0.99, 0.99)
    return y

def random_1s_segment(y, sr):
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    if len(y) < TARGET_LEN:
        pad = TARGET_LEN - len(y)
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > TARGET_LEN:
        max_start = len(y) - TARGET_LEN
        start = random.randint(0, max_start)
        y = y[start:start + TARGET_LEN]
    return y

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(RAW_NEG_DIR, "**", "*.wav"), recursive=True)

    idx = 0
    for path in files:
        print(f"Processing {path}")
        # load possibly long file
        y, sr = librosa.load(path, sr=None, mono=True)
        duration = len(y) / sr

        # choose how many segments you want per file
        n_segments = int(duration)  # roughly one per second
        for _ in range(n_segments):
            seg = random_1s_segment(y, sr)
            seg = rms_normalize(seg)
            out_name = f"neg_{idx:06d}.wav"
            out_path = os.path.join(OUT_DIR, out_name)
            sf.write(out_path, seg, TARGET_SR)
            idx += 1

if __name__ == "__main__":
    main()
