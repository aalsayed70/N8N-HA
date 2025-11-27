# src/preprocess_audio.py
import os
import glob
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000
TARGET_LEN = TARGET_SR  # 1 second
TOP_DB = 30             # silence threshold in dB

RAW_POS_DIR = "data/raw/positive"
PROCESSED_POS_TRAIN_DIR = "data/processed16k/positive/train"
PROCESSED_POS_VAL_DIR = "data/processed16k/positive/val"

# choose speaker split manually here
TRAIN_SPEAKERS = {"speaker01", "speaker02", "speaker03", "speaker04",
                  "speaker05", "speaker06", "speaker07"}
VAL_SPEAKERS = {"speaker08", "speaker09"}

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

def pad_or_trim(y):
    if len(y) < TARGET_LEN:
        pad = TARGET_LEN - len(y)
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > TARGET_LEN:
        # center-crop
        start = (len(y) - TARGET_LEN) // 2
        y = y[start:start + TARGET_LEN]
    return y

def process_file(in_path, out_path):
    y, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)

    # trim silence
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)

    # avoid empty clips
    if len(y) < 100:
        return False

    y = rms_normalize(y)
    y = pad_or_trim(y)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, y, TARGET_SR)
    return True

def main():
    for speaker_dir in sorted(Path(RAW_POS_DIR).glob("speaker*")):
        speaker_id = speaker_dir.name
        print("Processing", speaker_id)
        wavs = glob.glob(str(speaker_dir / "*.wav"))
        for wav_path in wavs:
            fname = os.path.basename(wav_path)
            if speaker_id in TRAIN_SPEAKERS:
                out_dir = PROCESSED_POS_TRAIN_DIR
            elif speaker_id in VAL_SPEAKERS:
                out_dir = PROCESSED_POS_VAL_DIR
            else:
                # default: train
                out_dir = PROCESSED_POS_TRAIN_DIR

            out_path = os.path.join(out_dir, fname)
            process_file(wav_path, out_path)

if __name__ == "__main__":
    main()
