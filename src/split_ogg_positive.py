# src/split_ogg_positive.py

import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# target settings
TARGET_SR = 16000
TARGET_LEN = TARGET_SR       # 1 second
TOP_DB = 30                  # silence threshold for splitting
MIN_SEGMENT_SEC = 0.25       # ignore tiny blips
RAW_POS_DIR = "data/raw/positive"
PROCESSED_POS_TRAIN_DIR = "data/processed16k/positive/train"
PROCESSED_POS_VAL_DIR = "data/processed16k/positive/val"

# adjust these to your desired train/val speaker split
TRAIN_SPEAKERS = {"speaker01", "speaker02", "speaker03",
                  "speaker04", "speaker05", "speaker06", "speaker07"}
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
        start = (len(y) - TARGET_LEN) // 2
        y = y[start:start + TARGET_LEN]
    return y


def process_speaker_file(ogg_path: Path):
    speaker_id = ogg_path.stem  # e.g. "speaker01"
    print(f"\n=== Processing {speaker_id} from {ogg_path.name} ===")

    # Load OGG as mono, 16 kHz
    y, sr = librosa.load(str(ogg_path), sr=TARGET_SR, mono=True)

    # Find non-silent intervals: list of [start_frame, end_frame]
    intervals = librosa.effects.split(y, top_db=TOP_DB)
    print(f"Detected {len(intervals)} non-silent segments before filtering.")

    # Decide output split directory
    if speaker_id in TRAIN_SPEAKERS:
        out_root = PROCESSED_POS_TRAIN_DIR
    elif speaker_id in VAL_SPEAKERS:
        out_root = PROCESSED_POS_VAL_DIR
    else:
        # default to train if not listed
        out_root = PROCESSED_POS_TRAIN_DIR

    os.makedirs(out_root, exist_ok=True)

    n_saved = 0
    for i, (start, end) in enumerate(intervals, start=1):
        seg = y[start:end]
        seg_dur = len(seg) / TARGET_SR

        # skip segments too short to be useful
        if seg_dur < MIN_SEGMENT_SEC:
            continue

        seg = rms_normalize(seg)
        seg = pad_or_trim(seg)

        out_name = f"{speaker_id}_dary_{i:03d}.wav"
        out_path = Path(out_root) / out_name
        sf.write(out_path, seg, TARGET_SR)
        n_saved += 1

    print(f"Saved {n_saved} utterances for {speaker_id}.")


def main():
    raw_dir = Path(RAW_POS_DIR)
    ogg_files = sorted(raw_dir.glob("*.ogg"))

    if not ogg_files:
        print(f"No OGG files found in {raw_dir}.")
        return

    for ogg_path in ogg_files:
        process_speaker_file(ogg_path)


if __name__ == "__main__":
    main()
