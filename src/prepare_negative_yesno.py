# Utility to fetch a small open dataset (YesNo) for negative examples,
# resample/pad to 1s @16kHz, and regenerate train/val metadata including negatives.
import glob
import os
import tarfile
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

TARGET_SR = 16000
TARGET_LEN = TARGET_SR  # 1s clips

YESNO_URL = "http://www.openslr.org/resources/1/waves_yesno.tar.gz"
RAW_SPEECH_DIR = Path("data/raw/negative/speech")
ARCHIVE_PATH = RAW_SPEECH_DIR / "waves_yesno.tar.gz"
EXTRACT_DIR = RAW_SPEECH_DIR / "waves_yesno"

NEG_TRAIN_DIR = Path("data/processed16k/negative/train")
NEG_VAL_DIR = Path("data/processed16k/negative/val")
POS_TRAIN_DIR = Path("data/processed16k/positive/train")
POS_VAL_DIR = Path("data/processed16k/positive/val")
META_DIR = Path("data/metadata")


def pad_or_trim(y: np.ndarray) -> np.ndarray:
    if len(y) < TARGET_LEN:
        pad = TARGET_LEN - len(y)
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > TARGET_LEN:
        start = (len(y) - TARGET_LEN) // 2
        y = y[start:start + TARGET_LEN]
    return y


def download_yesno():
    RAW_SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    if EXTRACT_DIR.exists() and list(EXTRACT_DIR.rglob("*.wav")):
        print("YesNo negatives already present.")
        return

    if not ARCHIVE_PATH.exists():
        print(f"Downloading YesNo negatives from {YESNO_URL} ...")
        urllib.request.urlretrieve(YESNO_URL, ARCHIVE_PATH)
        print(f"Saved archive to {ARCHIVE_PATH}")

    print("Extracting YesNo archive ...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR)
    print(f"Extracted to {EXTRACT_DIR}")


def prepare_negative_wavs(train_ratio: float = 0.8):
    wavs = sorted(EXTRACT_DIR.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No wav files found under {EXTRACT_DIR}")

    NEG_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    NEG_VAL_DIR.mkdir(parents=True, exist_ok=True)

    split_idx = int(len(wavs) * train_ratio)
    for i, wav_path in enumerate(wavs):
        subset_dir = NEG_TRAIN_DIR if i < split_idx else NEG_VAL_DIR
        y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        y = pad_or_trim(y)
        out_path = subset_dir / f"yesno_{i:03d}.wav"
        sf.write(out_path, y, TARGET_SR)
    print(f"Prepared {split_idx} train and {len(wavs) - split_idx} val negative clips.")


def rebuild_metadata():
    META_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for label, split, root in [
        (1, "train", POS_TRAIN_DIR),
        (0, "train", NEG_TRAIN_DIR),
        (1, "val", POS_VAL_DIR),
        (0, "val", NEG_VAL_DIR),
    ]:
        for wav_path in glob.glob(str(root / "*.wav")):
            records.append({"filepath": wav_path.replace("\\", "/"), "label": label, "split": split})

    if not records:
        raise RuntimeError("No audio files found to write metadata.")

    df = pd.DataFrame(records)
    for split in ["train", "val"]:
        split_df = df[df["split"] == split][["filepath", "label"]].reset_index(drop=True)
        out_path = META_DIR / f"{split}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"Wrote {len(split_df)} rows to {out_path}")


def main():
    download_yesno()
    prepare_negative_wavs()
    rebuild_metadata()


if __name__ == "__main__":
    main()
