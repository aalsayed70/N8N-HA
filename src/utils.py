# src/utils.py
import os
import glob
import csv

def create_metadata():
    train_csv = "data/metadata/train.csv"
    val_csv = "data/metadata/val.csv"
    os.makedirs(os.path.dirname(train_csv), exist_ok=True)

    # positives
    pos_train_files = glob.glob("data/processed16k/positive/train/*.wav")
    pos_val_files = glob.glob("data/processed16k/positive/val/*.wav")

    # negatives: we’ll do an 80/20 split
    all_neg_files = glob.glob("data/processed16k/negative/train/*.wav")
    all_neg_files = sorted(all_neg_files)
    split_idx = int(0.8 * len(all_neg_files))
    neg_train_files = all_neg_files[:split_idx]
    neg_val_files = all_neg_files[split_idx:]

    def write_csv(path, pos_files, neg_files):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label"])
            for p in pos_files:
                writer.writerow([p, 1])
            for n in neg_files:
                writer.writerow([n, 0])

    write_csv(train_csv, pos_train_files, neg_train_files)
    write_csv(val_csv, pos_val_files, neg_val_files)

if __name__ == "__main__":
    create_metadata()
