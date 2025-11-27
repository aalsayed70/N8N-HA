# src/evaluate.py
import os
import torch
from torch.utils.data import DataLoader

from dataset import WakeWordDataset
from model import DaryWakeWordCNN
from sklearn.metrics import roc_curve, auc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "experiments/dary_cnn/checkpoints/best_model.pt"

def compute_far_frr(scores, labels, threshold):
    # scores: probabilities
    preds = (scores >= threshold).astype(int)
    labels = labels.astype(int)

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()

    false_accepts = ((preds == 1) & (labels == 0)).sum()
    false_rejects = ((preds == 0) & (labels == 1)).sum()

    far = false_accepts / max(n_neg, 1)   # False Accept Rate
    frr = false_rejects / max(n_pos, 1)   # False Reject Rate
    return far, frr

def main():
    dataset = WakeWordDataset("data/metadata/val.csv", augment=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = DaryWakeWordCNN().to(DEVICE)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Run training first to create this file."
        )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(y_batch.numpy().tolist())

    import numpy as np
    scores = np.array(all_scores)
    labels = np.array(all_labels)

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        print(
            f"Cannot compute ROC/EER: validation labels contain only one class "
            f"(pos={n_pos}, neg={n_neg}). Please check the dataset."
        )
        return

    # ROC
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")

    # EER (Equal Error Rate, FAR ~ FRR)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer_threshold = thresholds[eer_idx]
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    print(f"EER: {eer:.4f} at threshold {eer_threshold:.3f}")

    # Threshold for target FAR
    target_far = 0.01  # 1%
    best_thr = None
    best_far = None
    best_frr = None
    for thr in np.linspace(0.0, 1.0, 201):
        far, frr = compute_far_frr(scores, labels, thr)
        if far <= target_far:
            best_thr = thr
            best_far = far
            best_frr = frr
            break

    if best_thr is not None:
        print(
            f"Threshold for FAR <= {target_far:.2%}: {best_thr:.3f} "
            f"(FAR={best_far:.4f}, FRR={best_frr:.4f})"
        )
    else:
        print("No threshold reached the target FAR.")

if __name__ == "__main__":
    main()
