# src/train.py
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import WakeWordDataset
from model import DaryWakeWordCNN
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
PATIENCE = 7

EXPERIMENT_DIR = "experiments/dary_cnn"
CHECKPOINT_PATH = os.path.join(EXPERIMENT_DIR, "checkpoints", "best_model.pt")
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

def compute_pos_weight(train_csv):
    df = pd.read_csv(train_csv)
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    pos_weight = n_neg / max(n_pos, 1)
    return torch.tensor(pos_weight, dtype=torch.float32)

def train():
    train_dataset = WakeWordDataset("data/metadata/train.csv", augment=True)
    val_dataset = WakeWordDataset("data/metadata/val.csv", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = DaryWakeWordCNN().to(DEVICE)

    pos_weight = compute_pos_weight("data/metadata/train.csv").to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=2)

    early_stopper = EarlyStopping()

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        # ---- TRAIN ----
        model.train()
        train_losses = []
        y_true_train, y_pred_train = [], []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)  # (B, 1, n_mels, time)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x_batch)           # (B,)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu().numpy()
            y_pred_train.extend(preds.tolist())
            y_true_train.extend(y_batch.cpu().numpy().tolist())

        train_loss = sum(train_losses) / len(train_losses)
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            y_true_train, y_pred_train, average="binary", zero_division=0
        )

        # ---- VALIDATION ----
        model.eval()
        val_losses = []
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().cpu().numpy()
                y_pred_val.extend(preds.tolist())
                y_true_val.extend(y_batch.cpu().numpy().tolist())

        val_loss = sum(val_losses) / len(val_losses)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            y_true_val, y_pred_val, average="binary", zero_division=0
        )

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"Learning rate reduced: {prev_lr:.6g} -> {new_lr:.6g}")
        early_stopper.step(val_loss)

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d} | {dt:.1f}s | "
            f"Train loss {train_loss:.4f} acc {train_acc:.3f} "
            f"prec {train_prec:.3f} rec {train_rec:.3f} f1 {train_f1:.3f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.3f} "
            f"prec {val_prec:.3f} rec {val_rec:.3f} f1 {val_f1:.3f}"
        )

        if early_stopper.should_stop:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train()
