    # src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DaryWakeWordCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # freq/ time halved

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, 1, n_mels, time)
        x = self.features(x)
        x = self.global_pool(x)  # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)      # (B, 1)
        return logits.squeeze(1) # (B,)
