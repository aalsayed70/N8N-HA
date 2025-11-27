# src/dataset.py
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import random
import soundfile as sf

class WakeWordDataset(Dataset):
    def __init__(self, csv_path, augment=False):
        self.df = pd.read_csv(csv_path)
        self.augment = augment
        self.sample_rate = 16000

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=40,
            center=True,
            power=2.0
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.df)

    def _simple_augment(self, waveform):
        # waveform: (1, T)
        # random gain
        if random.random() < 0.5:
            gain_db = random.uniform(-6, 6)
            waveform = waveform * (10 ** (gain_db / 20))

        # small time shift
        if random.random() < 0.5:
            shift = random.randint(- int(0.1 * self.sample_rate),
                                   int(0.1 * self.sample_rate))
            waveform = torch.roll(waveform, shifts=shift, dims=1)

        # add gaussian noise with low SNR
        if random.random() < 0.5:
            noise = torch.randn_like(waveform)
            snr_db = random.uniform(10, 30)
            signal_power = waveform.pow(2).mean()
            noise_power = noise.pow(2).mean()
            factor = torch.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
            waveform = waveform + noise * factor
        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label = torch.tensor(row["label"], dtype=torch.float32)

        # Use soundfile backend directly to avoid torchcodec dependency in torchaudio 2.9+
        waveform_np, sr = sf.read(filepath, dtype="float32", always_2d=False)
        if waveform_np.ndim == 2:  # stereo -> mono
            waveform_np = waveform_np.mean(axis=1)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)  # (1, T)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if self.augment:
            waveform = self._simple_augment(waveform)

        mel = self.mel_transform(waveform)  # (1, n_mels, time)
        log_mel = self.db_transform(mel)    # log scale

        # normalize per sample
        mean = log_mel.mean()
        std = log_mel.std() + 1e-9
        log_mel = (log_mel - mean) / std

        return log_mel, label
