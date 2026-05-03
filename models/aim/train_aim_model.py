import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER = "/Users/br1d3e/Library/CloudStorage/GoogleDrive-wangwangjiman@gmail.com/My Drive/Project Lipsy - Continued/project-lipsy/test/recordings/aim"

AXIS_CONFIG = {
    "yaw": {
        "key": "yawDiffPrev",
        "min": -50.0,
        "max": 50.0,
        "model_path": os.path.join(BASE_DIR, "aim_yaw_model.pth"),
        "scaler_path": os.path.join(BASE_DIR, "aim_yaw_scalers.pkl"),
    },
    "pitch": {
        "key": "PitchDiffPrev",
        "min": -20.0,
        "max": 10.0,
        "model_path": os.path.join(BASE_DIR, "aim_pitch_model.pth"),
        "scaler_path": os.path.join(BASE_DIR, "aim_pitch_scalers.pkl"),
    },
}


class AimDataset(Dataset):
    def __init__(self, folder, axis, seq_len=SEQ_LEN, scaler_x=None, scaler_y=None, fit_scalers=True):
        if axis not in AXIS_CONFIG:
            raise ValueError("axis must be 'yaw' or 'pitch'")

        self.axis = axis
        self.seq_len = seq_len
        self.samples = []
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

        config = AXIS_CONFIG[axis]
        file_features, file_targets, file_valid = self._load_files(folder, config)

        if fit_scalers:
            valid_features = [features[valid] for features, valid in zip(file_features, file_valid) if np.any(valid)]
            valid_targets = [targets[valid] for targets, valid in zip(file_targets, file_valid) if np.any(valid)]
            if not valid_features or not valid_targets:
                raise ValueError(f"No valid {axis} samples found in {folder}")

            self.scaler_x = StandardScaler().fit(np.vstack(valid_features))
            self.scaler_y = StandardScaler().fit(np.vstack(valid_targets))

        self._build_sequences(file_features, file_targets, file_valid)
        print(f"[Aim-{axis} Dataset] Loaded {len(self.samples)} sequences from {folder}")

    def _load_files(self, folder, config):
        file_features, file_targets, file_valid = [], [], []

        for file_name in sorted(os.listdir(folder)):
            if not file_name.endswith(".json"):
                continue

            path = os.path.join(folder, file_name)
            with open(path, "r") as f:
                data = json.load(f)

            data = sorted(data, key=lambda d: d["time"])
            if len(data) <= self.seq_len:
                continue

            features = np.array([
                [d["dx"], d["dy"], d["dz"], d["dist"],
                 d[config["key"]], d["targetVelX"], d["targetVelY"], d["targetVelZ"]]
                for d in data[:-1]
            ], dtype=np.float32)

            targets = np.array([
                [data[i + 1][config["key"]]]
                for i in range(len(data) - 1)
            ], dtype=np.float32)

            valid = np.array([
                config["min"] <= data[i][config["key"]] <= config["max"]
                and config["min"] <= data[i + 1][config["key"]] <= config["max"]
                for i in range(len(data) - 1)
            ], dtype=bool)

            file_features.append(features)
            file_targets.append(targets)
            file_valid.append(valid)

        return file_features, file_targets, file_valid

    def _build_sequences(self, file_features, file_targets, file_valid):
        for features, targets, valid in zip(file_features, file_targets, file_valid):
            if self.scaler_x is not None and self.scaler_y is not None:
                features = self.scaler_x.transform(features)
                targets = self.scaler_y.transform(targets)

            for start in range(len(features) - self.seq_len + 1):
                end = start + self.seq_len
                target_idx = end - 1
                if not np.all(valid[start:end]):
                    continue

                x_seq = features[start:end]
                y_seq = targets[target_idx]
                self.samples.append((x_seq, y_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class AimYawDataset(AimDataset):
    def __init__(self, folder, seq_len=SEQ_LEN, scaler_x=None, scaler_y=None, fit_scalers=True):
        super().__init__(folder, "yaw", seq_len, scaler_x, scaler_y, fit_scalers)


class AimPitchDataset(AimDataset):
    def __init__(self, folder, seq_len=SEQ_LEN, scaler_x=None, scaler_y=None, fit_scalers=True):
        super().__init__(folder, "pitch", seq_len, scaler_x, scaler_y, fit_scalers)


class AimLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_scalers(dataset, path):
    with open(path, "wb") as f:
        pickle.dump({
            "scaler_x": dataset.scaler_x,
            "scaler_y": dataset.scaler_y,
        }, f)


def train(model, dataloader, save_path, epochs=EPOCHS, lr=LEARNING_RATE):
    device = get_device()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=10)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()

        for xb, yb in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def train_axis(axis):
    config = AXIS_CONFIG[axis]
    use_scalers = False
    dataset = AimDataset(FOLDER, axis=axis, seq_len=SEQ_LEN, fit_scalers=use_scalers)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = AimLSTM(input_dim=8, output_dim=1)

    train(model, loader, save_path=config["model_path"])
    if use_scalers:
        save_scalers(dataset, config["scaler_path"])
        print(f"Scalers saved to {config['scaler_path']}")


if __name__ == "__main__":
    train_axis("yaw")
    train_axis("pitch")
