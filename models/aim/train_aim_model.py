import json
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


# ===== Dataset loader =====
class AimDataset(Dataset):
    def __init__(self, folder, seq_len=20, scaler_x=None, scaler_y=None, fit_scalers=True):
        self.seq_len = seq_len
        self.samples = []
        all_features, all_targets = [], []

        # --- 1st pass: load and collect all data for scaling ---
        for file in os.listdir(folder):
            if not file.endswith(".json"):
                continue
            with open(os.path.join(folder, file), "r") as f:
                data = json.load(f)
            data = sorted(data, key=lambda d: d["time"])

            features = np.array([
                [d["dx"], d["dy"], d["dz"], d["dist"],
                 d["yawDiffPrev"], d["PitchDiffPrev"],
                 d["targetVelX"], d["targetVelY"], d["targetVelZ"]]
                for d in data[:-1]
            ], dtype=np.float32)

            targets = np.array([
                [data[i+1]["yawDiffPrev"], data[i+1]["PitchDiffPrev"]]
                for i in range(len(data)-1)
            ], dtype=np.float32)

            all_features.append(features)
            all_targets.append(targets)

        all_features = np.vstack(all_features)
        all_targets = np.vstack(all_targets)

        # --- Fit or apply global scalers ---
        if fit_scalers:
            self.scaler_x = StandardScaler().fit(all_features)
            self.scaler_y = StandardScaler().fit(all_targets)
        else:
            self.scaler_x = scaler_x
            self.scaler_y = scaler_y

        # --- 2nd pass: sequence construction ---
        for file in os.listdir(folder):
            if not file.endswith(".json"):
                continue
            with open(os.path.join(folder, file), "r") as f:
                data = json.load(f)
            data = sorted(data, key=lambda d: d["time"])

            features = np.array([
                [d["dx"], d["dy"], d["dz"], d["dist"],
                 d["yawDiffPrev"], d["PitchDiffPrev"],
                 d["targetVelX"], d["targetVelY"], d["targetVelZ"]]
                for d in data[:-1]
            ], dtype=np.float32)
            targets = np.array([
                [data[i+1]["yawDiffPrev"], data[i+1]["PitchDiffPrev"]]
                for i in range(len(data)-1)
            ], dtype=np.float32)

            # features = self.scaler_x.transform(features)
            # targets = self.scaler_y.transform(targets)

            for i in range(len(features) - seq_len):
                x_seq = features[i:i+seq_len]
                y_seq = targets[i+seq_len]
                self.samples.append((x_seq, y_seq))

        print(f"✅ Loaded {len(self.samples)} sequences from {folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ===== LSTM Model =====
class AimLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ===== Training =====
def train(model, dataloader, epochs=200, lr=1e-3):
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=10)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for xb, yb in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
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
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

    torch.save(model, "aim_model_unscaled.pth")
    print("✅ Model saved to aim_model.pth")


# ===== Main =====
if __name__ == "__main__":
    # folder = "/Users/br1d3e/Library/CloudStorage/OneDrive-Personal/Project Lipsy - Continued/test/recordings/aim"
    folder = "D:/OneDrive/Project Lipsy - Continued/test/recordings/aim"
    dataset = AimDataset(folder, seq_len=20, fit_scalers=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = AimLSTM(input_dim=9)
    train(model, loader, epochs=200, lr=1e-3)
