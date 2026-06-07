import json
import os
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from models.torch_dataset import TorchDataset
from models.aim.aim_datasets import AimDataset

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt


SEQ_LEN = 20
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
LOSS_TYPE = "huber"
HUBER_BETA = 2.0
YAW_WEIGHT = 2.0
PITCH_WEIGHT = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models/aim")


class AimLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128, num_layers=2, output_dim=2):
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

class AimGRU(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, num_layers=1, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            # nn.Linear(hidden_dim, 32),
            # nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class AimSimpleMLP(nn.Module):
    def __init__(self, input_dim=16, output_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.layers(x)

class AimStackedMLP(nn.Module):
    def __init__(self, input_dim=16 * SEQ_LEN, output_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, output_dim),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.layers(x)

class Aim1dCNN(nn.Module):
    def __init__(self, input_dim=16, output_dim=2, conv_channels=None, linear_channels=None,
                 relu_after_final_conv=False):
        super().__init__()
        conv_channels = conv_channels or [32, 64, 128, 256]
        linear_channels = linear_channels or [128, 64]

        layers = []
        in_channels = input_dim
        for idx, out_channels in enumerate(conv_channels):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            if idx < len(conv_channels) - 1 or relu_after_final_conv:
                layers.append(nn.ReLU())
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())

        in_features = conv_channels[-1]
        for out_features in linear_channels:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features

        layers.append(nn.Linear(in_features, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.layers(x)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_train_criterion():
    if LOSS_TYPE == "mse":
        return nn.MSELoss(reduction="none")
    if LOSS_TYPE == "huber":
        return nn.SmoothL1Loss(beta=HUBER_BETA, reduction="none")
    raise ValueError("LOSS_TYPE must be 'mse' or 'huber'")


def weighted_aim_loss(pred, target, criterion):
    loss_by_axis = criterion(pred, target)
    yaw_loss = loss_by_axis[:, 0].mean()
    pitch_loss = loss_by_axis[:, 1].mean()
    return (YAW_WEIGHT * yaw_loss + PITCH_WEIGHT * pitch_loss)

def get_X_Y(dataset):
    if hasattr(dataset, "X") and hasattr(dataset, "Y"):
        return dataset.X, dataset.Y
    X = []
    Y = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X.append(x)
        Y.append(y)
    return torch.stack(X), torch.stack(Y)

def print_target_clamp_suggestions(dataset):
    targets = get_X_Y(dataset)[1]
    yaw = targets[:, 0]
    pitch = targets[:, 1]
    print("Suggested inference clamps from train targets:")
    print(f"  yawStep p1={np.percentile(yaw, 1):.6f} p99={np.percentile(yaw, 99):.6f}")
    print(f"  pitchStep p1={np.percentile(pitch, 1):.6f} p99={np.percentile(pitch, 99):.6f}")


def print_distribution_stats(name, values):
    print(
        f"{name}: "
        f"mean={np.mean(values):.6f} "
        f"std={np.std(values):.6f} "
        f"p50={np.percentile(values, 50):.6f} "
        f"p90={np.percentile(values, 90):.6f} "
        f"p95={np.percentile(values, 95):.6f} "
        f"p99={np.percentile(values, 99):.6f} "
        f"min={np.min(values):.6f} "
        f"max={np.max(values):.6f}"
    )


def save_model_checkpoint(model, save_path, checkpoint_meta=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if checkpoint_meta:
        checkpoint.update(checkpoint_meta)
    torch.save(checkpoint, save_path)


def train(model, dataloader, val_loader, save_path, checkpoint_meta=None,
          epochs=EPOCHS, lr=LEARNING_RATE, patience=25):
    device = get_device()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=10)
    criterion = make_train_criterion()

    best_loss = 1e9
    best_epoch = -1
    best_state_dict = None
    loss_count = 0
    epochs_run = 0
    for epoch in range(epochs):
        epochs_run = epoch + 1
        total_loss = 0.0
        model.train()

        for xb, yb in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = weighted_aim_loss(pred, yb, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        # print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")

        val_loss = eval_model(model, val_loader)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RMSE: {np.sqrt(val_loss):.6f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_model_checkpoint(model, save_path, checkpoint_meta)
            loss_count = 0
            print("model saved")
        else:
            loss_count += 1

        if loss_count > patience:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return {
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
    }


def eval_model(model, dataloader, graph=False, diagnostics=False):
    device = get_device()
    model.to(device)
    model.eval()

    total_loss = 0.0
    yaw_loss = 0.0
    pitch_loss = 0.0
    total_count = 0
    criterion = nn.MSELoss(reduction="sum")

    Y_true = []
    Y_pred = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            Y_true.append(y.cpu().numpy())
            Y_pred.append(pred.cpu().numpy())

            batch_yaw_loss = criterion(pred[:, 0], y[:, 0]).item()
            batch_pitch_loss = criterion(pred[:, 1], y[:, 1]).item()
            yaw_loss += batch_yaw_loss
            pitch_loss += batch_pitch_loss
            total_loss += batch_yaw_loss + batch_pitch_loss
            total_count += y.numel()

    avg_loss = total_loss / total_count
    print(f"Eval Loss: {avg_loss:.6f}")
    print(f"RMSE: {np.sqrt(avg_loss):.6f}")
    sample_count = total_count / 2
    print(f"Yaw RMSE: {np.sqrt(yaw_loss / sample_count):.6f}")
    print(f"Pitch RMSE: {np.sqrt(pitch_loss / sample_count):.6f}")

    Y_true = np.concatenate(Y_true, axis=0)
    Y_pred = np.concatenate(Y_pred, axis=0)
    if diagnostics or graph:
        print_distribution_stats("yaw_true", Y_true[:, 0])
        print_distribution_stats("yaw_pred", Y_pred[:, 0])
        print_distribution_stats("pitch_true", Y_true[:, 1])
        print_distribution_stats("pitch_pred", Y_pred[:, 1])
    if graph:
        yaw_true = Y_true[:, 0]
        yaw_pred = Y_pred[:, 0]
        yaw_error = yaw_pred - yaw_true
        pitch_true = Y_true[:, 1]
        pitch_pred = Y_pred[:, 1]
        pitch_error = pitch_pred - pitch_true

        plt.figure(figsize=(12, 6))
        plt.plot(yaw_true, label="yaw true", linewidth=1)
        plt.plot(yaw_pred, label="yaw pred", linewidth=1, alpha=0.8)
        # plt.plot(yaw_error, label="yaw pred - true", linewidth=1, alpha=0.75)
        plt.title("Aim Yaw Eval")
        plt.xlabel("sample")
        plt.ylabel("delta yaw")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.plot(pitch_true, label="pitch_true", linewidth=1)
        plt.plot(pitch_pred, label="pitch_pred", linewidth=1, alpha=0.8)
        # plt.plot(pitch_error, label="pitch_error", linewidth=1, alpha=0.75)
        plt.title("Aim Pitch Eval")
        plt.xlabel("sample")
        plt.ylabel("delta pitch")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return avg_loss


if __name__ == "__main__":
    aim_dataset = AimDataset(feature_mode="RNN", use_cache=True, build_cache=False)
    train_x, train_y = aim_dataset.get_train_X_Y(seq_len=SEQ_LEN)
    train_dataset = TorchDataset(train_x, train_y)
    if len(train_dataset) == 0:
        raise ValueError("No training data found.")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print_target_clamp_suggestions(train_dataset)

    val_x, val_y = aim_dataset.get_val_X_Y(seq_len=SEQ_LEN)
    val_dataset = TorchDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    sample_x = train_dataset[0][0]
    input_dim = sample_x.numel()
    feature_dim = sample_x.shape[-1]
    model_name = "LSTM 7"
    # architecture = f"{input_dim} -> conv1d 32 -> conv1d 64 -> conv1d 128 -> 256 -> 512 -> avg pool 1 -> linear 256 -> linear 128 -> linear 64 -> 2"
    architecture = f"{input_dim} -> {feature_dim}"
    # MLP input_dim should be sample_x.numel() because MLP flattens the full window.
    # model = AimSimpleMLP(input_dim=sample_x.numel())
    # model = AimStackedMLP(input_dim=input_dim, output_dim=2)
    model = AimLSTM(input_dim=sample_x.shape[-1], output_dim=2)
    # model = AimGRU(input_dim=sample_x.shape[-1], output_dim=2)
    # model = Aim1dCNN(input_dim=sample_x.shape[-1], output_dim=2, conv_channels=[32, 64, 128, 256, 512], linear_channels=[256, 128, 64])

    model_path = os.path.join(BASE_DIR, "aim_lstm_7.pth")
    checkpoint_meta = {
        "model_class": type(model).__name__,
        "model_name": model_name,
        "input_dim": feature_dim,
        "flat_input_dim": input_dim,
        "output_dim": 2,
        "architecture": architecture,
        "seq_len": SEQ_LEN,
    }
    train_info = train(
        model=model,
        dataloader=train_loader,
        val_loader=val_loader,
        save_path=model_path,
        checkpoint_meta=checkpoint_meta,
    )
    print(
        f"Training complete: "
        f"best_epoch={train_info['best_epoch']} "
        f"epochs_run={train_info['epochs_run']} "
        f"best_val_loss={train_info['best_val_loss']:.6f} "
        f"best_val_rmse={np.sqrt(train_info['best_val_loss']):.6f}"
    )

    eval_model(model=model, dataloader=val_loader, graph=True)
    print(f"Config: \n"
          f"  model: {model_name}\n"
          f"  input_dim: {input_dim}\n"
          f"  architecture: {architecture}\n"
          f"  seq_len: {SEQ_LEN}\n"
          f"  batch_size: {BATCH_SIZE}\n"
          f"  learning_rate: {LEARNING_RATE}\n"
          f"  epochs: {EPOCHS}\n"
          f"  loss_type: {LOSS_TYPE}\n"
          f"  huber_beta: {HUBER_BETA}\n"
          f"  yaw_weight: {YAW_WEIGHT}\n"
          f"  pitch_weight: {PITCH_WEIGHT}\n"
          f"  best_epoch: {train_info['best_epoch']}\n"
          f"  best_val_loss: {train_info['best_val_loss']:.6f}\n"
          f"  best_val_rmse: {np.sqrt(train_info['best_val_loss']):.6f}\n")
