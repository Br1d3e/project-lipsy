import os
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_aim_model import AXIS_CONFIG, BASE_DIR, AimDataset, AimLSTM, BATCH_SIZE, FOLDER, SEQ_LEN, get_device


PLOT_DIR = os.path.join(BASE_DIR, "eval_plots")
SHOW_PLOTS = os.environ.get("SHOW_PLOTS") == "1"


def load_scalers(path):
    with open(path, "rb") as f:
        scalers = pickle.load(f)
    return scalers["scaler_x"], scalers["scaler_y"]


def load_model(path, device):
    model = AimLSTM(input_dim=8, output_dim=1)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.to(device)
    model.eval()
    return model


def collect_predictions(model, loader, device):
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device))
            y_pred_all.append(pred.cpu().numpy())
            y_true_all.append(yb.numpy())

    return np.concatenate(y_true_all, axis=0), np.concatenate(y_pred_all, axis=0)


def finish_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_axis(axis, y_true, y_pred):
    label = "yawDiff" if axis == "yaw" else "pitchDiff"
    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, s=5, alpha=0.5)
    plt.xlabel(f"True next {label}")
    plt.ylabel(f"Pred next {label}")
    plt.title(f"{axis.capitalize()} model prediction")
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    plt.plot([low, high], [low, high], "r--")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:300], label="True")
    plt.plot(y_pred[:300], label="Pred", alpha=0.75)
    plt.title(f"{axis.capitalize()} next-error prediction over time")
    plt.legend()
    plt.show()


def evaluate_axis(axis):
    config = AXIS_CONFIG[axis]
    model_path = config["model_path"]
    scaler_path = config["scaler_path"]

    if not os.path.exists(model_path):
        print(f"Skipping {axis}: missing model file {model_path}")
        return
    if not os.path.exists(scaler_path):
        print(f"Skipping {axis}: missing scaler file {scaler_path}")
        return

    use_scaler = True
    if use_scaler:
        scaler_x, scaler_y = load_scalers(scaler_path)
        dataset = AimDataset(
            FOLDER,
            axis=axis,
            seq_len=SEQ_LEN,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            fit_scalers=False,
        )
    else:
        dataset = AimDataset(
            FOLDER,
            axis=axis,
            seq_len=SEQ_LEN,
            fit_scalers=False,
        )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    model = load_model(model_path, device)
    y_true, y_pred = collect_predictions(model, loader, device)
    if use_scaler:
        y_true = scaler_y.inverse_transform(y_true).reshape(-1)
        y_pred = scaler_y.inverse_transform(y_pred).reshape(-1)

    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"{axis} | samples={len(dataset)} | MAE={mae:.4f} | RMSE={rmse:.4f}")

    plot_axis(axis, y_true, y_pred)


if __name__ == "__main__":
    evaluate_axis("yaw")
    evaluate_axis("pitch")
