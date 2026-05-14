import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from train_aim_model import (
        AimGRU,
        AimLSTM,
        AimSimpleMLP,
        AimStackedMLP,
        Aim1dCNN,
        BATCH_SIZE,
        SEQ_LEN,
        get_device,
    )
except ModuleNotFoundError:
    from models.aim.train_aim_model import (
        AimGRU,
        AimLSTM,
        AimSimpleMLP,
        AimStackedMLP,
        Aim1dCNN,
        BATCH_SIZE,
        SEQ_LEN,
        get_device,
    )
from models.window_dataset import WindowDataset
from models.aim.aim_datasets import AimDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "aim_cnn_3.pth")

MODEL_CLASSES = {
    "AimSimpleMLP": AimSimpleMLP,
    "AimStackedMLP": AimStackedMLP,
    "AimLSTM": AimLSTM,
    "AimGRU": AimGRU,
    "Aim1dCNN": Aim1dCNN
}


def checkpoint_model_input_dim(model_class_name, checkpoint):
    input_dim = checkpoint["input_dim"]
    if model_class_name == "Aim1dCNN":
        state_dict = checkpoint["model_state_dict"]
        conv_weight = state_dict.get("layers.0.weight")
        if conv_weight is not None:
            return int(conv_weight.shape[1])
    return input_dim


def load_model(path, device):
    # Backward compatibility for old checkpoints saved with torch.save(model)
    # while train_aim_model.py was running as __main__.
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for name, model_class in MODEL_CLASSES.items():
            setattr(main_module, name, model_class)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_class_name = checkpoint.get("model_class", "Aim1dCNN")
        model_class = MODEL_CLASSES[model_class_name]
        model = model_class(
            input_dim=checkpoint_model_input_dim(model_class_name, checkpoint),
            output_dim=checkpoint.get("output_dim", 2),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = checkpoint
    model.to(device)
    model.eval()
    return model


def collect_predictions(model, loader, device):
    y_true_all, y_pred_all = [], []

    with (torch.no_grad()):
        for xb, yb in loader:
            pred = model(xb.to(device))
            y_pred_all.append(pred.cpu().numpy())
            y_true_all.append(yb.numpy())

    return np.concatenate(y_true_all, axis=0), np.concatenate(y_pred_all, axis=0)


def finish_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    # if SHOW_PLOTS:
    plt.show()
    plt.close()


def plot_axis(axis, y_true, y_pred):
    label = "yawDiff" if axis == "yaw" else "pitchDiff"

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
    plt.plot(y_true[:], label="True")
    plt.plot(y_pred[:], label="Pred", alpha=0.75)
    plt.title(f"{axis.capitalize()} next-error prediction over time")
    plt.legend()
    plt.show()

def print_axis_metrics(axis, y_true_axis, y_pred_axis, mask=None, name="all"):
    if mask is not None:
        y_true_axis = y_true_axis[mask]
        y_pred_axis = y_pred_axis[mask]

    if len(y_true_axis) == 0:
        print(f"{axis} | {name} | samples=0")
        return

    abs_err = np.abs(y_true_axis - y_pred_axis)

    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean((y_pred_axis - y_true_axis) ** 2))

    print(
        f"{axis} | {name} | samples={len(y_true_axis)} | "
        f"MAE={mae:.4f} | RMSE={rmse:.4f} | "
        f"median AE={np.median(abs_err):.4f} | "
        f"p90 AE={np.percentile(abs_err, 90):.4f} | "
        f"p95 AE={np.percentile(abs_err, 95):.4f} | "
        f"p99 AE={np.percentile(abs_err, 99):.4f} | "
        f"max AE={np.max(abs_err):.4f}"
    )

def evaluate_axis(axis):
    axis_idx = 0 if axis == "yaw" else 1
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Skipping {axis}: missing model file {model_path}")
        return

    aim_dataset = AimDataset()
    test_windows = aim_dataset.get_test_windows(seq_len=SEQ_LEN)
    test_dataset = WindowDataset(test_windows)
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    model = load_model(model_path, device)
    y_true, y_pred = collect_predictions(model, loader, device)
    y_true_axis = y_true[:, axis_idx]
    y_pred_axis = y_pred[:, axis_idx]

    yaw_true = y_true[:, 0]
    pitch_true = y_true[:, 1]

    normal_mask = (np.abs(yaw_true) <= 20.0) & (np.abs(pitch_true) <= 10.0)
    flick_mask = ~normal_mask

    print_axis_metrics(axis, y_true_axis, y_pred_axis, name="all")
    print_axis_metrics(axis, y_true_axis, y_pred_axis, mask=normal_mask, name="normal")
    print_axis_metrics(axis, y_true_axis, y_pred_axis, mask=flick_mask, name="flick/discontinuity")

    yaw_pred = y_pred[:, 0]
    yaw_ae = np.abs(yaw_true - yaw_pred)
    max_ae_idx = np.argmax(yaw_ae)
    print(f"Max yaw ae: yaw_pred = {yaw_pred[max_ae_idx]} | yaw_true = {yaw_true[max_ae_idx]} | ae = {yaw_ae[max_ae_idx]}")
    x_max_ae = loader.dataset.X[max_ae_idx].numpy()
    last_tick_features = x_max_ae[-1]
    yaw_diff = last_tick_features[14]
    pitch_diff = last_tick_features[15]
    distance = last_tick_features[10]
    history_start = max(0, max_ae_idx - 5)
    history = np.stack([x.numpy() for x in loader.dataset.X[history_start:max_ae_idx]])
    print(
        f"yawDiff={yaw_diff} | pitchDiff={pitch_diff} | distance={distance} | "
        # f"\nfeatures prev windows: {history}"
    )


    plot_axis(axis, y_true_axis, y_pred_axis)


if __name__ == "__main__":
    evaluate_axis("yaw")
    evaluate_axis("pitch")
