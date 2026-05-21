import os
import sys

from torch import nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from train_blockhit_model import (
        BlockhitStackedMLP,
        Blockhit1dCNN,
        BATCH_SIZE,
        SEQ_LEN,
        get_device,
        print_class_distribution,
        get_X_Y
    )
except ModuleNotFoundError:
    from models.blockhit.train_blockhit_model import (
        BlockhitStackedMLP,
        Blockhit1dCNN,
        BATCH_SIZE,
        SEQ_LEN,
        get_device,
        print_class_distribution,
        get_X_Y
)
from models.window_dataset import WindowDataset
from models.blockhit.blockhit_datasets import BlockhitDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "blockhit_stacked_mlp_9.pth")

MODEL_CLASSES = {
    "BlockhitStackedMLP": BlockhitStackedMLP,
    "Blockhit1dCNN": Blockhit1dCNN
}


def checkpoint_model_input_dim(model_class_name, checkpoint):
    state_dict = checkpoint["model_state_dict"]
    first_linear = state_dict.get("layers.0.weight")
    if first_linear is not None:
        return int(first_linear.shape[1])
    return int(checkpoint["input_dim"])


def load_model(path, device):
    # Backward compatibility for old checkpoints saved with torch.save(model)
    # while train_aim_model.py was running as __main__.
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for name, model_class in MODEL_CLASSES.items():
            setattr(main_module, name, model_class)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_class_name = checkpoint.get("model_class", "BlockhitStackedMLP")
        model_class = MODEL_CLASSES[model_class_name]
        model = model_class(
            input_dim=checkpoint_model_input_dim(model_class_name, checkpoint),
            output_dim=checkpoint.get("output_dim", 3),
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
    plt.show()
    plt.close()


def plot_axis(y_true, y_pred):
    label = "aUseItem"

    # plt.figure(figsize=(8, 6))
    # plt.scatter(y_true, y_pred, s=5, alpha=0.5)
    # plt.xlabel(f"True next {label}")
    # plt.ylabel(f"Pred next {label}")
    # plt.title(f"strafe model prediction")
    # low = min(y_true.min(), y_pred.min())
    # high = max(y_true.max(), y_pred.max())
    # plt.plot([low, high], [low, high], "r--")
    # plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:500], label="True")
    plt.plot(y_pred[:500], label="Pred", alpha=0.75)
    plt.title(f"blockhit next-error prediction over time")
    plt.legend()
    plt.show()

def print_metrics(y_true, y_pred, mask=None, name="all"):
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        print(f"{name} | samples=0")
        return

    y_target = logits_to_a_use(y_pred)
    acc = np.mean(y_true.reshape(-1) == y_target.reshape(-1))

    criterion = nn.CrossEntropyLoss()
    loss = criterion(torch.tensor(y_pred), torch.tensor(y_target).reshape(-1))

    ppl = np.exp(loss)

    print(
        f"{name} | samples={len(y_true)} | "
        f"acc={acc:.4f} | loss={loss:.4f} | perplexity={ppl:.4f}"
        # f"median AE={np.median(abs_err):.4f} | "
        # f"p90 AE={np.percentile(abs_err, 90):.4f} | "
        # f"p95 AE={np.percentile(abs_err, 95):.4f} | "
        # f"p99 AE={np.percentile(abs_err, 99):.4f} | "
        # f"max AE={np.max(abs_err):.4f}"
    )

def logits_to_a_use(y_pred, p_threshold=0.5):
    y_pred = torch.tensor(y_pred)
    probs = torch.softmax(y_pred, dim=1).numpy()
    return (probs[:, 1] > p_threshold).astype(int)

def evaluate_model():
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Skipping: missing model file {model_path}")
        return

    blockhit_dataset = BlockhitDataset()
    test_windows = blockhit_dataset.get_test_windows(seq_len=SEQ_LEN)
    test_dataset = WindowDataset(test_windows)
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    model = load_model(model_path, device)
    y_true, y_pred = collect_predictions(model, loader, device)
    y_actions = logits_to_a_use(y_pred, p_threshold=0.5)

    # baseline_rule(test_dataset)
    print_switch_metrics(y_pred, test_dataset)

    print_metrics(y_true, y_pred)
    print_class_distribution(test_dataset, "test")
    print_confusion_matrix(y_true, y_actions)

    # plot_axis(y_true, y_actions)

def baseline_rule(dataset):
    X, Y = get_X_Y(dataset)
    x = X[:, -1, :].numpy()
    y_true = Y.reshape(-1).numpy()
    y_rule = np.round(x[:, 6]).reshape(-1)
    print(np.mean(y_true == y_rule))
    return y_rule

def print_switch_metrics(y_pred, dataset):
    X, Y = get_X_Y(dataset)
    x = X[:, -1, :].numpy()
    y_true = Y.reshape(-1).numpy()

    prev = x[:, 6].reshape(-1)

    switch_mask = prev != y_true
    keep_mask = prev == y_true

    print(f"samples={len(y_true)}")
    print(f"switch samples={switch_mask.sum()} ({switch_mask.mean():.2%})")
    print(f"keep samples={keep_mask.sum()} ({keep_mask.mean():.2%})")

    baseline_pred = prev
    print("baseline overall:", np.mean(baseline_pred == y_true))
    print("baseline keep:", safe_mean(baseline_pred[keep_mask] == y_true[keep_mask]))
    print("baseline switch:", safe_mean(baseline_pred[switch_mask] == y_true[switch_mask]))

    y_actions = logits_to_a_use(y_pred)
    print("model overall:", np.mean(y_actions == y_true))
    print("model keep:", safe_mean(y_actions[keep_mask] == y_true[keep_mask]))
    print("model switch:", safe_mean(y_actions[switch_mask] == y_true[switch_mask]))
    switch_confidence = pred_confidence(y_pred[switch_mask])
    print(f"model switch confidence: avg={safe_mean(switch_confidence):.4f} "
          f"max={safe_max(switch_confidence):.4f} "
          f"min={safe_min(switch_confidence):.4f}")


def safe_mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def safe_max(values):
    return float(np.max(values)) if len(values) else float("nan")


def safe_min(values):
    return float(np.min(values)) if len(values) else float("nan")

def pred_confidence(y_pred):
    y_pred = torch.tensor(y_pred)
    probs = torch.softmax(y_pred, dim=1).numpy()
    return probs.max(axis=1)

def print_confusion_matrix(y_true_blockhit, y_pred_blockhit):
    true_cls = (y_true_blockhit).astype(int)
    pred_cls = (y_pred_blockhit).astype(int)

    cm = np.zeros((2, 2), dtype=int)

    for t, p in zip(true_cls, pred_cls):
        cm[t, p] += 1

    print("\nConfusion matrix")
    print("rows = true, cols = pred")
    print("classes: 0, 1")
    print(cm)

    print("\nPretty:")
    print("              pred 0   pred 1")
    labels = ["true  0", "true  1"]
    for i in range(2):
        print(f"{labels[i]:>8} {cm[i, 0]:8d} {cm[i, 1]:8d}")

    print("\nPer-class precision/recall:")
    for c, name in enumerate(["0", "1"]):
        tp = cm[c, c]
        precision = tp / max(cm[:, c].sum(), 1)
        recall = tp / max(cm[c, :].sum(), 1)
        print(
            f"class {c} / blockhit {name}: "
            f"precision={precision:.4f}, recall={recall:.4f}"
        )


if __name__ == "__main__":
    evaluate_model()
