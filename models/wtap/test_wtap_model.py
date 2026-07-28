import os
import sys

from torch import nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import shap

try:
    from train_wtap_model import (
        WtapStackedMLP,
        Wtap1dCNN,
        BATCH_SIZE,
        SEQ_LEN,
        get_device,
        print_class_distribution,
        wtap_to_class,
        get_X_Y
    )
except ModuleNotFoundError:
    from models.wtap.train_wtap_model import (
        WtapStackedMLP,
        Wtap1dCNN,
        BATCH_SIZE,
        SEQ_LEN,
        get_device,
        print_class_distribution,
        wtap_to_class,
        get_X_Y
)
from models.torch_dataset import TorchDataset
from models.wtap.wtap_datasets import WtapDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "wtap_cnn_3.pth")

MODEL_CLASSES = {
    "WtapStackedMLP": WtapStackedMLP,
    "Wtap1dCNN": Wtap1dCNN
}


def checkpoint_model_input_dim(model_class_name, checkpoint):
    state_dict = checkpoint["model_state_dict"]
    first_linear = state_dict.get("layers.0.weight")
    if first_linear is not None:
        return int(first_linear.shape[1])
    return int(checkpoint["input_dim"])


def load_model(path, device):
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for name, model_class in MODEL_CLASSES.items():
            setattr(main_module, name, model_class)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_class_name = checkpoint.get("model_class", "WtapStackedMLP")
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
    label = "aForward"

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
    plt.title(f"wtap next-error prediction over time")
    plt.legend()
    plt.show()

def print_metrics(y_true, y_pred, mask=None, name="all"):
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        print(f"{name} | samples=0")
        return

    y_target = logits_to_wtap(y_pred)
    acc = np.mean(y_true.reshape(-1) == y_target.reshape(-1))

    criterion = nn.CrossEntropyLoss()
    loss = criterion(torch.tensor(y_pred), wtap_to_class(torch.tensor(y_true)))

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

def logits_to_wtap(y_pred):
    return np.argmax(y_pred, axis=1) - 1

def evaluate_model():
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Skipping: missing model file {model_path}")
        return

    wtap_dataset = WtapDataset(use_cache=True, build_cache=False)
    test_X, test_Y = wtap_dataset.get_test_X_Y(seq_len=SEQ_LEN)
    test_dataset = TorchDataset(test_X, test_Y)
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    model = load_model(model_path, device)
    y_true, y_pred = collect_predictions(model, loader, device)
    y_actions = logits_to_wtap(y_pred)

    # baseline_rule(test_dataset)
    print_switch_metrics(y_pred, test_dataset)

    print_metrics(y_true, y_pred)
    print_class_distribution(test_dataset, "test")
    print_confusion_matrix(y_true, y_actions)

    plot_axis(y_true, y_actions)

    explainer = shap.DeepExplainer(model.cpu(), torch.tensor(test_X))

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

    y_actions = logits_to_wtap(y_pred)
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

def print_confusion_matrix(y_true_wtap, y_pred_wtap):
    true_cls = (y_true_wtap.reshape(-1) + 1).astype(int)
    pred_cls = (y_pred_wtap.reshape(-1) + 1).astype(int)

    cm = np.zeros((3, 3), dtype=int)

    for t, p in zip(true_cls, pred_cls):
        cm[t, p] += 1

    print("\nConfusion matrix")
    print("rows = true, cols = pred")
    print("classes: 0=-1, 1=0, 2=1")
    print(cm)

    print("\nPretty:")
    print("              pred -1   pred 0   pred 1")
    labels = ["true -1", "true  0", "true  1"]
    for i in range(3):
        print(f"{labels[i]:>8} {cm[i, 0]:8d} {cm[i, 1]:8d} {cm[i, 2]:8d}")

    print("\nPer-class precision/recall:")
    for c, name in enumerate(["-1", "0", "1"]):
        tp = cm[c, c]
        precision = tp / max(cm[:, c].sum(), 1)
        recall = tp / max(cm[c, :].sum(), 1)
        print(
            f"class {c} / wtap {name}: "
            f"precision={precision:.4f}, recall={recall:.4f}"
        )


if __name__ == "__main__":
    evaluate_model()
