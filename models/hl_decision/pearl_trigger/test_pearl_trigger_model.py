import os
import sys

from torch import nn
from torch.utils.data import DataLoader

from models.hl_decision.hl_decision_datasets import FEATURES
from models.torch_dataset import TorchDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import xgboost as xgb
import numpy as np
from sklearn.metrics import log_loss
import torch
import matplotlib.pyplot as plt
import shap
import pandas as pd

from models.hl_decision.train_hl_decision_model import TrainXgbModel
from models.hl_decision.pearl_trigger.pearl_trigger_datasets import PearlTriggerDataset
from models.hl_decision.pearl_trigger.train_pearl_trigger_model import CONTEXT, PearlTriggerLSTM, SEQ_LEN, BATCH_SIZE, \
    TrainPearlTorchModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pearl_trigger_xgb_7.json")
SHAP_BACKGROUND_SAMPLES = 128
SHAP_EVAL_SAMPLES = 512


def eval_xgb():
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Skipping: missing model file {model_path}")
        return

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    pearl_trigger_dataset = PearlTriggerDataset(use_cache=True, build_cache=False, pearl_type="offensive")
    x_test, y_test = pearl_trigger_dataset.get_test_X_Y(transition_context=CONTEXT, pearl_type="offensive")
    y_test = np.asarray(y_test, dtype=np.int64).reshape(-1)

    y_prob = model.predict_proba(x_test)
    loss = log_loss(y_test, y_prob)
    ppl = np.exp(loss)
    print(f"Test loss: {loss:.6f} | ppl: {ppl:.6f}")

    accuracy = model.score(x_test, y_test)
    print(f"Accuracy: {accuracy:.6f}")

    p_thres = 0.8
    y_pred = (y_prob[:, 1] > float(p_thres)).astype(np.int64)

    print(y_pred.sum(), y_test.sum())

    print(f"p_thres: {p_thres}")
    TrainXgbModel.print_confusion_matrix(y_test, y_pred)

    model.get_booster().feature_names = FEATURES

    xgb.plot_importance(model.get_booster(), importance_type='gain', max_num_features=30)
    # plt.show()

    x_eval_df = pd.DataFrame(x_test, columns=FEATURES)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_eval_df)

    # shap.summary_plot(shap_values, x_eval_df)


MODEL_CLASSES = {
    "PearlTriggerLSTM": PearlTriggerLSTM,
}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model(path, device=None):
    if device is None:
        device = get_device()
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for name, model_class in MODEL_CLASSES.items():
            setattr(main_module, name, model_class)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_class_name = checkpoint.get("model_class", "PearlTriggerLSTM")
        model_class = MODEL_CLASSES[model_class_name]
        model = model_class(
            input_dim=checkpoint.get("input_dim"),
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


def sequence_feature_names(seq_len, features):
    return [f"t-{seq_len - 1 - tick}:{feature}" for tick in range(seq_len) for feature in features]


def print_metrics(y_true, y_pred, p_threshold=0.5, mask=None, name="all"):
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        print(f"{name} | samples=0")
        return

    y_target = TrainPearlTorchModel.pearl_logits_to_class(y_pred, p_threshold=p_threshold).numpy()
    acc = np.mean(y_true.reshape(-1) == y_target.reshape(-1))

    criterion = nn.CrossEntropyLoss()
    loss = criterion(torch.tensor(y_pred), torch.tensor(y_true).reshape(-1).long())

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

def eval_torch(model_path=None):
    if model_path is None:
        model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Skipping: missing model file {model_path}")
        return

    device = get_device()

    pearl_trigger_dataset = PearlTriggerDataset(use_cache=True, build_cache=False, pearl_type="offensive")
    test_X, test_Y = pearl_trigger_dataset.get_test_X_Y(
        seq_len=SEQ_LEN,
        transition_context=CONTEXT,
        pearl_type="offensive",
    )
    test_dataset = TorchDataset(test_X, test_Y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(model_path)
    y_true, y_pred = collect_predictions(model, test_loader, device)

    for p_thres in np.linspace(0.4, 0.8, 8):
        print(f"\np_threshold = {p_thres}")
        y_actions = TrainPearlTorchModel.pearl_logits_to_class(y_pred, p_threshold=p_thres)

        print_metrics(y_true, y_pred, p_threshold=p_thres)
        TrainPearlTorchModel.print_confusion_matrix(y_true, y_actions)

    if len(test_X) == 0:
        return

    model_cpu = model.cpu()
    background_X = torch.as_tensor(test_X[:min(SHAP_BACKGROUND_SAMPLES, len(test_X))], dtype=torch.float32)
    explain_X = torch.as_tensor(test_X[:min(SHAP_EVAL_SAMPLES, len(test_X))], dtype=torch.float32)
    explainer = shap.DeepExplainer(model_cpu, data=background_X)
    shap_values = explainer.shap_values(explain_X, check_additivity=False)

    if isinstance(shap_values, list):
        class_one_shap = shap_values[1]
    elif shap_values.ndim == explain_X.ndim + 1:
        class_one_shap = shap_values[..., 1]
    else:
        class_one_shap = shap_values
    shap.summary_plot(
        class_one_shap.reshape(class_one_shap.shape[0], -1),
        explain_X.numpy().reshape(explain_X.shape[0], -1),
        feature_names=sequence_feature_names(test_X.shape[1], FEATURES),
    )

if __name__ == "__main__":
    # eval_xgb()

    model_path = os.path.join(BASE_DIR, "pearl_trigger_lstm_5.pth")
    eval_torch(model_path)
