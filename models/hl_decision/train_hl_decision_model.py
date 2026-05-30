import os

from sklearn.metrics import log_loss

from models.datasets import Datasets
from models.window_dataset import WindowDataset
from models.hl_decision.hl_decision_datasets import HLDecisionDataset, TARGETS

from matplotlib import pyplot as plt
import numpy as np
import torch
import xgboost as xgb


SEQ_LEN = 1
EPOCHS = 50

class TrainXgbModel:
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    @staticmethod
    def get_X_Y(dataset):
        if isinstance(dataset, tuple) and len(dataset) == 2:
            X, Y = dataset
            X = torch.as_tensor(X, dtype=torch.float32)
            if X.dim() > 2:
                X = X.flatten(start_dim=1)
            Y = torch.as_tensor(Y).long().view(-1)
            return X, Y

        if len(dataset) == 0:
            raise ValueError("empty dataset: no high-level decision windows were generated")

        X = []
        Y = []
        for i in range(len(dataset)):
            x, y = dataset[i]
            X.append(x)
            Y.append(y)
        X = torch.stack(X).flatten(start_dim=1)
        Y = torch.stack(Y).long().view(-1)
        return X, Y

    def train(self, model, save_path=None):
        if not isinstance(model, xgb.XGBClassifier):
            raise ValueError(f"Invalid model: {model!r}")
        train_X, train_Y = self.get_X_Y(self.train_dataset)
        val_X, val_Y = self.get_X_Y(self.val_dataset)

        model.fit(train_X, train_Y, eval_set=[(val_X, val_Y)])

        print(f"Training Complete.")
        if model.early_stopping_rounds > 0:
            print(f"Best: #{model.best_iteration}, loss: {model.best_score:.6f}")
        print(f"Config: {model.get_xgb_params()}")

        if save_path is not None:
            model.save_model(save_path)
            print(f"Saved model to {save_path}")

        return model

    def eval(self, model):
        if not isinstance(model, xgb.XGBClassifier):
            raise ValueError(f"Invalid model: {model!r}")
        if not model.evals_result():
            raise ValueError(f"Invalid model evals: {model!r}")

        val_X, val_Y = self.get_X_Y(self.val_dataset)
        val_X_np = val_X.detach().cpu().numpy()
        val_Y_np = val_Y.detach().cpu().numpy().astype(np.int64)
        Y_prob = model.predict_proba(val_X_np)
        loss = log_loss(val_Y_np, Y_prob)
        ppl = np.exp(loss)
        print(f"Eval loss: {loss:.6f} | ppl: {ppl:.6f}")

        accuracy = model.score(val_X_np, val_Y_np)
        print(f"Accuracy: {accuracy:.6f}")

    @staticmethod
    def print_confusion_matrix(y_true, y_pred, module_name=""):
        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f"y_true and y_pred length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}")

        cm = np.zeros((2, 2), dtype=int)

        for t, p in zip(y_true, y_pred):
            if t not in (0, 1) or p not in (0, 1):
                continue
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
                f"class {c} / hl decision {module_name} {name}: "
                f"precision={precision:.4f}, recall={recall:.4f}"
            )
