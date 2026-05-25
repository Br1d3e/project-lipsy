import os

from sklearn.metrics import log_loss

from models.datasets import Datasets
from models.window_dataset import WindowDataset
from models.hl_decision.hl_decision_datasets import HLDecisionDataset, TARGETS

from matplotlib import pyplot as plt
import numpy as np
import torch
import xgboost as xgb


SEQ_LEN = 20
EPOCHS = 50

class TrainXgbModel:
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    @staticmethod
    def get_X_Y(dataset):
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
        Y_pred = model.predict(val_X)
        loss = log_loss(val_Y, Y_pred)
        ppl = np.exp(loss)
        print(f"Eval loss: {loss:.6f} | ppl: {ppl:.6f}")

        accuracy = model.score(val_X, val_Y)
        print(f"Accuracy: {accuracy:.6f}")

