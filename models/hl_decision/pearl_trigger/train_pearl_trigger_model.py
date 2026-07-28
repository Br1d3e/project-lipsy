import os

import torch
from matplotlib import pyplot as plt
from torch import nn

from models.hl_decision.hl_decision_datasets import FEATURES
from models.hl_decision.pearl_trigger.pearl_trigger_datasets import PearlTriggerDataset
from models.hl_decision.train_hl_decision_model import TrainXgbModel, TrainTorchModel
from models.torch_dataset import TorchDataset

import xgboost as xgb
import numpy as np
import shap
import pandas as pd
from sklearn.metrics import log_loss


BEFORE_CONTEXT = 20
AFTER_CONTEXT = 10
SEQ_LEN = 20
LEARNING_RATE = 0.001
EPOCHS = 150
PATIENCE = 25
CLASS_WEIGHTS = torch.tensor((1.0, 5.0), dtype=torch.float32)
BATCH_SIZE = 128

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pearl_trigger_xgb_6.json")


class TrainPearlXgb(TrainXgbModel):
    def __init__(self, train_data, val_data):
        self.train_dataset = train_data
        self.val_dataset = val_data
        super().__init__(self.train_dataset, self.val_dataset)

    def train(self, model=None, save_path=None):
        return TrainXgbModel.train(self, model, save_path)

    def eval(self, model=None, show_plot=False):
        TrainXgbModel.eval(self, model)
        x_eval, y_eval = self.get_X_Y(self.val_dataset)
        x_eval = x_eval.detach().cpu().numpy()
        y_eval = y_eval.detach().cpu().numpy().astype(np.int64)

        p = model.predict_proba(x_eval)[:, 1]

        for p_thres in np.linspace(0.2, 0.8, 13):
            y_pred = (p > float(p_thres)).astype(np.int64)
            print("\np_thres=", p_thres)
            print("pred positives:", y_pred.sum())
            self.print_confusion_matrix(y_eval, y_pred)

        if np.any(y_eval == 1):
            print("true pearl prob:", np.percentile(p[y_eval == 1], [0, 1, 5, 25, 50, 75, 95, 99, 100]))
        if np.any(y_eval == 0):
            print("true no-pearl prob:", np.percentile(p[y_eval == 0], [0, 1, 5, 25, 50, 75, 95, 99, 100]))

        eps = 1e-12
        loss = -(y_eval * np.log(p + eps) + (1 - y_eval) * np.log(1 - p + eps))
        idx = np.argsort(loss)[-20:]

        for i in idx:
            print(int(i), "y=", float(y_eval[i]), "p_pearl=", float(p[i]), "loss=", float(loss[i]))

        false_negatives = (y_eval == 1) & (p < 0.2)
        true_positives = (y_eval == 1) & (p > 0.7)
        false_positives = (y_eval == 0) & (p > 0.6)

        # print("false_negatives | col: myHp", x_eval[false_negatives, 0])
        # print(np.unique(x_eval[false_negatives, 0], return_counts=True))
        # print("false_negatives | col: targetDistance", x_eval[false_negatives, 28])
        # print("true_positives | col: myHp", x_eval[true_positives, 0])
        # print("true_positives | col: targetDistance", x_eval[true_positives, 28])
        # print(f"false_positives | col: myHp", x_eval[false_positives, 0])

        booster = model.get_booster()
        booster.feature_names = FEATURES

        if show_plot:
            xgb.plot_importance(booster, importance_type='gain', max_num_features=30)
            plt.show()

        x_eval_df = pd.DataFrame(x_eval, columns=FEATURES)

        bad_pos = (
                (y_eval == 1) &
                (
                        (x_eval[:, FEATURES.index("canPearl")] == 0) |
                        (x_eval[:, FEATURES.index("pearlCooldownTicksRemaining")] > 0) |
                        (x_eval[:, FEATURES.index("targetExists")] == 0)
                    # (x_eval[:, FEATURES.index("targetEstimatedHp")] <= 0)
                )
        )
        print("bad positive pearl samples:", bad_pos.sum(), "/", (y_eval == 1).sum())
        print(x_eval_df.iloc[bad_pos].to_string())

        print("samples where targetHp is 0: ", (x_eval[:, FEATURES.index("targetEstimatedHp")] == 0).sum(), "/",
              (y_eval.shape[0]))

        # print(x_eval_df[240:260].to_string())

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_eval)
        if show_plot:
            shap.summary_plot(shap_values, x_eval_df)

    @staticmethod
    def print_confusion_matrix(y_eval, y_pred):
        TrainXgbModel.print_confusion_matrix(y_eval, y_pred, "pearl")


class PearlTriggerLSTM(nn.Module):
    def __init__(self, input_dim=57, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TrainPearlTorchModel(TrainTorchModel):
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        super().__init__(train_dataset, val_dataset, BATCH_SIZE)

    def train(self, model=None, save_path=None, checkpoint_meta=None):
        if save_path is None:
            save_path = MODEL_PATH
        return super().train(
            model,
            save_path,
            epochs=EPOCHS,
            checkpoint_meta=checkpoint_meta,
            lr=LEARNING_RATE,
            patience=PATIENCE,
            class_weights=CLASS_WEIGHTS,
        )

    def eval(self, model, graph=False, confusion_matrix=False, dataloader=None):
        return super().eval(
            model,
            graph=graph,
            confusion_matrix=confusion_matrix,
            class_weights=CLASS_WEIGHTS,
            module_name="pearl",
            dataloader=dataloader,
        )

    @staticmethod
    def pearl_logits_to_class(logits, p_threshold=0.5):
        return TrainTorchModel.binary_logits_to_class(logits, p_threshold=p_threshold)

    @staticmethod
    def print_confusion_matrix(y_eval, y_pred, module_name="pearl"):
        TrainTorchModel.print_confusion_matrix(y_eval, y_pred, module_name)


if __name__ == "__main__":
    pearl_trigger_dataset = PearlTriggerDataset(use_cache=True, build_cache=False, pearl_type="offensive")
    # train_data = pearl_trigger_dataset.get_train_X_Y(
    #     transition_context=BEFORE_CONTEXT,
    #     after_transition_context=AFTER_CONTEXT,
    #     positive_to_negative=0.01,
    #     sample_mode="even",
    #     pearl_type="offensive",
    # )
    # val_data = pearl_trigger_dataset.get_val_X_Y(
    #     transition_context=BEFORE_CONTEXT,
    #     after_transition_context=AFTER_CONTEXT,
    #     positive_to_negative=None,
    #     sample_mode="random",
    #     pearl_type="offensive",
    # )
    #
    # train_pearl_xgb = TrainPearlXgb(train_data, val_data)
    # model = xgb.XGBClassifier(
    #     booster="gbtree",
    #     objective="binary:logistic",
    #     eval_metric="logloss",
    #
    #     n_estimators=500,
    #     learning_rate=0.05,
    #     max_depth=5,
    #     min_child_weight=5,
    #     gamma=1.0,
    #
    #     scale_pos_weight=3.0,
    #
    #     subsample=1.0,
    #     colsample_bytree=0.85,
    #     reg_lambda=2.0,
    #     reg_alpha=0.0,
    #
    #     early_stopping_rounds=30,
    # )
    # model = train_pearl_xgb.train(model, save_path=MODEL_PATH)
    #
    # train_pearl_xgb.eval(model)


    train_X, train_Y = pearl_trigger_dataset.get_train_X_Y(seq_len=SEQ_LEN, transition_context=BEFORE_CONTEXT,
                                                           after_transition_context=AFTER_CONTEXT, sample_mode="even",
                                                           positive_to_negative=1 / 10,
                                                           pearl_type="offensive")
    val_X, val_Y = pearl_trigger_dataset.get_val_X_Y(seq_len=SEQ_LEN, transition_context=BEFORE_CONTEXT,
                                                    after_transition_context=AFTER_CONTEXT, sample_mode="random",
                                                     positive_to_negative=1 / 15,
                                                     pearl_type="offensive")
    train_dataset = TorchDataset(train_X, train_Y)
    val_dataset = TorchDataset(val_X, val_Y)
    trainer = TrainPearlTorchModel(train_dataset, val_dataset)

    sample_x = torch.tensor(train_dataset[0][0], dtype=torch.float32)
    input_dim = sample_x.shape[-1]
    output_dim = 2

    model = PearlTriggerLSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=output_dim)
    model_name = "Pearl Trigger LSTM 1"
    model_path = os.path.join(BASE_DIR, "pearl_trigger_lstm_4.pth")
    architecture = f"LSTM(input_dim={input_dim}, hidden=64, layers=1) -> 64 -> {output_dim}"

    checkpoint_meta = {
        "model_class": type(model).__name__,
        "model_name": model_name,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "architecture": architecture,
        "seq_len": SEQ_LEN,
    }

    train_info = trainer.train(model, save_path=model_path, checkpoint_meta=checkpoint_meta)

    print(
        f"Training complete: "
        f"best_epoch={train_info['best_epoch']} "
        f"epochs_run={train_info['epochs_run']} "
        f"best_val_loss={train_info['best_val_loss']:.6f} "
        f"best_val_ppl={np.exp(train_info['best_val_loss']):.6f}"
    )

    trainer.eval(model=model, graph=True, confusion_matrix=True)
    print(f"Config: \n"
          f"  model: \n{model}\n"
          f"  before_context: {BEFORE_CONTEXT}\n"
          f"  after_context: {AFTER_CONTEXT}\n"
          # f"  batch_size: {BATCH_SIZE}\n"
          # f"  learning_rate: {LEARNING_RATE}\n"
          # f"  epochs: {EPOCHS}\n"
          f"  loss_type: CrossEntropy\n"
          f"  weights: {CLASS_WEIGHTS.numpy()}\n"
          )
