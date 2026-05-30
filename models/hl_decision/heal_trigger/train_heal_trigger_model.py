import os

from models.hl_decision.heal_trigger.heal_trigger_datasets import HealTriggerDataset
from models.hl_decision.train_hl_decision_model import TrainXgbModel

import xgboost as xgb
import numpy as np
from sklearn.metrics import log_loss

SEQ_LEN = 1
CONTEXT = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heal_trigger_xgb_4.json")

class TrainHealXgb(TrainXgbModel):
    def __init__(self, train_data, val_data):
        self.train_dataset = train_data
        self.val_dataset = val_data
        super().__init__(self.train_dataset, self.val_dataset)

    def train(self, model=None, save_path=None):
        return TrainXgbModel.train(self, model, save_path)

    def eval(self, model=None):
        TrainXgbModel.eval(self, model)
        x_eval, y_eval = self.get_X_Y(self.val_dataset)
        x_eval = x_eval.detach().cpu().numpy()
        y_eval = y_eval.detach().cpu().numpy().astype(np.int64)

        p = model.predict_proba(x_eval)[:, 1]

        for p_thres in np.linspace(0.2, 0.6, 9):
            y_pred = (p > float(p_thres)).astype(np.int64)
            print("\np_thres=", p_thres)
            print("pred positives:", y_pred.sum())
            self.print_confusion_matrix(y_eval, y_pred)

        if np.any(y_eval == 1):
            print("true heal prob:", np.percentile(p[y_eval == 1], [0, 1, 5, 25, 50, 75, 95, 99, 100]))
        if np.any(y_eval == 0):
            print("true no-heal prob:", np.percentile(p[y_eval == 0], [0, 1, 5, 25, 50, 75, 95, 99, 100]))

        eps = 1e-12
        loss = -(y_eval * np.log(p + eps) + (1 - y_eval) * np.log(1 - p + eps))
        idx = np.argsort(loss)[-20:]

        for i in idx:
            print(int(i), "y=", float(y_eval[i]), "p_heal=", float(p[i]), "loss=", float(loss[i]))

        false_negatives = (y_eval == 1) & (p < 0.2)
        true_positives = (y_eval == 1) & (p > 0.7)
        false_positives = (y_eval == 0) & (p > 0.6)

        print("false_negatives | col: myHp", x_eval[false_negatives, 0])
        print(np.unique(x_eval[false_negatives, 0], return_counts=True))
        # print("false_negatives | col: targetDistance", x_eval[false_negatives, 28])
        # print("true_positives | col: myHp", x_eval[true_positives, 0])
        # print("true_positives | col: targetDistance", x_eval[true_positives, 28])
        print(f"false_positives | col: myHp", x_eval[false_positives, 0])

    @staticmethod
    def print_confusion_matrix(y_eval, y_pred):
        TrainXgbModel.print_confusion_matrix(y_eval, y_pred, "heal")


if __name__ == "__main__":
    heal_trigger_dataset = HealTriggerDataset(use_cache=True, build_cache=False)
    train_data = heal_trigger_dataset.get_train_X_Y(transition_context=CONTEXT, prefer_hard_negative=True,
                                                    positive_to_negative=0.20, sample_mode="even")
    val_data = heal_trigger_dataset.get_val_X_Y(transition_context=CONTEXT,
                                                positive_to_negative=0.20, sample_mode="random")

    train_heal_xgb = TrainHealXgb(train_data, val_data)
    model = xgb.XGBClassifier(
        booster="gbtree",
        objective="binary:logistic",
        eval_metric="logloss",

        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        gamma=1.0,

        scale_pos_weight=3.0,

        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.0,

        early_stopping_rounds=30,
    )
    model = train_heal_xgb.train(model, save_path=MODEL_PATH)

    train_heal_xgb.eval(model)
