import os

import xgboost as xgb
import numpy as np
from sklearn.metrics import log_loss

from models.hl_decision.train_hl_decision_model import TrainXgbModel
from models.hl_decision.heal_trigger.heal_trigger_datasets import HealTriggerDataset
from models.hl_decision.heal_trigger.train_heal_trigger_model import CONTEXT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heal_trigger_xgb_4.json")


def eval_xgb():
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Skipping: missing model file {model_path}")
        return

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    heal_trigger_dataset = HealTriggerDataset(use_cache=True, build_cache=False)
    x_test, y_test = heal_trigger_dataset.get_test_X_Y(transition_context=CONTEXT)
    y_test = np.asarray(y_test, dtype=np.int64).reshape(-1)

    y_prob = model.predict_proba(x_test)
    loss = log_loss(y_test, y_prob)
    ppl = np.exp(loss)
    print(f"Test loss: {loss:.6f} | ppl: {ppl:.6f}")

    accuracy = model.score(x_test, y_test)
    print(f"Accuracy: {accuracy:.6f}")

    p_thres = 0.65
    y_pred = (y_prob[:, 1] > float(p_thres)).astype(np.int64)

    print(f"p_thres: {p_thres:.6f}")
    TrainXgbModel.print_confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    eval_xgb()
