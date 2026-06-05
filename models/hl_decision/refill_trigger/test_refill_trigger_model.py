import os
import sys

from models.hl_decision.hl_decision_datasets import FEATURES

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import xgboost as xgb
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import shap
import pandas as pd

from models.hl_decision.train_hl_decision_model import TrainXgbModel
from models.hl_decision.refill_trigger.refill_trigger_datasets import RefillTriggerDataset
from models.hl_decision.refill_trigger.train_refill_trigger_model import CONTEXT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "refill_trigger_xgb_4.json")


def eval_xgb():
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Skipping: missing model file {model_path}")
        return

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    pearl_trigger_dataset = RefillTriggerDataset(use_cache=True, build_cache=False)
    x_test, y_test = pearl_trigger_dataset.get_test_X_Y(transition_context=CONTEXT)
    y_test = np.asarray(y_test, dtype=np.int64).reshape(-1)

    y_prob = model.predict_proba(x_test)
    loss = log_loss(y_test, y_prob)
    ppl = np.exp(loss)
    print(f"Test loss: {loss:.6f} | ppl: {ppl:.6f}")

    accuracy = model.score(x_test, y_test)
    print(f"Accuracy: {accuracy:.6f}")

    p_thres = 0.5
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


if __name__ == "__main__":
    eval_xgb()
