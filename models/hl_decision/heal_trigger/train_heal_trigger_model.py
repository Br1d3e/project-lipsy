from models.hl_decision.heal_trigger.heal_trigger_datasets import HealTriggerDataset
from models.hl_decision.train_hl_decision_model import TrainXgbModel
from models.window_dataset import WindowDataset

import xgboost as xgb

SEQ_LEN = 20
CONTEXT = 20

class TrainHealXgb(TrainXgbModel):
    def __init__(self, train_windows, val_windows):
        self.train_dataset = WindowDataset(train_windows)
        self.val_dataset = WindowDataset(val_windows)
        super().__init__(self.train_dataset, self.val_dataset)

    def train(self, model=None, save_path=None):
        return TrainXgbModel.train(self, model, save_path)

    def eval(self, model=None):
        TrainXgbModel.eval(self, model)


if __name__ == "__main__":
    heal_trigger_dataset = HealTriggerDataset()
    train_windows = heal_trigger_dataset.get_train_windows(seq_len=SEQ_LEN, transition_context=CONTEXT, prefer_hard_negative=True,
                                                           positive_to_negative=0.3333, sample_mode="even")
    val_windows = heal_trigger_dataset.get_val_windows(seq_len=SEQ_LEN, transition_context=CONTEXT,
                                                       positive_to_negative=0.3333, sample_mode="even")

    train_heal_xgb = TrainHealXgb(train_windows, val_windows)
    model = xgb.XGBClassifier(
        booster='gbtree',
        gamma=1.5,
        objective='binary:logistic',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        early_stopping_rounds=25,
    )
    model = train_heal_xgb.train(model)
    train_heal_xgb.eval(model)