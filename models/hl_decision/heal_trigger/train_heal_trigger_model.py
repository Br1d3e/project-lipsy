from models.hl_decision.heal_trigger.heal_trigger_datasets import HealTriggerDataset
from models.hl_decision.train_hl_decision_model import TrainXgbModel

import xgboost as xgb

SEQ_LEN = 1
CONTEXT = 20

class TrainHealXgb(TrainXgbModel):
    def __init__(self, train_data, val_data):
        self.train_dataset = train_data
        self.val_dataset = val_data
        super().__init__(self.train_dataset, self.val_dataset)

    def train(self, model=None, save_path=None):
        return TrainXgbModel.train(self, model, save_path)

    def eval(self, model=None):
        TrainXgbModel.eval(self, model)


if __name__ == "__main__":
    heal_trigger_dataset = HealTriggerDataset()
    train_data = heal_trigger_dataset.get_train_X_Y(seq_len=SEQ_LEN, transition_context=CONTEXT, prefer_hard_negative=True,
                                                    positive_to_negative=0.3333, sample_mode="even")
    val_data = heal_trigger_dataset.get_val_X_Y(seq_len=SEQ_LEN, transition_context=CONTEXT,
                                                positive_to_negative=0.3333, sample_mode="even")

    train_heal_xgb = TrainHealXgb(train_data, val_data)
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
