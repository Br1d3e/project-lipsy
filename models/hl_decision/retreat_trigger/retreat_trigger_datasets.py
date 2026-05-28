from models.hl_decision.hl_decision_datasets import DEFAULT_TRANSITION_CONTEXT, HLDecisionDataset


HIGH_HP_THRESHOLD = 14
LOW_HP_THRESHOLD = 8
HIGH_COMBO_THRESHOLD = 4


class RetreatTriggerDataset(HLDecisionDataset):
    label_name = "retreat"

    def __init__(self, datasets=None, features=None):
        super().__init__(datasets=datasets, features=features, label_name=self.label_name)

    @staticmethod
    def hard_negative_mask(frames):
        return (
            (
                (frames["myHp"] <= LOW_HP_THRESHOLD) |
                (frames["targetCombo"] >= HIGH_COMBO_THRESHOLD) |
                (frames["ticksSinceLastDamageTaken"] <= 10)
            ) &
            (frames["targetDistance"] <= 4.5)
        )

    @staticmethod
    def easy_negative_mask(frames):
        return (
            (frames["myHp"] > HIGH_HP_THRESHOLD) &
            (frames["targetCombo"] <= 1) &
            (frames["targetDistance"] <= 5.0)
        )

    def get_train_X_Y(self, seq_len=1, transition_context=DEFAULT_TRANSITION_CONTEXT, positive_to_negative=0.5,
                      prefer_hard_negative=False, sample_mode="even", features=None, msg=True):
        return super().get_train_X_Y(
            transition_context=transition_context,
            hard_negative_mask=self.hard_negative_mask,
            easy_negative_mask=self.easy_negative_mask,
            positive_to_negative=positive_to_negative,
            prefer_hard_negative=prefer_hard_negative,
            sample_mode=sample_mode,
            features=features,
            seq_len=seq_len,
            msg=msg,
        )

    def get_val_X_Y(self, seq_len=1, transition_context=DEFAULT_TRANSITION_CONTEXT, positive_to_negative=0.5,
                    prefer_hard_negative=False, sample_mode="even", features=None, msg=True):
        return super().get_val_X_Y(
            transition_context=transition_context,
            hard_negative_mask=self.hard_negative_mask,
            easy_negative_mask=self.easy_negative_mask,
            positive_to_negative=positive_to_negative,
            prefer_hard_negative=prefer_hard_negative,
            sample_mode=sample_mode,
            features=features,
            seq_len=seq_len,
            msg=msg,
        )

    def get_test_X_Y(self, seq_len=1, transition_context=DEFAULT_TRANSITION_CONTEXT, features=None):
        return super().get_test_X_Y(
            transition_context=transition_context,
            features=features,
            seq_len=seq_len,
        )


if __name__ == "__main__":
    dataset = RetreatTriggerDataset()
    print("Train:")
    train_X, train_Y = dataset.get_train_X_Y(transition_context=1, prefer_hard_negative=True)
    dataset.target_distribution(train_Y)
    print("Val:")
    val_X, val_Y = dataset.get_val_X_Y(transition_context=1, prefer_hard_negative=True)
    dataset.target_distribution(val_Y)
    print("Test:")
    test_X, test_Y = dataset.get_test_X_Y(transition_context=1)
    dataset.target_distribution(test_Y)
