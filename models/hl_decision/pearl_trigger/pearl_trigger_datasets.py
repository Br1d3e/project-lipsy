from models.hl_decision.hl_decision_datasets import DEFAULT_TRANSITION_CONTEXT, HLDecisionDataset


HIGH_HP_THRESHOLD = 16
LOW_HP_THRESHOLD = 6


class PearlTriggerDataset(HLDecisionDataset):
    label_name = "pearl"

    def __init__(self, datasets=None, features=None, pearl_type="all"):
        self.pearl_type = pearl_type
        super().__init__(datasets=datasets, features=features, label_name=self.label_name)

    @staticmethod
    def hard_negative_mask(frames):
        return (
            (
                (frames["hpDiff"] <= -5) |
                (frames["myHp"] >= HIGH_HP_THRESHOLD) |
                (frames["targetEstimatedHp"] <= LOW_HP_THRESHOLD)
            ) &
            (frames["canPearl"] == True)
        )

    @staticmethod
    def easy_negative_mask(frames):
        return frames["canPearl"] == False

    @staticmethod
    def pearl_label_type(pearl_type):
        if pearl_type is None:
            pearl_type = "all"
        if pearl_type not in ("all", "offensive", "defensive"):
            raise ValueError(f"Invalid pearl type. Expected 'all', 'offensive', or 'defensive', got {pearl_type!r}")
        if pearl_type == "offensive":
            return ["tryhard", "side", "agro"]
        if pearl_type == "defensive":
            return "retreat"
        return ["tryhard", "side", "agro", "retreat"]

    def get_train_X_Y(self, seq_len=1, transition_context=DEFAULT_TRANSITION_CONTEXT, positive_to_negative=0.5,
                      prefer_hard_negative=False, sample_mode="even", pearl_type=None, features=None, msg=True):
        return super().get_train_X_Y(
            transition_context=transition_context,
            label_type=self.pearl_label_type(pearl_type or self.pearl_type),
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
                    prefer_hard_negative=False, sample_mode="even", pearl_type=None, features=None, msg=True):
        return super().get_val_X_Y(
            transition_context=transition_context,
            label_type=self.pearl_label_type(pearl_type or self.pearl_type),
            hard_negative_mask=self.hard_negative_mask,
            easy_negative_mask=self.easy_negative_mask,
            positive_to_negative=positive_to_negative,
            prefer_hard_negative=prefer_hard_negative,
            sample_mode=sample_mode,
            features=features,
            seq_len=seq_len,
            msg=msg,
        )

    def get_test_X_Y(self, seq_len=1, transition_context=DEFAULT_TRANSITION_CONTEXT, pearl_type=None, features=None):
        return super().get_test_X_Y(
            transition_context=transition_context,
            label_type=self.pearl_label_type(pearl_type or self.pearl_type),
            features=features,
            seq_len=seq_len,
        )


PearlTriggerDecisionDataset = PearlTriggerDataset


if __name__ == "__main__":
    dataset = PearlTriggerDataset()
    for pearl_type in ("offensive", "defensive"):
        print(f"[{pearl_type}] Train:")
        train_X, train_Y = dataset.get_train_X_Y(positive_to_negative=0.333,
                                                 prefer_hard_negative=True,
                                                 pearl_type=pearl_type)
        dataset.target_distribution(train_Y)
        print(f"[{pearl_type}] Val:")
        val_X, val_Y = dataset.get_val_X_Y(positive_to_negative=0.25,
                                           prefer_hard_negative=True,
                                           sample_mode="random",
                                           pearl_type=pearl_type)
        dataset.target_distribution(val_Y)
        print(f"[{pearl_type}] Test:")
        test_X, test_Y = dataset.get_test_X_Y(pearl_type=pearl_type)
        dataset.target_distribution(test_Y)
