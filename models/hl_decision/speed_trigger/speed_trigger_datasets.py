from models.hl_decision.hl_decision_datasets import (
    DEFAULT_ATTACK_SAMPLE_RATIO,
    DEFAULT_AFTER_TRANSITION_CONTEXT,
    DEFAULT_TRANSITION_CONTEXT,
    HLDecisionDataset,
)


HIGH_DURATION_THRESHOLD = 20 * 50
LOW_DURATION_THRESHOLD = 20 * 15


class SpeedTriggerDecisionDataset(HLDecisionDataset):
    label_name = "speed"

    def __init__(self, datasets=None, features=None):
        super().__init__(datasets=datasets, features=features, label_name=self.label_name)

    @staticmethod
    def hard_negative_mask(frames):
        return (
            (frames["mySpeedDurationTicks"] <= LOW_DURATION_THRESHOLD) &
            (frames["canSpeed"] == True)
        )

    @staticmethod
    def easy_negative_mask(frames):
        return (
            (frames["mySpeedDurationTicks"] > HIGH_DURATION_THRESHOLD) |
            (frames["canSpeed"] == False)
        )

    def get_train_X_Y(self, seq_len=1, stride=1, transition_context=DEFAULT_TRANSITION_CONTEXT,
                      after_transition_context=DEFAULT_AFTER_TRANSITION_CONTEXT,
                      attack_sample_ratio=DEFAULT_ATTACK_SAMPLE_RATIO, positive_to_negative=0.5,
                      prefer_hard_negative=False, sample_mode="even", features=None, msg=True):
        return super().get_train_X_Y(
            transition_context=transition_context,
            after_transition_context=after_transition_context,
            attack_sample_ratio=attack_sample_ratio,
            hard_negative_mask=self.hard_negative_mask,
            easy_negative_mask=self.easy_negative_mask,
            positive_to_negative=positive_to_negative,
            prefer_hard_negative=prefer_hard_negative,
            sample_mode=sample_mode,
            features=features,
            seq_len=seq_len,
            msg=msg,
        )

    def get_val_X_Y(self, seq_len=1, stride=1, transition_context=DEFAULT_TRANSITION_CONTEXT,
                    after_transition_context=DEFAULT_AFTER_TRANSITION_CONTEXT,
                    attack_sample_ratio=DEFAULT_ATTACK_SAMPLE_RATIO, positive_to_negative=0.5,
                    prefer_hard_negative=False, sample_mode="even", features=None, msg=True):
        return super().get_val_X_Y(
            transition_context=transition_context,
            after_transition_context=after_transition_context,
            attack_sample_ratio=attack_sample_ratio,
            hard_negative_mask=self.hard_negative_mask,
            easy_negative_mask=self.easy_negative_mask,
            positive_to_negative=positive_to_negative,
            prefer_hard_negative=prefer_hard_negative,
            sample_mode=sample_mode,
            features=features,
            seq_len=seq_len,
            msg=msg,
        )

    def get_test_X_Y(self, seq_len=1, stride=1, transition_context=DEFAULT_TRANSITION_CONTEXT, features=None):
        return super().get_test_X_Y(
            transition_context=transition_context,
            features=features,
            seq_len=seq_len,
        )


if __name__ == "__main__":
    dataset = SpeedTriggerDecisionDataset()
    print("Train:")
    train_X, train_Y = dataset.get_train_X_Y(transition_context=1, prefer_hard_negative=True,
                                             positive_to_negative=0.25)
    dataset.target_distribution(train_Y)
    print("Val:")
    val_X, val_Y = dataset.get_val_X_Y(transition_context=1, positive_to_negative=0.25,
                                       sample_mode="random")
    dataset.target_distribution(val_Y)
    print("Test:")
    test_X, test_Y = dataset.get_test_X_Y(transition_context=1)
    dataset.target_distribution(test_Y)
