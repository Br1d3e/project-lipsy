from models.hl_decision.hl_decision_datasets import (
    DEFAULT_ATTACK_SAMPLE_RATIO,
    DEFAULT_AFTER_TRANSITION_CONTEXT,
    DEFAULT_TRANSITION_CONTEXT,
    HLDecisionDataset,
)


FULL_HOTBAR_POTS = 6


class RefillTriggerDataset(HLDecisionDataset):
    label_name = "refill"

    def __init__(self, datasets=None, features=None, use_cache=False, build_cache = True):
        super().__init__(datasets=datasets, features=features, label_name=self.label_name, use_cache=use_cache, build_cache=build_cache)

    @staticmethod
    def hard_negative_mask(frames):
        return (
            (frames["hotbarHealthPotions"] < FULL_HOTBAR_POTS) &
            (frames["targetExists"] == True) &
            (frames["inInventory"] == False)
            # (
            #     (frames["myHp"] <= LOW_HP_THRESHOLD) |
            #     (frames["targetCombo"] >= HIGH_COMBO_THRESHOLD) |
            #     (frames["ticksSinceLastDamageTaken"] <= 10)
            # ) &
            # (frames["targetDistance"] <= 4.5)
        )

    @staticmethod
    def easy_negative_mask(frames):
        return (
            (frames["hotbarHealthPotions"] >= FULL_HOTBAR_POTS) |
            (frames["inInventory"] == True)
            # (frames["myHp"] > HIGH_HP_THRESHOLD) &
            # (frames["targetCombo"] <= 1) &
            # (frames["targetDistance"] <= 5.0)
        )

    def get_train_X_Y(self, seq_len=1, transition_context=DEFAULT_TRANSITION_CONTEXT,
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

    def get_val_X_Y(self, seq_len=1, transition_context=DEFAULT_TRANSITION_CONTEXT,
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

    def get_test_X_Y(self, seq_len=1, transition_context=DEFAULT_TRANSITION_CONTEXT, features=None):
        return super().get_test_X_Y(
            transition_context=transition_context,
            features=features,
            seq_len=seq_len,
        )


if __name__ == "__main__":
    dataset = RefillTriggerDataset(use_cache=True, build_cache=False)
    print("Train:")
    train_X, train_Y = dataset.get_train_X_Y(transition_context=20, prefer_hard_negative=False, sample_mode="even")
    dataset.target_distribution(train_Y)
    print("Val:")
    val_X, val_Y = dataset.get_val_X_Y(transition_context=20, prefer_hard_negative=False, sample_mode="random")
    dataset.target_distribution(val_Y)
    print("Test:")
    test_X, test_Y = dataset.get_test_X_Y(transition_context=20)
    dataset.target_distribution(test_Y)
