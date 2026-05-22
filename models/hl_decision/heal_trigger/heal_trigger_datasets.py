import numpy as np

from models.hl_decision.hl_decision_datasets import HLDecisionDataset


FEATURES = [
    # self
    "myHp",
    "myCombo",
    "targetCombo",

    # resources
    "enemyPearlAvailable",
    "hotbarHealthPotions",
    "inventoryHealthPotions",

    # timers
    "ticksSinceLastAttack",
    "ticksSinceLastPot",
    "ticksSinceLastDamageTaken",

    # space
    "blockDistanceFront",
    "blockDistanceBack",
    "blockDistanceLeft",
    "blockDistanceRight",

    # target
    "targetExists",
    "targetDistance",
    "targetMotionX",
    "targetMotionZ",
    "targetEstimatedHp",
    "targetFood",
    "targetEating",
    "targetSpeeding",
    "targetPearling",
    "targetPotting",

    # relation
    "hpDiff",

    # derived
    "canHeal",

    # previous state
    "s_attack",
    "s_heal",
    "s_pearl",
    "s_speed",
    "s_eat",
    "s_refill",
    "s_retreat",
]

HIGH_HP_THRESHOLD = 12
LOW_HP_THRESHOLD = 10


class HealTriggerDataset(HLDecisionDataset):
    def __init__(self):
        super().__init__(features=FEATURES)

    @staticmethod
    def hard_negative_mask(frames):
        return (
            (frames["myHp"] <= LOW_HP_THRESHOLD) &
            (frames["canHeal"] == True)
        )

    @staticmethod
    def easy_negative_mask(frames):
        return (
            (frames["myHp"] > HIGH_HP_THRESHOLD) |
            (frames["canHeal"] == False)
        )

    def positive_windows(self, rounds, transition_context=0, seq_len=1):
        windows = self.make_windows(
            rounds,
            self.feature_list,
            label_name="heal",
            mode="positive",
            transition_context=transition_context,
            seq_len=seq_len,
        )
        print(f"positive_windows: {len(windows)}")
        return windows

    def hard_negative_windows(self, rounds, transition_context=0, seq_len=1):
        windows = self.make_windows(
            rounds,
            self.feature_list,
            label_name="heal",
            mode="negative",
            transition_context=transition_context,
            seq_len=seq_len,
            candidate_mask_fn=self.hard_negative_mask,
            mask_window="last",
        )
        print(f"hard_negative_windows: {len(windows)}")
        return windows

    def easy_negative_windows(self, rounds, transition_context=0, seq_len=1):
        windows = self.make_windows(
            rounds,
            self.feature_list,
            label_name="heal",
            mode="negative",
            transition_context=transition_context,
            seq_len=seq_len,
            candidate_mask_fn=self.easy_negative_mask,
            mask_window="last",
        )
        print(f"easy_negative_windows: {len(windows)}")
        return windows

    def split_train_val_windows(self, rounds, transition_context=0, seq_len=1):
        windows = []
        windows = self.positive_windows(rounds, transition_context, seq_len)
        windows.extend(self.easy_negative_windows(rounds, transition_context, seq_len))
        windows.extend(self.hard_negative_windows(rounds, transition_context, seq_len))
        return windows

    def realistic_windows(self, rounds, transition_context=0, seq_len=1):
        windows = self.make_windows(
            rounds,
            self.feature_list,
            label_name="heal",
            mode=None,
            transition_context=transition_context,
            seq_len=seq_len,
        )
        print(f"realistic_windows: {len(windows)}")
        return windows

    def target_distribution(self, windows):
        counts = {0: 0, 1: 0}
        for _, target in windows:
            counts[int(np.asarray(target).reshape(-1)[0])] += 1
        print(f"target_distribution: {counts}")
        return counts

    def get_train_windows(self, seq_len=1, stride=1, transition_context=0):
        return self.split_train_val_windows(self.train_data, transition_context, seq_len)

    def get_val_windows(self, seq_len=1, stride=1, transition_context=0):
        return self.split_train_val_windows(self.val_data, transition_context, seq_len)

    def get_test_windows(self, seq_len=1, stride=1, transition_context=0):
        return self.realistic_windows(self.test_data, transition_context, seq_len)


if __name__ == "__main__":
    heal_trigger_dataset = HealTriggerDataset()
    print("Train:")
    train_windows = heal_trigger_dataset.get_train_windows(seq_len=1, transition_context=1)
    heal_trigger_dataset.target_distribution(train_windows)
    print("Val:")
    val_windows = heal_trigger_dataset.get_val_windows(seq_len=1, stride=1, transition_context=1)
    heal_trigger_dataset.target_distribution(val_windows)
    print("Test:")
    test_windows = heal_trigger_dataset.get_test_windows(seq_len=1, stride=1, transition_context=1)
    heal_trigger_dataset.target_distribution(test_windows)
