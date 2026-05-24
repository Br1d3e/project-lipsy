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

    def split_train_val_windows(self, rounds, transition_context=0, seq_len=1, positive_to_negative=0.5, prefer_hard_negative=False):
        windows = []
        positive = self.positive_windows(rounds, transition_context, seq_len)
        easy_negative = self.easy_negative_windows(rounds, transition_context, seq_len)
        hard_negative = self.hard_negative_windows(rounds, transition_context, seq_len)
        negative = easy_negative + hard_negative

        windows_len = len(positive) + len(negative)

        positive_ratio = positive_to_negative / (1 + positive_to_negative)
        positive_len = min(round(windows_len * positive_ratio), len(positive))
        negative_len = min(round(positive_len / positive_to_negative), len(negative))

        positive = self._sample_windows(positive, positive_len)
        if prefer_hard_negative:
            negative = hard_negative + self._sample_windows(negative, negative_len - len(hard_negative))
        negative = self._sample_windows(negative, negative_len)

        windows.extend(positive)
        windows.extend(negative)

        return windows

    def _sample_windows(self, windows: list, required: int, mode="even") -> list:
        if len(windows) <= required:
            return windows

        if mode == "even":
            sample_idx = np.linspace(0, len(windows) - 1, num=required).astype(int)
        elif mode == "random":
            rng = np.random.default_rng(seed=42)
            sample_idx = rng.choice(len(windows), size=required, replace=False)
        else:
            raise ValueError("Invalid mode.")

        return [windows[i] for i in sample_idx]

    # def _sample_windows(self, windows: list, required: int) -> list:
    #     if len(windows) == 1:
    #         return windows
    #     if len(windows) <= required:
    #         return windows
    #
    #     mid = (len(windows) + 1) // 2
    #     if required == 1:
    #         return [windows[mid]]
    #
    #     left = self._sample_windows(windows[:mid], required // 2)
    #     right = self._sample_windows(windows[mid:], required - required // 2)
    #
    #     if isinstance(left, list) and isinstance(right, list):
    #         return list(self._flatten_once(left + right))
    #     if isinstance(left, tuple) and isinstance(right, list):
    #         return list(self._flatten_once(right)) + list(left)
    #     if isinstance(left, list) and isinstance(right, tuple):
    #         return list(self._flatten_once(left)) + list(right)
    #     if isinstance(left, tuple) and isinstance(right, tuple):
    #         return list(left) + list(right)
    #
    #     raise ValueError("Invalid windows structure.")

    def _flatten_once(self, nested):
        for item in nested:
            yield item

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

    def get_train_windows(self, seq_len=1, stride=1, transition_context=0, positive_to_negative=0.5, prefer_hard_negative=False):
        return self.split_train_val_windows(self.train_data, transition_context, seq_len, positive_to_negative, prefer_hard_negative)

    def get_val_windows(self, seq_len=1, stride=1, transition_context=0, positive_to_negative=0.5, prefer_hard_negative=False):
        return self.split_train_val_windows(self.val_data, transition_context, seq_len, positive_to_negative, prefer_hard_negative)

    def get_test_windows(self, seq_len=1, stride=1, transition_context=0):
        return self.realistic_windows(self.test_data, transition_context, seq_len)


if __name__ == "__main__":
    heal_trigger_dataset = HealTriggerDataset()
    print("Train:")
    train_windows = heal_trigger_dataset.get_train_windows(seq_len=1, transition_context=1, prefer_hard_negative=True)
    heal_trigger_dataset.target_distribution(train_windows)
    print("Val:")
    val_windows = heal_trigger_dataset.get_val_windows(seq_len=1, stride=1, transition_context=1)
    heal_trigger_dataset.target_distribution(val_windows)
    print("Test:")
    test_windows = heal_trigger_dataset.get_test_windows(seq_len=1, stride=1, transition_context=1)
    heal_trigger_dataset.target_distribution(test_windows)
