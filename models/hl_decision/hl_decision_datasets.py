import numpy as np
import pandas as pd
import os

from models.datasets import Datasets
from models.datasets import TRAIN_FOLDER
TRAIN_FOLDER = os.path.join(TRAIN_FOLDER, "session_1778405079341")

FEATURES = [
    # self
    "myHp",
    "myFoodLevel",
    "myHasSpeedPotion",
    "mySpeedDurationTicks",
    "myCombo",
    "targetCombo",

    # resources
    "pearlCooldownTicksRemaining",
    "enemyPearlAvailable",
    "enemyPearlCooldownTicksRemaining",
    "hotbarHealthPotions",
    "inventoryFood",
    "inventorySpeedPotions",
    "inventoryHealthPotions",
    "inventoryPearls",

    # timers
    "ticksSinceLastAttack",
    "ticksSinceLastBlock",
    "ticksSinceLastPot",
    "ticksSinceLastEat",
    "ticksSinceLastSpeed",
    "ticksSinceLastDamageTaken",
    "ticksSinceLastHitLanded",
    "ticksSinceLastPearl",
    "ticksSinceLastEnemyPearl",

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
    "targetEstimatedPotions",
    "targetHasSpeedPotion",
    "targetSpeedDurationTicks",
    "targetFood",
    "targetEating",
    "targetSpeeding",
    "targetPearling",
    "targetPotting",

    # relation
    "hpDiff",

    # derived
    "canHeal",
    "canPearl",
    "canSpeed",
    "canEat",
    
    # current state
    "s_attack",
    "s_heal",
    "s_pearl",
    "s_speed",
    "s_eat",
    "s_refill",
    "s_retreat"
]

states = ["s_attack", "s_heal", "s_pearl", "s_speed", "s_eat", "s_refill", "s_retreat"]
TARGETS = {"attack": 0, "retreat": 1, "heal": 2, "pearl": 3, "eat": 4, "speed": 5, "refill": 6}
FEATURE_DEFAULTS = {
    "ticksSinceLastAttack": 100000,
    "ticksSinceLastBlock": 100000,
    "ticksSinceLastPot": 100000,
    "ticksSinceLastEat": 100000,
    "ticksSinceLastSpeed": 100000,
    "ticksSinceLastDamageTaken": 100000,
    "ticksSinceLastHitLanded": 100000,
    "ticksSinceLastPearl": 100000,
    "ticksSinceLastEnemyPearl": 100000,
    "blockDistanceFront": 50,
    "blockDistanceBack": 50,
    "blockDistanceLeft": 50,
    "blockDistanceRight": 50,
    "targetEstimatedPotions": 29,
    "s_attack": 0,
    "s_retreat": 0,
    "s_heal": 0,
    "s_pearl": 0,
    "s_speed": 0,
    "s_eat": 0,
    "s_refill": 0,
}

class HLDecisionDataset:
    def __init__(self):
        self.datasets = Datasets(TRAIN_FOLDER, keep_frames=False)
        self.train_data = self.datasets.get_train_data()
        self.val_data = self.datasets.get_val_data()
        self.test_data = self.datasets.get_test_data()
        self.labels = self.datasets.get_labels()
        self.feature_list = FEATURES

    def make_windows(self, rounds: list[pd.DataFrame], features, transition_context=0, seq_len=1):
        data_windows = []
        columns = features

        for round_df in rounds:
            round_df = round_df.set_index("recordingTickIndex", drop=False)
            round_start = round_df["recordingTickIndex"].min()
            round_end = round_df["recordingTickIndex"].max()

            labels = self.datasets.labels_for_round(round_df)
            round_df = self._col_states(round_df, labels)
            for i in range(1, len(labels)):
                prev_label = labels[i - 1]
                curr_label = labels[i]
                curr_name = self.datasets.label_name(curr_label)
                if curr_name not in TARGETS:
                    continue
                curr_start = curr_label["start_tick"]
                prev_end = prev_label["end_tick"]
                overlap_start = max(round_start, prev_end - transition_context)
                overlap_end = min(round_end, curr_start)

                if overlap_start >= overlap_end:
                    continue

                available = round_df.loc[
                    (overlap_start <= round_df["recordingTickIndex"]) &
                    (round_df["recordingTickIndex"] < overlap_end) &
                    (round_df["targetExists"] == True),
                    columns
                ].sort_index()
                available = available.fillna(FEATURE_DEFAULTS)

                if len(available) <= seq_len:
                    continue

                values = available.to_numpy(dtype=np.float32, copy=True)
                values = values[np.isfinite(values).all(axis=1)]
                if len(values) <= seq_len:
                    continue

                target_value = np.asarray([TARGETS[curr_name]], dtype=np.float32)
                for i in range(0, len(values) - seq_len):
                    data_windows.append((
                        values[i:i + seq_len],
                        target_value,
                    ))

        return data_windows

    def _col_states(self, frames: pd.DataFrame, labels) -> pd.DataFrame:
        frames = frames.iloc[np.argsort(frames["recordingTickIndex"].to_numpy())].copy()
        frames[states] = 0

        frames_ticks = frames["recordingTickIndex"].to_numpy()
        frames_start = frames_ticks.min()
        frames_end = frames_ticks.max()

        for label in labels:
            label_name = self.datasets.label_name(label).strip().lower()
            label_start = label["start_tick"]
            label_end = label["end_tick"]

            overlap_start = max(frames_start, label_start)
            overlap_end = min(frames_end, label_end)
            if overlap_start >= overlap_end:
                continue

            if label_name in TARGETS.keys():
                row_idx = ((overlap_start <= frames["recordingTickIndex"]) &
                           (frames["recordingTickIndex"] <= overlap_end))
                unassigned_idx = row_idx & (frames[states].sum(axis=1) == 0)
                frames.loc[unassigned_idx, f"s_{label_name}"] = 1

        active_rows = frames[states].sum(axis=1) > 0
        active_state = pd.Series(np.nan, index=frames.index, dtype=object)
        active_state.loc[active_rows] = frames.loc[active_rows, states].idxmax(axis=1)
        previous_state = active_state.ffill().shift(1)

        frames[states] = 0.0
        for state in states:
            frames[state] = (previous_state == state).astype(np.float32)

        return frames

    def window_state_distribution(self, windows):
        # x_windows = [[frames_0, frames_1, frames_2...], [frames_1, frames_2, frames_3...]]
        if not windows:
            print("No windows")
            return np.zeros(len(states), dtype=np.float32)
        x_windows = [window[0] for window in windows]
        x_windows = np.array(x_windows)
        states_arr = x_windows[:, -1, -7:]
        states_count = np.sum(states_arr, axis=0)
        for i in range(7):
            print(f"{states[i]}: {states_count[i]}")
        return states_count

    def rounds_state_distribution(self, rounds: pd.DataFrame):
        states_arr = rounds[states].to_numpy()
        states_count = np.sum(states_arr, axis=0)
        print(states_count)

    def get_features_targets(self, rounds, seq_len=1, transition_context=0):
        return self.make_windows(
            rounds,
            self.feature_list,
            transition_context=transition_context,
            seq_len=seq_len,
        )

    def get_train_windows(self, seq_len=1, stride=1, transition_context=0):
        return self.get_features_targets(self.train_data, seq_len, transition_context)

    def get_val_windows(self, seq_len=1, stride=1, transition_context=0):
        return self.get_features_targets(self.val_data, seq_len, transition_context)

    def get_test_windows(self, seq_len=1, stride=1, transition_context=0):
        return self.get_features_targets(self.test_data, seq_len, transition_context)
