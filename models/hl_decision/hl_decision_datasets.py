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

STATES = ["s_attack", "s_heal", "s_pearl", "s_speed", "s_eat", "s_refill", "s_retreat"]
TARGETS = {"attack": 0, "heal": 1, "pearl": 2, "speed": 3, "eat": 4, "refill": 5, "retreat": 6}
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
    def __init__(self, features: list[str] = None):
        self.datasets = Datasets(TRAIN_FOLDER, keep_frames=False)
        self.train_data = self.datasets.get_train_data()
        self.val_data = self.datasets.get_val_data()
        self.test_data = self.datasets.get_test_data()
        self.labels = self.datasets.get_labels()
        self.feature_list = FEATURES
        if isinstance(features, list):
            self.feature_list = features

    def make_windows(
            self,
            rounds: list[pd.DataFrame],
            features=None,
            label_name=None,
            label_type: list | tuple | str | None=None,
            mode=None,
            transition_context=0,
            seq_len=1,
            candidate_mask_fn=None,
            mask_window="last",
    ):
        if mode not in (None, "positive", "negative"):
            raise ValueError(f"mode must be None, 'positive', or 'negative', got {mode!r}")
        if mode in ("positive", "negative") and not label_name:
            raise ValueError("label_name is required for positive/negative trigger modes")
        if mask_window not in ("last", "all"):
            raise ValueError(f"mask_window must be 'last' or 'all', got {mask_window!r}")

        data_windows = []
        columns = list(features if features is not None else self.feature_list)

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
                curr_type = curr_label["type"]
                target_value = None
                if mode == "positive":
                    if curr_name != label_name:
                        continue
                    target_value = np.asarray([1], dtype=np.float32)
                elif mode == "negative":
                    if curr_name == label_name:
                        continue
                    target_value = np.asarray([0], dtype=np.float32)
                else:
                    if curr_name not in TARGETS.keys():
                        continue
                    if label_name:
                        target_value = np.asarray([1 if curr_name == label_name else 0], dtype=np.float32)
                    else:
                        target_value = np.asarray([TARGETS[curr_name]], dtype=np.float32)

                if isinstance(label_type, str) and curr_type != label_type:
                    continue
                if (isinstance(label_type, list) or isinstance(label_type, tuple)) and curr_type not in label_type:
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

                if len(available) < seq_len:
                    continue

                values = available.to_numpy(dtype=np.float32, copy=True)
                valid_rows = np.isfinite(values).all(axis=1)
                candidate_mask = np.ones(len(available), dtype=bool)
                if candidate_mask_fn is not None:
                    candidate_mask = np.asarray(candidate_mask_fn(available), dtype=bool)
                    if len(candidate_mask) != len(available):
                        raise ValueError(
                            f"candidate_mask_fn returned {len(candidate_mask)} rows for {len(available)} available rows"
                        )

                if len(values) < seq_len:
                    continue

                for i in range(0, len(values) - seq_len + 1):
                    if not valid_rows[i:i + seq_len].all():
                        continue
                    if mask_window == "all":
                        if not candidate_mask[i:i + seq_len].all():
                            continue
                    elif not candidate_mask[i + seq_len - 1]:
                        continue

                    data_windows.append((
                        values[i:i + seq_len],
                        target_value
                    ))

        return data_windows

    def _col_states(self, frames: pd.DataFrame, labels) -> pd.DataFrame:
        frames = frames.iloc[np.argsort(frames["recordingTickIndex"].to_numpy())].copy()
        frames[STATES] = 0

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

            if label_name in ["attack", "heal", "pearl", "speed", "eat", "refill", "retreat"]:
                row_idx = ((overlap_start <= frames["recordingTickIndex"]) &
                           (frames["recordingTickIndex"] <= overlap_end))
                unassigned_idx = row_idx & (frames[STATES].sum(axis=1) == 0)
                frames.loc[unassigned_idx, f"s_{label_name}"] = 1

        active_rows = frames[STATES].sum(axis=1) > 0
        active_state = pd.Series(np.nan, index=frames.index, dtype=object)
        active_state.loc[active_rows] = frames.loc[active_rows, STATES].idxmax(axis=1)
        previous_state = active_state.ffill().shift(1)

        frames[STATES] = 0.0
        for state in STATES:
            frames[state] = (previous_state == state).astype(np.float32)

        return frames

    def window_state_distribution(self, windows):
        # x_windows = [[frames_0, frames_1, frames_2...], [frames_1, frames_2, frames_3...]]
        if not windows:
            print("No windows")
            return np.zeros(len(STATES), dtype=np.float32)
        x_windows = [window[0] for window in windows]
        x_windows = np.array(x_windows)
        states_arr = x_windows[:, -1, -7:]
        states_count = np.sum(states_arr, axis=0)
        for i in range(7):
            print(f"{STATES[i]}: {states_count[i]}")
        return states_count

    def rounds_state_distribution(self, rounds: pd.DataFrame):
        states_arr = rounds[STATES].to_numpy()
        states_count = np.sum(states_arr, axis=0)
        print(states_count)

    def index_of_features(self, feature_name: str):
        if feature_name in self.feature_list:
            return self.feature_list.index(feature_name)

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
