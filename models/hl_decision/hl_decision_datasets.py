import numpy as np
import pandas as pd
import os
import pickle

from models.datasets import Datasets
from models.datasets import TRAIN_FOLDER

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
    'targetRelX',
    'targetRelZ',
    "targetMotionX",
    "targetMotionZ",
    'targetYaw',
    'targetPitch',
    'targetDeltaYaw',
    'targetDeltaPitch',
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
]

STATES = ["s_attack", "s_heal", "s_pearl", "s_speed", "s_eat", "s_refill"]
DEFAULT_TRANSITION_CONTEXT = 20
TARGETS = {"attack": 0, "heal": 1, "pearl": 2, "speed": 3, "eat": 4, "refill": 5}
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
    "s_heal": 0,
    "s_pearl": 0,
    "s_speed": 0,
    "s_eat": 0,
    "s_refill": 0,
}

class HLDecisionDataset:
    label_name = None
    label_type = None

    def __init__(self, datasets=None, features: list[str] = None, label_name=None, label_type=None, use_cache=False, build_cache=True):
        self.feature_list = FEATURES
        if isinstance(features, list):
            self.feature_list = features
        if label_name is not None:
            self.label_name = label_name
        if label_type is not None:
            self.label_type = label_type

        self.datasets = datasets if datasets is not None else Datasets(TRAIN_FOLDER, keep_frames=False)

        self.cache_dir = os.path.join(TRAIN_FOLDER, ".cache")
        if use_cache:
            self.train_frames = self.load_pickle_cache("train")
            self.val_frames = self.load_pickle_cache("val")
            self.test_frames = self.load_pickle_cache("test")
        else:
            self.train_rounds = self.datasets.get_train_data()
            self.val_rounds = self.datasets.get_val_data()
            self.test_rounds = self.datasets.get_test_data()

            self.train_frames = self.concatenate_frames(self.train_rounds)
            self.val_frames = self.concatenate_frames(self.val_rounds)
            self.test_frames = self.concatenate_frames(self.test_rounds)

        if build_cache:
            self.save_pickle_cache(self.train_frames, "train")
            self.save_pickle_cache(self.val_frames, "val")
            self.save_pickle_cache(self.test_frames, "test")

    def cache_name(self, name):
        return f"hl_v1_{name}.pickle"

    def load_pickle_cache(self, name):
        path = os.path.join(self.cache_dir, self.cache_name(name))
        if os.path.exists(path):
            return pd.read_pickle(path)
        raise FileNotFoundError(f"Cache {self.cache_name(name)} not found.")

    def save_pickle_cache(self, frames, name):
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, self.cache_name(name))
        frames.reset_index(drop=True).to_pickle(path)

    def _col_labels_states(self, frames: pd.DataFrame, labels) -> pd.DataFrame:
        frames = frames.iloc[np.argsort(frames["recordingTickIndex"].to_numpy())].copy()
        frames[STATES] = 0
        frames["label"] = None
        frames["label_type"] = None
        frames["label_start"] = -1

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
                row_idx = np.zeros(len(frames), dtype=bool)
                start_idx = np.searchsorted(frames_ticks, overlap_start, side="left")
                end_idx = np.searchsorted(frames_ticks, overlap_end, side="right")
                row_idx[start_idx:end_idx] = True

                unassigned_idx = row_idx & (frames[STATES].sum(axis=1) == 0)
                frames.loc[unassigned_idx, f"s_{label_name}"] = 1

                frames.loc[unassigned_idx, "label"] = label_name
                frames.loc[unassigned_idx, "label_type"] = label["type"]
                frames.loc[unassigned_idx, "label_start"] = label["start_tick"]

        active_rows = frames[STATES].sum(axis=1) > 0
        active_state = pd.Series(np.nan, index=frames.index, dtype=object)
        active_state.loc[active_rows] = frames.loc[active_rows, STATES].idxmax(axis=1)
        previous_state = active_state.ffill().shift(1)

        frames[STATES] = 0.0
        for state in STATES:
            frames[state] = (previous_state == state).astype(np.float32)

        return frames

    def concatenate_frames(self, rounds):
        parts = []
        for i, round_df in enumerate(rounds):
            round_df = round_df.set_index("recordingTickIndex", drop=False)
            round_df["round_id"] = i
            round_df = self._col_labels_states(round_df, self.datasets.labels_for_round(round_df))
            round_df = round_df.set_index(["round_id", "recordingTickIndex"], drop=False)

            parts.append(round_df)

        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, axis=0, copy=False)

    def get_train_frames(self):
        return self.train_frames

    def get_val_frames(self):
        return self.val_frames

    def get_test_frames(self):
        return self.test_frames

    def get_frames(self):
        return self.train_frames, self.val_frames, self.test_frames

    def select_X_Y(
            self,
            frames: pd.DataFrame,
            features=None,
            feature_mask_fn=None,
            target=None,
            target_mask_fn=None,
        default_features=None
    ):
        if features is None:
            features = self.feature_list
        if default_features is None:
            default_features = FEATURE_DEFAULTS
        features_df = frames[features].fillna(default_features)
        feature_matrix = features_df.to_numpy(np.float32)

        if feature_mask_fn is not None:
            feature_mask = np.asarray(feature_mask_fn(frames), dtype=bool)
        else:
            feature_mask = np.ones(len(frames), dtype=bool)
        if feature_mask.shape[0] != len(frames):
            raise ValueError(f"Invalid feature mask shape. Expected {len(frames)}, got {feature_mask.shape[0]}")

        target_value = np.zeros(len(frames), dtype=np.float32)
        if isinstance(target, int) or isinstance(target, float):
            target_value.fill(target)
        elif isinstance(target, np.ndarray):
            if target.shape[0] != len(frames):
                raise ValueError(f"Invalid target shape. Expected {len(frames)}, got {target.shape[0]}")
            target_value = target.astype(np.float32, copy=False)
        elif target_mask_fn is not None:
            target_value = np.asarray(target_mask_fn(frames), dtype=np.float32)
            if target_value.shape[0] != len(frames):
                raise ValueError(f"Invalid target mask shape. Expected {len(frames)}, got {target_value.shape[0]}")
        else:
            raise ValueError("No target specified")

        return feature_matrix[feature_mask], target_value[feature_mask]

    def positive_prev_mask(self, frames, transition_context=DEFAULT_TRANSITION_CONTEXT, label_name=None, label_type=None):
        positive_prev_mask = np.zeros(len(frames), dtype=bool)  # global mask (frames,)

        if transition_context < 1:
            raise ValueError("transition_context must be >= 1")

        for round_id, round_df in frames.groupby(frames["round_id"], sort=False):
            round_idx = frames.index.get_indexer(round_df.index)
            ticks = round_df["recordingTickIndex"].to_numpy()

            label_mask = np.ones(len(round_df), dtype=bool)
            if label_name is not None:
                label_mask &= (round_df["label"] == label_name).to_numpy()
            if label_type is not None:
                if isinstance(label_type, (list, tuple, set)):
                    label_mask &= round_df["label_type"].isin(label_type).to_numpy()
                else:
                    label_mask &= (round_df["label_type"] == label_type).to_numpy()

            label_start = np.unique(round_df.loc[label_mask, "label_start"].to_numpy())
            round_mask = np.zeros(len(round_df), dtype=bool)
            for start in label_start:
                round_mask |= (ticks >= start - transition_context) & (ticks < start)

            positive_prev_mask[round_idx] = round_mask

        return positive_prev_mask

    @staticmethod
    def concatenate_X_Y(
            pos_X,
            pos_Y,
            neg_X,
            neg_Y,
    ):
        if pos_X.shape[0] != pos_Y.shape[0]:
            raise ValueError(f"Dimension 0 must be same for pos_X and pos_Y, but got {pos_X.shape!r} and {pos_Y.shape!r} instead.")
        if neg_X.shape[0] != neg_Y.shape[0]:
            raise ValueError(f"Dimension 0 must be same for neg_X and neg_Y, but got {neg_X.shape!r} and {neg_Y.shape!r} instead.")

        X = np.concatenate([pos_X, neg_X], axis=0)
        Y = np.concatenate([pos_Y, neg_Y], axis=0)

        return X, Y

    @staticmethod
    def sample_X_Y(X: np.ndarray, Y: np.ndarray, required: int, mode="even"):
        if mode not in ["even", "random", None]:
            raise ValueError(f"Expected mode: 'even' or 'random', got {mode!r}.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have the same dimension 0, got X:{X.shape!r} and Y:{Y.shape!r} instead.")

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)

        if mode is None or X.shape[0] <= required:
            return X, Y
        if required <= 0:
            return X[:0], Y[:0]

        if mode == "even":
            sample_idx = np.linspace(0, X.shape[0] - 1, required).astype(int)
        elif mode == "random":
            rng = np.random.default_rng(seed=42)
            sample_idx = rng.choice(X.shape[0], required, replace=False)

        return X[sample_idx], Y[sample_idx]

    def smart_sample(
            self,
            pos_X,
            pos_Y,
            hard_X,
            hard_Y,
            easy_X,
            easy_Y,
            positive_to_negative = 0.5,
            prefer_hard_negative = False,
            mode = "even",
            msg=False,
    ):
        pos_len = pos_X.shape[0]
        hard_negative_len = hard_X.shape[0]
        easy_negative_len = easy_X.shape[0]
        negative_X = np.concatenate((hard_X, easy_X), axis=0)
        negative_Y = np.concatenate((hard_Y, easy_Y), axis=0)
        negative_len = hard_negative_len + easy_negative_len
        total_len = pos_len + hard_negative_len + easy_negative_len

        if msg:
            print("Raw:")
            print(
                f"positive: {pos_len} | negative: {negative_len} | hard_negative: {hard_negative_len} | easy_negative: {easy_negative_len}")
            print(f"total: {total_len}")

        if pos_len == 0:
            if msg:
                print("No positive samples; returning available negatives only.")
            return negative_X, negative_Y
        if negative_len == 0:
            if msg:
                print("No negative samples; returning available positives only.")
            return pos_X, pos_Y

        positive_ratio = positive_to_negative / (1 + positive_to_negative)
        desired_pos_len = min(round(total_len * positive_ratio), pos_len)
        desired_negative_len = min(round(desired_pos_len / positive_to_negative), negative_len)

        pos_X, pos_Y = self.sample_X_Y(pos_X, pos_Y, desired_pos_len, mode=mode)
        if prefer_hard_negative:
            easy_required = max(0, desired_negative_len - hard_negative_len)
            easy_X, easy_Y = self.sample_X_Y(easy_X, easy_Y, easy_required, mode=mode)
            negative_X = np.concatenate((hard_X, easy_X), axis=0)
            negative_Y = np.concatenate((hard_Y, easy_Y), axis=0)
        else:
            negative_X, negative_Y = self.sample_X_Y(negative_X, negative_Y, desired_negative_len, mode=mode)

        if msg:
            print("Sampled:")
            print(f"pos_X: {pos_X.shape} | pos_Y: {pos_Y.shape}")
            print(f"negative_X: {negative_X.shape} | negative_Y: {negative_Y.shape}")
            print(f"total: {pos_X.shape[0] + negative_X.shape[0]}")

        return self.concatenate_X_Y(pos_X, pos_Y, negative_X, negative_Y)


    def get_sampled_X_Y(
            self,
            frames,
            transition_context=DEFAULT_TRANSITION_CONTEXT,
            label_name=None,
            label_type=None,
            hard_negative_mask=True,
            easy_negative_mask=True,
            positive_to_negative=0.5,
            prefer_hard_negative=False,
            sample_mode="even",
            features=None,
            msg=True,
    ):
        if label_name is None:
            label_name = self.label_name
        if label_type is None:
            label_type = self.label_type
        if label_name is None:
            raise ValueError("label_name is required")

        positive_mask = self.positive_prev_mask(frames, transition_context, label_name, label_type)
        valid_mask = np.asarray(self.is_valid_tick(frames), dtype=bool)
        if callable(hard_negative_mask):
            hard_negative_mask = np.asarray(hard_negative_mask(frames), dtype=bool)
        else:
            hard_negative_mask = np.asarray(hard_negative_mask, dtype=bool)
        if callable(easy_negative_mask):
            easy_negative_mask = np.asarray(easy_negative_mask(frames), dtype=bool)
        else:
            easy_negative_mask = np.asarray(easy_negative_mask, dtype=bool)

        pos_X, pos_Y = self.select_X_Y(
            frames=frames,
            features=features,
            feature_mask_fn=lambda f:
                valid_mask &
                positive_mask,
            target=1
        )
        hard_X, hard_Y = self.select_X_Y(
            frames,
            features=features,
            feature_mask_fn = lambda f:
                valid_mask &
                hard_negative_mask &
                ~positive_mask,
            target=0
        )
        easy_X, easy_Y = self.select_X_Y(
            frames,
            features=features,
            feature_mask_fn=lambda f:
                valid_mask &
                easy_negative_mask &
                ~positive_mask,
            target=0
        )
        sampled_X, sampled_Y = self.smart_sample(
            pos_X,
            pos_Y,
            hard_X,
            hard_Y,
            easy_X,
            easy_Y,
            positive_to_negative=positive_to_negative,
            prefer_hard_negative=prefer_hard_negative,
            mode=sample_mode,
            msg=msg
        )
        return sampled_X, sampled_Y

    def get_realistic_X_Y(
            self,
            frames,
            transition_context=DEFAULT_TRANSITION_CONTEXT,
            label_name=None,
            label_type=None,
            features=None,
    ):
        if label_name is None:
            label_name = self.label_name
        if label_type is None:
            label_type = self.label_type
        positive_mask = self.positive_prev_mask(frames, transition_context, label_name, label_type)
        targets = np.asarray(positive_mask, dtype=np.float32)
        X, Y = self.select_X_Y(
            frames=frames,
            features=features,
            feature_mask_fn=lambda f: self.is_valid_tick(frames),
            target=targets,
        )
        return X, Y

    def get_train_X_Y(
            self,
            transition_context=DEFAULT_TRANSITION_CONTEXT,
            label_name=None,
            label_type=None,
            hard_negative_mask=True,
            easy_negative_mask=True,
            positive_to_negative=0.5,
            prefer_hard_negative=False,
            sample_mode="even",
            features=None,
            seq_len=1,
            msg=True,
    ):
        return self.get_sampled_X_Y(
            self.train_frames,
            transition_context=transition_context,
            label_name=label_name,
            label_type=label_type,
            hard_negative_mask=hard_negative_mask,
            easy_negative_mask=easy_negative_mask,
            positive_to_negative=positive_to_negative,
            prefer_hard_negative=prefer_hard_negative,
            sample_mode=sample_mode,
            features=features,
            msg=msg,
        )


    def get_val_X_Y(
            self,
            transition_context=DEFAULT_TRANSITION_CONTEXT,
            label_name=None,
            label_type=None,
            hard_negative_mask=True,
            easy_negative_mask=True,
            positive_to_negative=0.5,
            prefer_hard_negative=False,
            sample_mode="even",
            features=None,
            seq_len=1,
            msg=True,
    ):
        return self.get_sampled_X_Y(
            self.val_frames,
            transition_context=transition_context,
            label_name=label_name,
            label_type=label_type,
            hard_negative_mask=hard_negative_mask,
            easy_negative_mask=easy_negative_mask,
            positive_to_negative=positive_to_negative,
            prefer_hard_negative=prefer_hard_negative,
            sample_mode=sample_mode,
            features=features,
            msg=msg,
        )

    def get_test_X_Y(
            self,
            transition_context=DEFAULT_TRANSITION_CONTEXT,
            label_name=None,
            label_type=None,
            features=None,
            seq_len=1,
    ):
        return self.get_realistic_X_Y(
            self.test_frames,
            transition_context=transition_context,
            label_name=label_name,
            label_type=label_type,
            features=features
        )

    def index_of_features(self, feature_name: str):
        if feature_name in self.feature_list:
            return self.feature_list.index(feature_name)

    @staticmethod
    def is_valid_tick(frames):
        return frames["targetExists"] == True

    @staticmethod
    def target_distribution(X, Y=None):
        if Y is None:
            Y = X
        Y = np.asarray(Y).reshape(-1)
        values, counts = np.unique(Y.astype(int), return_counts=True)
        result = {int(v): int(c) for v, c in zip(values, counts)}
        print(f"target_distribution: {result}")
        return result
