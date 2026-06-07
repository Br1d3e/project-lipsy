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
            self.train_rounds = self.datasets.get_train_rounds()
            self.val_rounds = self.datasets.get_val_rounds()
            self.test_rounds = self.datasets.get_test_rounds()

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

    def select_seq_X_Y(
            self,
            frames: pd.DataFrame,
            features=None,
            target=None,
            seq_len=1,
            input_mask_fn=None,
            candidate_mask_fn=None,
            target_mask_fn=None,
            default_features=None,
    ):
        if features is None:
            features = self.feature_list
        if default_features is None:
            default_features = FEATURE_DEFAULTS

        if frames.empty:
            feature_dim = len(features) if features is not None else 0
            target_dim = len(target) if target is not None else 0
            return np.empty((0, seq_len, feature_dim), dtype=np.float32), np.empty((0, target_dim),
                                                                                   dtype=np.float32)
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")

        features_df = frames[features].copy() if features is not None else frames.copy()
        features_df = features_df.fillna(default_features)

        targets_value = np.zeros(len(features_df), dtype=np.float32)
        if isinstance(target, int) or isinstance(target, float):
            targets_value.fill(target)
        elif isinstance(target, np.ndarray):
            if target.shape[0] != len(features_df):
                raise ValueError(f"Invalid target shape. Expected {len(features_df)}, got {target.shape[0]}.")
            targets_value = target.astype(np.float32, copy=False)
        elif callable(target_mask_fn):
            targets_value = np.asarray(target_mask_fn(frames), dtype=np.float32)
            if targets_value.shape[0] != len(features_df):
                raise ValueError(f"Invalid target mask shape. Expected {len(features_df)}, got {targets_value.shape[0]}.")
        else:
            raise ValueError("No target specified")

        if callable(candidate_mask_fn):
            candidate_mask = np.asarray(candidate_mask_fn(frames), dtype=bool)
        elif candidate_mask_fn is None:
            candidate_mask = np.ones(len(frames), dtype=bool)
        else:
            candidate_mask = np.asarray(candidate_mask_fn, dtype=bool)
        if candidate_mask.shape[0] != len(frames):
            raise ValueError(f"Invalid candidate mask shape. Expected {len(frames)}, got {candidate_mask.shape[0]}")

        input_valid_mask = np.ones(len(frames), dtype=bool)
        if input_mask_fn is not None:
            input_valid_mask = np.asarray(input_mask_fn(frames), dtype=bool)
        if input_valid_mask.shape[0] != len(frames):
            raise ValueError(f"Invalid input mask shape. Expected {len(frames)}, got {input_valid_mask.shape[0]}")

        frames = frames[input_valid_mask].copy()
        features_df = features_df[input_valid_mask]
        feature_matrix = features_df.to_numpy(np.float32)
        targets_value = targets_value[input_valid_mask]
        candidate_mask = candidate_mask[input_valid_mask]

        X = []
        Y = []

        if "round_id" in frames.columns:
            round_groups = frames.groupby(frames["round_id"], sort=False)
        else:
            round_groups = [(None, frames)]

        for _, round_df in round_groups:
            round_idx = frames.index.get_indexer(round_df.index)
            if len(round_idx) < seq_len:
                continue

            round_candidate_mask = candidate_mask[round_idx]
            for end in range(seq_len - 1, len(round_idx)):
                start = end - seq_len + 1

                if end >= len(round_idx):
                    continue

                keep = round_candidate_mask[end]
                if not keep:
                    continue

                frame_idx = round_idx[start:end + 1]
                target_idx = round_idx[end]
                x = feature_matrix[frame_idx]
                y = targets_value[target_idx]
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    continue
                X.append(x)
                Y.append(y)

        feature_dim = feature_matrix.shape[1]
        target_dim = 1 if targets_value.ndim == 1 else targets_value.shape[1]
        if not X:
            return np.empty((0, seq_len, feature_dim), dtype=np.float32), np.empty((0, target_dim),
                                                                                   dtype=np.float32)

        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        return np.asarray(X, dtype=np.float32), Y

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

    @staticmethod
    def shuffle_X_Y(X: np.ndarray, Y: np.ndarray, seed=42):
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have the same dimension 0, got X:{X.shape!r} and Y:{Y.shape!r} instead.")
        if X.shape[0] <= 1:
            return X, Y
        rng = np.random.default_rng(seed=seed)
        idx = rng.permutation(X.shape[0])
        return X[idx], Y[idx]

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

        sampled_X, sampled_Y = self.concatenate_X_Y(pos_X, pos_Y, negative_X, negative_Y)
        return self.shuffle_X_Y(sampled_X, sampled_Y)


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

    def get_seq_sampled_X_Y(
            self,
            frames,
            seq_len=DEFAULT_TRANSITION_CONTEXT,
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
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        # if seq_len < transition_context:
        #     print(f"Warning: seq_len={seq_len} is smaller than transition_context={transition_context}.")

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

        pos_X, pos_Y = self.select_seq_X_Y(
            frames=frames,
            features=features,
            seq_len=seq_len,
            input_mask_fn=lambda f:
                valid_mask,
            candidate_mask_fn=lambda f:
                positive_mask,
            target=1
        )
        hard_X, hard_Y = self.select_seq_X_Y(
            frames,
            features=features,
            seq_len=seq_len,
            input_mask_fn = lambda f:
                valid_mask,
            candidate_mask_fn=lambda f:
                hard_negative_mask &
                ~positive_mask,
            target=0
        )
        easy_X, easy_Y = self.select_seq_X_Y(
            frames,
            features=features,
            seq_len=seq_len,
            input_mask_fn=lambda f:
                valid_mask,
            candidate_mask_fn=lambda f:
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

    def get_seq_realistic_X_Y(
            self,
            frames,
            seq_len=DEFAULT_TRANSITION_CONTEXT,
            transition_context=DEFAULT_TRANSITION_CONTEXT,
            label_name=None,
            label_type=None,
            features=None,
    ):
        if label_name is None:
            label_name = self.label_name
        if label_type is None:
            label_type = self.label_type
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        # if seq_len < transition_context:
            # print(f"Warning: seq_len={seq_len} is smaller than transition_context={transition_context}.")

        positive_mask = self.positive_prev_mask(frames, transition_context, label_name, label_type)
        targets = np.asarray(positive_mask, dtype=np.float32)
        X, Y = self.select_seq_X_Y(
            frames=frames,
            features=features,
            seq_len=seq_len,
            input_mask_fn=lambda f: self.is_valid_tick(frames),
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
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        if seq_len == 1:
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
        return self.get_seq_sampled_X_Y(
            self.train_frames,
            seq_len=seq_len,
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
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        if seq_len == 1:
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
        return self.get_seq_sampled_X_Y(
            self.val_frames,
            seq_len=seq_len,
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
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        if seq_len == 1:
            return self.get_realistic_X_Y(
                self.test_frames,
                transition_context=transition_context,
                label_name=label_name,
                label_type=label_type,
                features=features
            )
        return self.get_seq_realistic_X_Y(
            self.test_frames,
            seq_len=seq_len,
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


if __name__ == "__main__":
    ds = HLDecisionDataset(label_name="pearl", label_type=["tryhard", "side"], use_cache=True, build_cache=False)
    train_X, train_Y = ds.get_train_X_Y(transition_context=20, seq_len=40)
    test_X, test_Y = ds.get_test_X_Y(transition_context=20, seq_len=40)

    print(train_X.shape, train_Y.shape)
    print(test_X.shape, test_Y.shape)
