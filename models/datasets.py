import json
import os

import pandas as pd
import numpy as np

TRAIN_FOLDER = "/Users/br1d3e/Library/CloudStorage/GoogleDrive-wangwangjiman@gmail.com/My Drive/Project Lipsy - Continued/project-lipsy/recorder-1.7.10/recordings"
# TRAIN_FOLDER = os.path.join(FOLDER, "session_1778405758091")
# TRAIN_FOLDER = os.path.join(FOLDER, "session_1778405079341")

FRAMES_NAME = "frames.jsonl.gz"
LABELS_NAME = "labels.json"
EVENTS_NAME = "events.jsonl"
ROUND_START_TRIM_TICKS = 20
ROUND_END_TRIM_TICKS = 20

class Datasets:
    def __init__(self, folder, keep_frames=True, use_cache=False, build_cache=True):
        self.folder = folder
        self.keep_frames = keep_frames
        self.cache_dir = os.path.join(TRAIN_FOLDER, ".cache")

        if use_cache:
            self.frames, self.labels, self.rounds = None, None, None
            self.train_rounds, self.val_rounds, self.test_rounds = None, None, None     # deprecate rounds list
            self.train_frames = self.load_pickle_cache("train")
            self.val_frames = self.load_pickle_cache("val")
            self.test_frames = self.load_pickle_cache("test")
        else:
            if "/" in folder and "session" not in folder:
                self.frames = []
                self.labels = []
                self.rounds = []
                for session_frame, session_label, session_events in self.iter_folders(folder):
                    if self.keep_frames:
                        self.frames.append(session_frame)
                    self.labels.append(session_label)
                    self.rounds.extend(self.split_rounds(session_frame, session_label))
            else:
                self.frames, self.labels = self.load_folder(folder)     # session frame / labels
                self.rounds = self.split_rounds(self.frames, self.labels)
                if not self.keep_frames:
                    self.frames = None
            self.train_rounds, self.val_rounds, self.test_rounds = self.split_train_val_test(self.rounds)
            self.train_frames = self.concatenate_frames(self.train_rounds)
            self.val_frames = self.concatenate_frames(self.val_rounds)
            self.test_frames = self.concatenate_frames(self.test_rounds)

        if build_cache:
            self.save_pickle_cache(self.train_frames, "train")
            self.save_pickle_cache(self.val_frames, "val")
            self.save_pickle_cache(self.test_frames, "test")

    def _read_jsonl(self, path):
        if not path:
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if path[-3:] == ".gz":
                return pd.read_json(path, lines=True, compression="gzip")
            return pd.read_json(f, lines=True)

    def cache_name(self, name):
        return f"ds_v2_{name}.pickle"

    def load_pickle_cache(self, name):
        path = os.path.join(self.cache_dir, self.cache_name(name))
        if os.path.exists(path):
            return pd.read_pickle(path)
        raise FileNotFoundError(f"Cache {self.cache_name(name)} not found.")

    def save_pickle_cache(self, frames, name):
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, self.cache_name(name))
        frames.reset_index(drop=True).to_pickle(path)

    def load_folders(self, folder):
        frames = []
        labels = []
        for session_frames, session_labels, session_events in self.iter_folders(folder):
            frames.append(session_frames)
            labels.append(session_labels)

        return frames, labels

    def iter_folders(self, folder):
        for session in sorted(os.listdir(folder)):
            session_path = os.path.join(folder, session)
            if not os.path.isdir(session_path):
                continue

            frames_path = os.path.join(session_path, FRAMES_NAME)
            if not os.path.exists(frames_path):
                continue
            session_frames = self._read_jsonl(frames_path)
            self._col_holding_sword(session_frames)
            self._derived_features(session_frames)
            labels_path = os.path.join(session_path, LABELS_NAME)
            if not os.path.exists(labels_path):
                continue
            with open(labels_path, "r") as f:
                session_labels = json.load(f)

            session_labels = session_labels["labels"]
            session_labels = sorted(session_labels, key=lambda l: l["start_tick"])

            events_path = os.path.join(session_path, EVENTS_NAME)
            if not os.path.exists(events_path):
                continue
            # session_events = self._read_jsonl(events_path)
            session_events = None

            yield session_frames, session_labels, session_events

    def load_folder(self, folder):
        frames_path = os.path.join(folder, FRAMES_NAME)
        frames = self._read_jsonl(frames_path)
        self._col_holding_sword(frames)
        self._derived_features(frames)
        labels_path = os.path.join(folder, LABELS_NAME)
        with open(labels_path, "r") as f:
            labels = json.load(f)

        labels = labels["labels"]
        labels = sorted(labels, key=lambda l: l["start_tick"])

        return frames, labels

    def split_rounds(self, frames: pd.DataFrame, labels=None):
        # Round start: player holds fire resistance before entering the match.
        # Round end: servers put item.paper in the hotbar as the play-again item.
        rounds = []
        if frames.empty:
            return rounds

        frames = frames.sort_values("recordingTickIndex").reset_index(drop=True)
        ticks = frames["recordingTickIndex"].astype(int)
        held_items = frames.get("myHeldItem", pd.Series("", index=frames.index)).fillna("").astype(str).str.lower()
        held_types = frames.get("myHeldItemType", pd.Series("", index=frames.index)).fillna("").astype(str).str.lower()
        hotbars = frames.get("hotbar", pd.Series("", index=frames.index)).map(self.hotbar_to_text)

        is_fire_res = held_items.str.contains("fire resistance", regex=False) | held_types.str.contains("fire_resistance", regex=False)
        has_play_again_paper = hotbars.str.contains("item.paper", regex=False)
        fire_res_start = is_fire_res & ~is_fire_res.shift(1, fill_value=False)
        paper_start = has_play_again_paper & ~has_play_again_paper.shift(1, fill_value=False)

        round_start = None
        if "targetExists" in frames.columns and bool(frames["targetExists"].iloc[0]) and not bool(has_play_again_paper.iloc[0]):
            round_start = int(ticks.iloc[0])

        for idx in frames.index:
            tick = int(ticks.loc[idx])
            if round_start is None:
                if bool(fire_res_start.loc[idx]):
                    round_start = tick
                continue

            if bool(paper_start.loc[idx]) and tick > round_start:
                round_end = tick
                start_tick = round_start + ROUND_START_TRIM_TICKS
                end_tick = round_end - ROUND_END_TRIM_TICKS
                if start_tick <= end_tick:
                    round_df = frames[
                        (frames["recordingTickIndex"] >= start_tick) &
                        (frames["recordingTickIndex"] <= end_tick)
                    ].copy()
                    round_df.attrs["labels"] = labels or []
                    rounds.append(round_df)
                round_start = None

        return rounds

    @staticmethod
    def hotbar_to_text(hotbar):
        if isinstance(hotbar, list):
            return " ".join(str(item).lower() for item in hotbar if item is not None)
        return str(hotbar).lower()

    def split_train_val_test(self, rounds):
        round_len = len(rounds)
        if round_len == 0:
            return [], [], []
        if round_len == 1:
            return rounds, [], []
        if round_len == 2:
            return rounds[:1], [], rounds[1:]

        train_len = int(round_len * 0.80)
        val_len = max(1, int(round_len * 0.10))
        if train_len + val_len >= round_len:
            train_len = round_len - 2
            val_len = 1
        test_len = round_len - train_len - val_len
        train_rounds = rounds[:train_len]
        val_rounds = rounds[train_len:train_len + val_len]
        test_rounds = rounds[train_len + val_len:]

        return train_rounds, val_rounds, test_rounds

    def _col_labels(self, frames: pd.DataFrame, labels) -> pd.DataFrame:
        frames = frames.iloc[np.argsort(frames["recordingTickIndex"].to_numpy())].copy()
        frames["label"] = None
        frames["label_type"] = None
        frames["label_start"] = -1

        frames_ticks = frames["recordingTickIndex"].to_numpy()
        frames_start = frames_ticks.min()
        frames_end = frames_ticks.max()

        for label in labels:
            label_name = self.label_name(label).strip().lower()
            label_start = label["start_tick"]
            label_end = label["end_tick"]

            overlap_start = max(frames_start, label_start)
            overlap_end = min(frames_end, label_end)
            if overlap_start >= overlap_end:
                continue

            if label_name in ["attack", "heal", "pearl", "refill", "speed", "eat"]:
                row_idx = np.zeros(len(frames), dtype=bool)
                start_idx = np.searchsorted(frames_ticks, overlap_start, side="left")
                end_idx = np.searchsorted(frames_ticks, overlap_end, side="right")
                row_idx[start_idx:end_idx] = True

                frames.loc[row_idx, f"s_{label_name}"] = 1

                frames.loc[row_idx, "label"] = label_name
                frames.loc[row_idx, "label_type"] = label["type"]
                frames.loc[row_idx, "label_start"] = label["start_tick"]

        return frames

    def _col_event_target_heal(self, frames: pd.DataFrame, events) -> pd.DataFrame:
        ...

    def concatenate_frames(self, rounds):
        parts = []
        for i, round_df in enumerate(rounds):
            round_df = round_df.set_index("recordingTickIndex", drop=False)
            round_df["round_id"] = i
            round_df = self._col_labels(round_df, self.labels_for_round(round_df))
            round_df = round_df.set_index(["round_id", "recordingTickIndex"], drop=False)

            parts.append(round_df)

        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, axis=0, copy=False)

    def get_datasets_rounds(self):
        return self.train_rounds, self.val_rounds, self.test_rounds

    def get_train_rounds(self):
        if self.train_rounds is None:
            raise ValueError("train_rounds is not available when using cached frames.")
        return self.train_rounds
    def get_val_rounds(self):
        if self.val_rounds is None:
            raise ValueError("val_rounds is not available when using cached frames.")
        return self.val_rounds
    def get_test_rounds(self):
        if self.test_rounds is None:
            raise ValueError("test_rounds is not available when using cached frames.")
        return self.test_rounds

    def get_train_frames(self):
        return self.train_frames
    def get_val_frames(self):
        return self.val_frames
    def get_test_frames(self):
        return self.test_frames

    def get_labels(self):
        return self.labels

    def select_X_Y(
            self,
            frames: pd.DataFrame,
            features=None,
            targets=None,
            seq_len=1,
            feature_mask_fn=None,
            target_mask_fn=None,
            filter_mode="last",
    ):
        if filter_mode not in ["last", "all", None]:
            raise ValueError(f"filter_mode must be last, all, or None, got {filter_mode!r}")

        if frames.empty:
            feature_dim = len(features) if features is not None else 0
            target_dim = len(targets) if targets is not None else 0
            return np.empty((0, seq_len, feature_dim), dtype=np.float32), np.empty((0, target_dim), dtype=np.float32)
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")

        features_df = frames[features] if features is not None else frames
        feature_matrix = features_df.to_numpy(np.float32)

        targets_df = frames[targets] if targets is not None else frames
        targets_value = targets_df.to_numpy(np.float32)

        if callable(feature_mask_fn):
            feature_mask = np.asarray(feature_mask_fn(frames), dtype=bool)
        elif feature_mask_fn is None:
            feature_mask = np.ones(len(frames), dtype=bool)
        else:
            feature_mask = np.asarray(feature_mask_fn, dtype=bool)
        if feature_mask.shape[0] != len(frames):
            raise ValueError(f"Invalid feature mask shape. Expected {len(frames)}, got {feature_mask.shape[0]}")

        if callable(target_mask_fn):
            target_mask = np.asarray(target_mask_fn(frames), dtype=bool)
        elif target_mask_fn is None:
            target_mask = np.ones(len(frames), dtype=bool)
        else:
            target_mask = np.asarray(target_mask_fn, dtype=bool)
        if target_mask.shape[0] != len(frames):
            raise ValueError(f"Invalid target mask shape. Expected {len(frames)}, got {target_mask.shape[0]}")

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

            round_feature_mask = feature_mask[round_idx]
            round_target_mask = target_mask[round_idx]
            for end in range(seq_len - 1, len(round_idx)):
                start = end - seq_len + 1
                target_pos = end + 1

                if target_pos >= len(round_idx):
                    continue

                if filter_mode == "last":
                    keep = round_feature_mask[end]
                elif filter_mode == "all":
                    keep = round_feature_mask[start:end + 1].all()
                else:
                    keep = True
                if not keep or not round_target_mask[target_pos]:
                    continue

                frame_idx = round_idx[start:end + 1]
                target_idx = round_idx[target_pos]
                x = feature_matrix[frame_idx]
                y = targets_value[target_idx]
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    continue
                X.append(x)
                Y.append(y)

        feature_dim = feature_matrix.shape[1]
        target_dim = targets_value.shape[1]
        if not X:
            return np.empty((0, seq_len, feature_dim), dtype=np.float32), np.empty((0, target_dim), dtype=np.float32)

        return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)

    @staticmethod
    def label_active_mask(
            frames: pd.DataFrame,
            label_name=None,
            label_type=None,
    ):
        if label_name not in ["attack", "heal", "pearl", "retreat", "refill", "speed", "eat"]:
            raise ValueError(f"Invalid label filter. Expected ['attack', 'heal', 'pearl', 'retreat', 'refill', "
                             f"'speed', 'eat'], got {label_name!r}")

        if frames.empty or "round_id" not in frames.columns or "recordingTickIndex" not in frames.columns:
            raise ValueError(f"Invalid frames columns. Expected 'round_id', 'recordingTickIndex'")

        label_mask = np.zeros(len(frames), dtype=bool)
        for round_id, round_df in frames.groupby(frames["round_id"], sort=False):
            round_idx = frames.index.get_indexer(round_df.index)

            round_mask = np.ones(len(round_df), dtype=bool)
            if label_name is not None:
                round_mask &= (round_df["label"] == label_name).to_numpy()
            if label_type is not None:
                if isinstance(label_type, (list, tuple, set)):
                    round_mask &= round_df["label_type"].isin(label_type).to_numpy()
                else:
                    round_mask &= (round_df["label_type"] == label_type).to_numpy()

            label_mask[round_idx] = round_mask

        return label_mask

    def select_X_Y_by_label(
            self,
            frames: pd.DataFrame,
            features=None,
            targets=None,
            seq_len=1,
            label_name=None,
            label_type=None,
            feature_mask_fn=None,
            target_mask_fn=None,
            filter_mode="all",
    ):
        label_target_mask = self.label_active_mask(frames, label_name, label_type)
        if target_mask_fn is not None:
            extra_target_mask = target_mask_fn(frames) if callable(target_mask_fn) else target_mask_fn
            extra_target_mask = np.asarray(extra_target_mask, dtype=bool)
            if extra_target_mask.shape[0] != len(frames):
                raise ValueError(f"Invalid target mask shape. Expected {len(frames)}, got {extra_target_mask.shape[0]}")
            label_target_mask &= extra_target_mask
        return self.select_X_Y(
            frames,
            features,
            targets,
            seq_len,
            feature_mask_fn=feature_mask_fn,
            target_mask_fn=label_target_mask,
            filter_mode=filter_mode,
        )

    def make_windows(self, rounds, features, targets, seq_len=1, stride=1, label_filter: str=None,
                     candidate_mask_fn=None):
        data_windows = []
        columns = features + targets
        feature_count = len(features)

        for round_df in rounds:
            round_df = round_df.set_index("recordingTickIndex", drop=False)
            round_start = round_df["recordingTickIndex"].min()
            round_end = round_df["recordingTickIndex"].max()

            for label in self.labels_for_round(round_df):
                if label_filter is not None and self.label_name(label) != label_filter:
                    continue

                label_start = label["start_tick"]
                label_end = label["end_tick"]
                overlap_start = max(round_start, label_start)
                overlap_end = min(round_end, label_end)

                if overlap_start >= overlap_end:
                    continue

                available_mask = (
                    (overlap_start <= round_df["recordingTickIndex"]) &
                    (round_df["recordingTickIndex"] <= overlap_end) &
                    (round_df["targetExists"] == True) &
                    (round_df["myHeldItemType"] == "item.swordDiamond")
                )
                available = round_df.loc[available_mask, columns].sort_index()

                if len(available) <= seq_len:
                    continue

                candidate_mask = np.ones(len(available), dtype=bool)
                if callable(candidate_mask_fn):
                    candidate_mask = np.asarray(candidate_mask_fn(available), dtype=bool)
                if candidate_mask.shape[0] != len(available):
                    raise ValueError(f"Invalid candidate mask shape. Expected {len(available)}, got {candidate_mask.shape[0]}")

                values = available[candidate_mask].to_numpy(dtype=np.float32, copy=True)
                values = values[np.isfinite(values).all(axis=1)]
                if len(values) <= seq_len:
                    continue

                feature_values = values[:, :feature_count]
                target_values = values[:, feature_count:]
                for i in range(0, len(values) - seq_len, stride):
                    data_windows.append((
                        feature_values[i:i + seq_len],
                        target_values[i + seq_len],
                    ))

        return data_windows

    def labels_for_round(self, round_df):
        labels = round_df.attrs.get("labels")
        if labels is not None:
            return labels
        if self.labels and isinstance(self.labels[0], list):
            return [label for session_labels in self.labels for label in session_labels]
        return self.labels

    @staticmethod
    def label_name(label):
        if not isinstance(label, dict):
            return ""
        return str(label.get("label") or label.get("suggested_label") or "").strip().lower()

    def _col_holding_sword(self, frames: pd.DataFrame | None):
        if isinstance(frames, pd.DataFrame):
            frames["myHoldingSword"] = frames["myHeldItemType"] == "item.swordDiamond"

    def _derived_features(self, frames: pd.DataFrame | None):
        if isinstance(frames, pd.DataFrame):
            if "hotbarHealthPotions" not in frames.columns:
                frames["hotbarHealthPotions"] = frames.get("inventoryHealthPotions", 0)
            if "ticksSinceLastBlock" not in frames.columns:
                frames["ticksSinceLastBlock"] = 100000
            if "targetEstimatedPotions" not in frames.columns:
                frames["targetEstimatedPotions"] = 29
            frames["canHeal"] = (frames["inventoryHealthPotions"] > 0) & (frames["myHp"] < 15)    # loose hp boundary
            frames["canSpeed"] = frames["inventorySpeedPotions"] > 0
            frames["canEat"] = (frames["inventoryFood"] > 0) & (frames["myFoodLevel"] < 20)
            frames["canPearl"] = (frames["inventoryPearls"] > 0) & (frames["pearlCooldownTicksRemaining"] <= 0)
