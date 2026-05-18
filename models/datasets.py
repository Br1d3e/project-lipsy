import json
import os

import pandas as pd
import numpy as np

TRAIN_FOLDER = "/Users/br1d3e/Library/CloudStorage/GoogleDrive-wangwangjiman@gmail.com/My Drive/Project Lipsy - Continued/project-lipsy/recorder-1.7.10/recordings"
# TRAIN_FOLDER = os.path.join(FOLDER, "session_1778405758091")
# TRAIN_FOLDER = os.path.join(FOLDER, "session_1778405079341")

FRAMES_NAME = "frames.jsonl.gz"
LABELS_NAME = "labels.json"
ROUND_START_TRIM_TICKS = 10
ROUND_END_TRIM_TICKS = 20

class Datasets:
    def __init__(self, folder, keep_frames=True):
        self.folder = folder
        self.keep_frames = keep_frames
        if "/" in folder and "session" not in folder:
            self.frames = []
            self.labels = []
            self.rounds = []
            for session_frame, session_label in self.iter_folders(folder):
                if self.keep_frames:
                    self.frames.append(session_frame)
                self.labels.append(session_label)
                self.rounds.extend(self.split_rounds(session_frame, session_label))
        else:
            self.frames, self.labels = self.load_folder(folder)     # session frame / labels
            self.rounds = self.split_rounds(self.frames, self.labels)
            if not self.keep_frames:
                self.frames = None
        self.train_data, self.val_data, self.test_data = self.split_train_val_test(self.rounds)

    def _read_jsonl(self, path):
        if not path:
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if path[-3:] == ".gz":
                return pd.read_json(path, lines=True, compression="gzip")
            return pd.read_json(f, lines=True)


    def load_folders(self, folder):
        frames = []
        labels = []
        for session_frames, session_labels in self.iter_folders(folder):
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
            labels_path = os.path.join(session_path, LABELS_NAME)
            if not os.path.exists(labels_path):
                continue
            with open(labels_path, "r") as f:
                session_labels = json.load(f)

            session_labels = session_labels["labels"]
            session_labels = sorted(session_labels, key=lambda l: l["start_tick"])

            yield session_frames, session_labels

    def load_folder(self, folder):
        frames_path = os.path.join(folder, FRAMES_NAME)
        frames = self._read_jsonl(frames_path)
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

        train_len = int(round_len * 0.80)
        val_len = int(round_len * 0.10)
        test_len = round_len - train_len - val_len
        train_rounds = rounds[:train_len]
        val_rounds = rounds[train_len:train_len + val_len]
        test_rounds = rounds[train_len + val_len:]

        return train_rounds, val_rounds, test_rounds

    def get_datasets(self):
        return self.train_data, self.val_data, self.test_data

    def get_train_data(self):
        return self.train_data
    def get_val_data(self):
        return self.val_data
    def get_test_data(self):
        return self.test_data

    def get_labels(self):
        return self.labels

    def make_windows(self, rounds, features, targets, seq_len=1, stride=1, label_filter="attack"):
        data_windows = []
        columns = features + targets
        feature_count = len(features)

        for round_df in rounds:
            round_df = round_df.set_index("recordingTickIndex", drop=False)
            round_start = round_df["recordingTickIndex"].min()
            round_end = round_df["recordingTickIndex"].max()

            for label in self.labels_for_round(round_df):
                if self.label_name(label) != label_filter:
                    continue

                label_start = label["start_tick"]
                label_end = label["end_tick"]
                overlap_start = max(round_start, label_start)
                overlap_end = min(round_end, label_end)

                if overlap_start >= overlap_end:
                    continue

                available = round_df.loc[
                    (overlap_start <= round_df["recordingTickIndex"]) &
                    (round_df["recordingTickIndex"] <= overlap_end) &
                    (round_df["targetExists"] == True),
                    columns
                ].sort_index()

                if len(available) <= seq_len:
                    continue

                values = available.to_numpy(dtype=np.float32, copy=True)
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
