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
    def __init__(self, folder):
        self.folder = folder
        if "/" in folder and "session" not in folder:
            self.frames, self.labels = self.load_folders(folder)
            self.rounds = []
            for session_frame, session_label in zip(self.frames, self.labels):
                self.rounds.extend(self.split_rounds(session_frame))
        else:
            self.frames, self.labels = self.load_folder(folder)
            self.rounds = self.split_rounds(self.frames)
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
        for session in sorted(os.listdir(folder)):
            session_path = os.path.join(folder, session)
            if not os.path.isdir(session_path):
                continue

            frames_path = os.path.join(session_path, FRAMES_NAME)
            session_frames = self._read_jsonl(frames_path)
            labels_path = os.path.join(session_path, LABELS_NAME)
            with open(labels_path, "r") as f:
                session_labels = json.load(f)

            session_labels = session_labels["labels"]

            frames.append(session_frames)
            labels.append(session_labels)

        return frames, labels

    def load_folder(self, folder):
        frames_path = os.path.join(folder, FRAMES_NAME)
        frames = self._read_jsonl(frames_path)
        labels_path = os.path.join(folder, LABELS_NAME)
        with open(labels_path, "r") as f:
            labels = json.load(f)

        labels = labels["labels"]
        labels = sorted(labels, key=lambda l: l["start_tick"])

        return frames, labels

    def split_rounds(self, frames: pd.DataFrame):
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
                    rounds.append(frames[
                        (frames["recordingTickIndex"] >= start_tick) &
                        (frames["recordingTickIndex"] <= end_tick)
                    ].copy())
                round_start = None

        return rounds

    @staticmethod
    def hotbar_to_text(hotbar):
        if isinstance(hotbar, list):
            return " ".join(str(item).lower() for item in hotbar if item is not None)
        return str(hotbar).lower()

    def split_train_val_test(self, rounds):
        round_len = len(rounds)
        print(round_len)

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
