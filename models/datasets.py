import json
import os

import pandas as pd
import numpy as np

FOLDER = "/Users/br1d3e/Library/CloudStorage/GoogleDrive-wangwangjiman@gmail.com/My Drive/Project Lipsy - Continued/project-lipsy/recorder-1.7.10/recordings"
TRAIN_FOLDER = os.path.join(FOLDER, "session_1778405758091")
# TRAIN_FOLDER = os.path.join(FOLDER, "session_1778405079341")

FRAMES_NAME = "frames.jsonl.gz"
LABELS_NAME = "labels.json"

class Datasets:
    def __init__(self, folder):
        self.folder = folder
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
        # gap = targetExists = false, -100 <= pos <= 100 ～ hold & use fire res potion
        rounds = []
        round_start = -1
        round_end = -1
        for idx, frame in frames.iterrows():
            if round_start != -1 and round_end != -1:
                round = frames[
                    (frames["recordingTickIndex"] >= round_start + 50) &
                    (frames["recordingTickIndex"] <= round_end - 50)
                    ].copy()
                rounds.append(round)
                round_start = -1
                round_end = -1
                continue

            my_item = frame["myHeldItem"]
            is_fire_res = "fire resistance" in my_item.lower() if isinstance(my_item, str) else False
            if round_start == -1 and is_fire_res:
                round_start = frame["recordingTickIndex"]
            myPosX = frame["myPosX"]
            myPosZ = frame["myPosZ"]
            if pd.isna(myPosX):
                myPosX = 10000
            if pd.isna(myPosZ):
                myPosZ = 10000
            is_spawn = -100 < myPosX < 100 and -100 < myPosZ < 100
            # is_target_dead = frame["targetExists"] == False
            if is_spawn:
                round_end = frame["recordingTickIndex"]

        # print(len(rounds))
        return rounds

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

