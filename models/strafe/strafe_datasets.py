import numpy as np
import pandas as pd

from models.datasets import Datasets
from models.datasets import TRAIN_FOLDER


FEATURES = ["myMotionX", "myMotionY", "myMotionZ", "myYaw", "myDeltaYaw", "myMoveForward", "myMoveStrafing",
            'targetRelX', 'targetRelY', 'targetRelZ', 'targetDistance',
            'targetMotionX', 'targetMotionY', 'targetMotionZ', 'yawDiffToTarget', 'pitchDiffToTarget']

TARGETS = ["aStrafe"]

class StrafeDataset:
    def __init__(self):
        self.datasets = Datasets(TRAIN_FOLDER)
        self.train_data = self.datasets.get_train_data()
        self.val_data = self.datasets.get_val_data()
        self.test_data = self.datasets.get_test_data()
        self.labels = self.datasets.get_labels()
        self.feature_list = FEATURES

    def make_windows(self, rounds: list[pd.DataFrame], seq_len = 1, stride = 1):
        windows = []
        for round_df in rounds:
            round_df = round_df.set_index("recordingTickIndex", drop=False)
            round_start = round_df["recordingTickIndex"].min()
            round_end = round_df["recordingTickIndex"].max()
            for label in self.labels:
                label_name = label["label"].strip().lower()
                if label_name == "attack":
                    label_start = label["start_tick"]
                    label_end = label["end_tick"]

                    overlap_start = max(round_start, label_start)
                    overlap_end = min(round_end, label_end)

                    if overlap_start >= overlap_end:
                        continue

                    available = round_df[np.logical_and(overlap_start <= round_df["recordingTickIndex" ], round_df["recordingTickIndex"] <= overlap_end)]
                    available = available.sort_index()
                    available = available[
                        (available["targetExists"] == True) &
                        np.isfinite(available[self.feature_list + TARGETS]).all(axis=1)
                    ]

                    if len(available) <= seq_len:
                        continue
                    for i in range(0, len(available) - seq_len, stride):
                        window = available.iloc[i:i + seq_len]
                        target = available.iloc[i + seq_len][TARGETS]
                        windows.append((window, target))
        return windows

    def get_features_targets(self, rounds, seq_len = 1, stride = 1):
        data_windows = []
        windows = self.make_windows(rounds, seq_len, stride)
        for window, target in windows:
            feature = window[self.feature_list].to_numpy(dtype=np.float32)
            target = target.to_numpy(dtype=np.float32)
            data_windows.append((feature, target))

        return data_windows

    def get_train_windows(self, seq_len = 1, stride = 1):
        return self.get_features_targets(self.train_data, seq_len, stride)

    def get_val_windows(self, seq_len = 1, stride = 1):
        return self.get_features_targets(self.val_data, seq_len, stride)

    def get_test_windows(self, seq_len = 1, stride = 1):
        return self.get_features_targets(self.test_data, seq_len, stride)
