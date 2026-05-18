from models.datasets import Datasets
from models.datasets import TRAIN_FOLDER


FEATURES = ["myMotionX", "myMotionY", "myMotionZ", "myYaw", "myPitch",
            "myMoveForward", "myMoveStrafing", 'targetRelX', 'targetRelY', 'targetRelZ', 'targetDistance',
            'targetMotionX', 'targetMotionY', 'targetMotionZ', 'yawDiffToTarget', 'pitchDiffToTarget']

FEATURES_HISTORY = ["myMotionX", "myMotionY", "myMotionZ", "myYaw", "myPitch", "myDeltaYaw", "myDeltaPitch",
            "myMoveForward", "myMoveStrafing", 'targetRelX', 'targetRelY', 'targetRelZ', 'targetDistance',
            'targetMotionX', 'targetMotionY', 'targetMotionZ', 'targetYaw', 'targetPitch', 'targetDeltaYaw',
            'targetDeltaPitch', 'yawDiffToTarget', 'pitchDiffToTarget']

TARGETS = ["myDeltaYaw", "myDeltaPitch"]

class AimDataset:
    def __init__(self, feature_mode = "MLP"):
        self.datasets = Datasets(TRAIN_FOLDER, keep_frames=False)
        self.train_data = self.datasets.get_train_data()
        self.val_data = self.datasets.get_val_data()
        self.test_data = self.datasets.get_test_data()
        self.labels = self.datasets.get_labels()
        if feature_mode == "MLP":
            self.feature_list = FEATURES
        elif feature_mode == "RNN":
            self.feature_list = FEATURES_HISTORY

    def make_windows(self, rounds, seq_len = 1, stride = 1):
        return self.datasets.make_windows(rounds, self.feature_list, TARGETS, seq_len, stride)

    def get_features_targets(self, rounds, seq_len = 1, stride = 1):
        return self.make_windows(rounds, seq_len, stride)

    def get_train_windows(self, seq_len = 1, stride = 1):
        return self.get_features_targets(self.train_data, seq_len, stride)

    def get_val_windows(self, seq_len = 1, stride = 1):
        return self.get_features_targets(self.val_data, seq_len, stride)

    def get_test_windows(self, seq_len = 1, stride = 1):
        return self.get_features_targets(self.test_data, seq_len, stride)
