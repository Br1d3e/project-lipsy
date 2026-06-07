from models.datasets import Datasets
from models.datasets import TRAIN_FOLDER


FEATURES = ["myMotionX", "myMotionY", "myMotionZ", "myYaw", "myPitch",
            "myMoveForward", "myMoveStrafing", 'targetRelX', 'targetRelY', 'targetRelZ', 'targetDistance',
            'targetMotionX', 'targetMotionY', 'targetMotionZ', 'yawDiffToTarget', 'pitchDiffToTarget']

FEATURES_HISTORY = ["myMotionX", "myMotionY", "myMotionZ", "myYaw", "myPitch", "myDeltaYaw", "myDeltaPitch",
            "myMoveForward", "myMoveStrafing", 'targetRelX', 'targetRelY', 'targetRelZ', 'targetDistance',
            'targetMotionX', 'targetMotionY', 'targetMotionZ', 'yawDiffToTarget', 'pitchDiffToTarget']

TARGETS = ["myDeltaYaw", "myDeltaPitch"]

class AimDataset:
    def __init__(self, feature_mode = "MLP", use_cache = False, build_cache = True):
        self.datasets = Datasets(TRAIN_FOLDER, keep_frames=False, use_cache=use_cache, build_cache=build_cache)
        # self.train_data = self.datasets.get_train_rounds()
        # self.val_data = self.datasets.get_val_rounds()
        # self.test_data = self.datasets.get_test_rounds()
        self.train_frames = self.datasets.get_train_frames()
        # print(self.train_frames.groupby(self.train_frames["round_id"], sort=True).head())
        self.val_frames = self.datasets.get_val_frames()
        self.test_frames = self.datasets.get_test_frames()
        if feature_mode == "MLP":
            self.feature_list = FEATURES
        elif feature_mode == "RNN":
            self.feature_list = FEATURES_HISTORY

    @staticmethod
    def aim_mask_fn(frames):
        return frames["myHeldItemType"] == "item.swordDiamond"

    def make_windows(self, rounds, seq_len = 1, stride = 1):
        return self.datasets.make_windows(rounds, self.feature_list, TARGETS, seq_len, stride, label_filter="attack",
                                          candidate_mask_fn=None)
    def get_X_Y(self, frames, seq_len = 1, extra_candidate_mask_fn = None ):
        if extra_candidate_mask_fn is None:
            extra_candidate_mask_fn = self.aim_mask_fn
        return self.datasets.select_X_Y_by_label(
            frames, self.feature_list, TARGETS, seq_len,
            label_name="attack",
            extra_candidate_mask_fn=extra_candidate_mask_fn,
        )

    def get_features_targets(self, rounds, seq_len = 1, stride = 1):
        return self.make_windows(rounds, seq_len, stride)

    def get_train_windows(self, seq_len = 1):
        return self.get_features_targets(self.datasets.get_train_rounds(), seq_len)

    def get_val_windows(self, seq_len = 1):
        return self.get_features_targets(self.datasets.get_val_rounds(), seq_len)

    def get_test_windows(self, seq_len = 1):
        return self.get_features_targets(self.datasets.get_test_rounds(), seq_len)

    def get_train_X_Y(self, seq_len = 1):
        return self.get_X_Y(self.train_frames, seq_len)

    def get_val_X_Y(self, seq_len = 1):
        return self.get_X_Y(self.val_frames, seq_len)

    def get_test_X_Y(self, seq_len = 1):
        return self.get_X_Y(self.test_frames, seq_len)

if __name__ == "__main__":
    from time import time
    b4 = time()
    ds = AimDataset(use_cache = True, build_cache = False)
    train_X, train_Y =ds.get_train_X_Y(seq_len=20)
    print(train_X.shape, train_Y.shape, "time used:", time() - b4)
