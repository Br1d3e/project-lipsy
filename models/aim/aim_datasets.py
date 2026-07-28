import numpy as np

from models.datasets import Datasets
from models.datasets import TRAIN_FOLDER


FEATURES = ["myMotionX", "myMotionY", "myMotionZ", "myYaw", "myPitch",
            "myMoveForward", "myMoveStrafing", 'targetRelX', 'targetRelY', 'targetRelZ', 'targetDistance',
            'targetMotionX', 'targetMotionY', 'targetMotionZ', 'yawDiffToTarget', 'pitchDiffToTarget']

FEATURES_HISTORY = ["myMotionX", "myMotionY", "myMotionZ", "myYaw", "myPitch", "myDeltaYaw", "myDeltaPitch",
            "myMoveForward", "myMoveStrafing", 'targetRelX', 'targetRelY', 'targetRelZ', 'targetDistance',
            'targetMotionX', 'targetMotionY', 'targetMotionZ', 'yawDiffToTarget', 'pitchDiffToTarget']

TARGETS = ["myDeltaYaw", "myDeltaPitch"]
DEFAULT_YAW_BUCKETS = (15.0, 45.0)
DEFAULT_YAW_BUCKET_DISTRIBUTION = (0.4, 0.3, 0.3)

class AimDataset:
    def __init__(self, feature_mode="MLP", use_cache=False, build_cache=True,
                 yaw_buckets=DEFAULT_YAW_BUCKETS,
                 yaw_bucket_distribution=DEFAULT_YAW_BUCKET_DISTRIBUTION):
        if len(yaw_buckets) != 2 or yaw_buckets[0] < 0 or yaw_buckets[0] >= yaw_buckets[1]:
            raise ValueError("yaw_buckets must contain two increasing thresholds")
        if len(yaw_bucket_distribution) != 3 or any(value < 0 for value in yaw_bucket_distribution):
            raise ValueError("yaw_bucket_distribution must contain three non-negative values")
        distribution_sum = sum(yaw_bucket_distribution)
        if distribution_sum <= 0:
            raise ValueError("yaw_bucket_distribution must have a positive sum")
        self.yaw_buckets = yaw_buckets
        self.yaw_bucket_distribution = np.asarray(yaw_bucket_distribution, dtype=np.float64) / distribution_sum
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
    def get_X_Y(self, frames, seq_len=1, feature_mask_fn=None, target_mask_fn=None, filter_mode="all"):
        if feature_mask_fn is None:
            feature_mask_fn = self.aim_mask_fn
        return self.datasets.select_X_Y_by_label(
            frames, self.feature_list, TARGETS, seq_len,
            label_name="attack",
            feature_mask_fn=feature_mask_fn,
            target_mask_fn=target_mask_fn,
            filter_mode=filter_mode,
        )

    def yaw_bucket_mask(self, frames, bucket):
        abs_delta_yaw = frames["myDeltaYaw"].abs()
        small_max, medium_max = self.yaw_buckets
        if bucket == "small":
            return abs_delta_yaw <= small_max
        if bucket == "medium":
            return (abs_delta_yaw > small_max) & (abs_delta_yaw <= medium_max)
        if bucket == "large":
            return abs_delta_yaw > medium_max
        raise ValueError(f"Unknown yaw bucket: {bucket!r}")

    def get_bucketed_X_Y(self, frames, seq_len=1):
        bucket_data = [
            self.get_X_Y(
                frames,
                seq_len,
                feature_mask_fn=lambda f, name=name: self.aim_mask_fn(f) & self.yaw_bucket_mask(f, name),
                filter_mode="last",
            )
            for name in ("small", "medium", "large")
        ]
        bucket_counts = np.asarray([X.shape[0] for X, _ in bucket_data])
        if np.any((bucket_counts == 0) & (self.yaw_bucket_distribution > 0)):
            raise ValueError(f"Cannot sample requested yaw distribution; bucket counts are {bucket_counts.tolist()}")

        X = np.concatenate([data[0] for data in bucket_data], axis=0)
        Y = np.concatenate([data[1] for data in bucket_data], axis=0)
        sample_weights = np.concatenate([
            np.full(count, desired / count if count else 0.0, dtype=np.float64)
            for count, desired in zip(bucket_counts, self.yaw_bucket_distribution)
        ])
        return X, Y, sample_weights

    def get_features_targets(self, rounds, seq_len = 1, stride = 1):
        return self.make_windows(rounds, seq_len, stride)

    def get_train_windows(self, seq_len = 1):
        return self.get_features_targets(self.datasets.get_train_rounds(), seq_len)

    def get_val_windows(self, seq_len = 1):
        return self.get_features_targets(self.datasets.get_val_rounds(), seq_len)

    def get_test_windows(self, seq_len = 1):
        return self.get_features_targets(self.datasets.get_test_rounds(), seq_len)

    def get_train_X_Y(self, seq_len=1, balance_yaw_buckets=True, return_sample_weights=False):
        if balance_yaw_buckets:
            X, Y, sample_weights = self.get_bucketed_X_Y(self.train_frames, seq_len)
        else:
            X, Y = self.get_X_Y(self.train_frames, seq_len)
            sample_weights = np.ones(X.shape[0], dtype=np.float64)
        if return_sample_weights:
            return X, Y, sample_weights
        return X, Y

    def get_val_X_Y(self, seq_len = 1):
        return self.get_X_Y(self.val_frames, seq_len)

    def get_test_X_Y(self, seq_len = 1):
        return self.get_X_Y(self.test_frames, seq_len)

if __name__ == "__main__":
    from time import time
    b4 = time()
    ds = AimDataset(use_cache=True, build_cache=False)
    train_X, train_Y, sample_weights = ds.get_train_X_Y(seq_len=20, balance_yaw_buckets=True, return_sample_weights=True)
    print(train_X.shape, train_Y.shape, "time used:", time() - b4)
    print(np.unique(sample_weights, return_counts=True))
