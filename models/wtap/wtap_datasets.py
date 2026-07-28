from models.datasets import Datasets
from models.datasets import TRAIN_FOLDER


FEATURES = ["myMotionX", "myMotionY", "myMotionZ", "myMoveForward", "myMoveStrafing",
            'targetRelX', 'targetRelY', 'targetRelZ', 'targetDistance', 'myCombo', "targetCombo", "myBlockingSword",
            "myHurtTime", "ticksSinceLastDamageTaken", "ticksSinceLastHitLanded",
            'blockDistanceFront', 'blockDistanceBack', 'blockDistanceLeft', 'blockDistanceRight',
            'aAttack', "mySwingProgress",
            'targetMotionX', 'targetMotionY', 'targetMotionZ', 'yawDiffToTarget', 'pitchDiffToTarget']

TARGETS = ["aForward"]

FEATURE_DEFAULTS = {
    "myBlockingSword": 0.0,
    "ticksSinceLastDamageTaken": 100000.0,
    "ticksSinceLastHitLanded": 100000.0,
    "blockDistanceFront": 50.0,
    "blockDistanceBack": 50.0,
    "blockDistanceLeft": 50.0,
    "blockDistanceRight": 50.0,
}

class WtapDataset:
    def __init__(self, use_cache=False, build_cache=True):
        self.datasets = Datasets(TRAIN_FOLDER, keep_frames=False, use_cache=use_cache, build_cache=build_cache)
        self.train_frames = self.datasets.get_train_frames()
        self.val_frames = self.datasets.get_val_frames()
        self.test_frames = self.datasets.get_test_frames()
        self.feature_list = FEATURES

    @staticmethod
    def wtap_mask_fn(frames):
        return (
            (frames["myHeldItemType"] == "item.swordDiamond") &
            (frames["targetExists"] == True)
        )

    def make_windows(self, rounds, seq_len = 1, stride = 1):
        return self.datasets.make_windows(rounds, self.feature_list, TARGETS, seq_len, stride, label_filter="attack")

    def get_X_Y(self, frames, seq_len=1, feature_mask_fn=None):
        if feature_mask_fn is None:
            feature_mask_fn = self.wtap_mask_fn
        frames = frames.copy()
        frames[list(FEATURE_DEFAULTS)] = frames[list(FEATURE_DEFAULTS)].fillna(FEATURE_DEFAULTS)
        return self.datasets.select_X_Y_by_label(
            frames, self.feature_list, TARGETS, seq_len,
            label_name="attack",
            feature_mask_fn=feature_mask_fn
        )

    def get_features_targets(self, rounds, seq_len = 1, stride = 1):
        return self.make_windows(rounds, seq_len, stride)

    def get_train_windows(self, seq_len = 1, stride = 1):
        return self.get_features_targets(self.datasets.get_train_rounds(), seq_len, stride)

    def get_val_windows(self, seq_len = 1, stride = 1):
        return self.get_features_targets(self.datasets.get_val_rounds(), seq_len, stride)

    def get_test_windows(self, seq_len = 1, stride = 1):
        return self.get_features_targets(self.datasets.get_test_rounds(), seq_len, stride)

    def get_train_X_Y(self, seq_len=1):
        return self.get_X_Y(self.train_frames, seq_len)

    def get_val_X_Y(self, seq_len=1):
        return self.get_X_Y(self.val_frames, seq_len)

    def get_test_X_Y(self, seq_len=1):
        return self.get_X_Y(self.test_frames, seq_len)
