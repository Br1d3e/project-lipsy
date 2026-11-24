"""Evaluate initiation with the same label_start targets and metrics as training."""
from pathlib import Path

from torch.utils.data import DataLoader

from models.hl_decision.hl_decision_datasets import HLDecisionDataset
from models.hl_decision.train_hl_decision_model import TrainTorchModel, SEQ_LEN, BATCH_SIZE
from models.torch_dataset import HLMultiTaskDataset


def evaluate_split(model, dataset, split="test"):
    seq_len = getattr(model, "checkpoint_meta", {}).get("seq_len", SEQ_LEN)
    getters = {"train": dataset.get_train_X_Y, "val": dataset.get_val_X_Y,
               "test": dataset.get_test_X_Y}
    x, y = getters[split](seq_len=seq_len, return_auxiliary_targets=True)
    loader = DataLoader(HLMultiTaskDataset(x, y), batch_size=BATCH_SIZE, shuffle=False)
    print(f"===== {split} set =====")
    return TrainTorchModel(model=model).eval(
        dataloader=loader, inspect_initiation=True, confusion_matrix=True,
    )


if __name__ == "__main__":
    model_path = Path(__file__).resolve().parent / "models" / "hl_multihead_16_benchmark.pt"
    model = TrainTorchModel.load_model(model_path)
    features = getattr(model, "checkpoint_meta", {}).get("features")
    dataset = HLDecisionDataset(build_cache=False, use_cache=True, features=features)
    evaluate_split(model, dataset)
