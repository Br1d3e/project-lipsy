import torch
from torch.utils.data.dataset import Dataset


class TorchDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


class HLMultiTaskDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = {
            key: torch.as_tensor(val, dtype=torch.float32)
            for key, val in Y.items()
        }
        for key, value in self.Y.items():
            if len(value) != len(self.X):
                raise ValueError(
                    f"Target {key!r} has {len(value)} samples, expected {len(self.X)}."
                )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], {
            key: val[i] for key, val in self.Y.items()
        }
