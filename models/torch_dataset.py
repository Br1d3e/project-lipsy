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
