from torch.utils.data import Dataset
import torch

class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows
        self.X = [torch.tensor(x, dtype=torch.float32) for x, y in self.windows]
        self.Y = [torch.tensor(y, dtype=torch.float32) for x, y in self.windows]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]