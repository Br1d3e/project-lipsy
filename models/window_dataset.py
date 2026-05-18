from torch.utils.data import Dataset
import torch


class WindowTensorView:
    def __init__(self, windows, column):
        self.windows = windows
        self.column = column

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        return torch.as_tensor(self.windows[i][self.column], dtype=torch.float32)


class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows
        self.X = WindowTensorView(windows, 0)
        self.Y = WindowTensorView(windows, 1)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
