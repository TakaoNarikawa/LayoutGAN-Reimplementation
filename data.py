import numpy as np
import torch
from torch.utils.data import Dataset


class MnistLayoutDataset(Dataset):
    def __init__(self, npx_path) -> None:
        super().__init__()
        self.data = np.load(npx_path)

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        data = data * 28.0 / 27.0 
        # data: (128, 2)
        data = data.permute(1, 0)
        # data: (2, 128)
        return data

    def __len__(self):
        return len(self.data)

class PubLayNetDataset(Dataset):
    def __init__(self, npx_path) -> None:
        super().__init__()
        self.data = np.load(npx_path)

    def __getitem__(self, index):
        batch = torch.Tensor(self.data[index])
        # data: (NUM, DIM+CLS)
        batch = batch.permute(1, 0)
        # data: (DIM+CLS, NUM)
        return batch

    def __len__(self):
        return len(self.data)
