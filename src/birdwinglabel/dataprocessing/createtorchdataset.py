import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# create class to turn dataset into torch DataLoader
# use this with crossentropyloss
class MarkerDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        markers = torch.tensor(self.data.iloc[idx]['markers_matrix'], dtype=torch.float32)
        label = self.data.iloc[idx]['label']
        label = torch.tensor(label, dtype=torch.long).squeeze()
        return markers, label


# use this with BCElogitsloss
class HotMarkerDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        markers = torch.tensor(self.data.iloc[idx]['markers_matrix'], dtype=torch.float32)
        label = self.data.iloc[idx]['label']
        label = torch.tensor(label, dtype=torch.long).squeeze()
        # One-hot encode: shape [8, 8]
        label = F.one_hot(label, num_classes=8).float()
        return markers, label