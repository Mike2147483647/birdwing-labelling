import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



# create class to turn dataset into torch DataLoader
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