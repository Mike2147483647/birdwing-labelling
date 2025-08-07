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
    def __init__(self, dataframe, num_class):
        self.data = dataframe
        self.num_class = num_class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        markers = torch.tensor(self.data.iloc[idx]['markers_matrix'], dtype=torch.float32)
        label = self.data.iloc[idx]['label']
        label = torch.tensor(label, dtype=torch.long).squeeze()
        # One-hot encode: shape [8, 8]
        label = F.one_hot(label, num_classes=self.num_class).float()
        return markers, label


# Autoencoders
class MarkerCoordsDataset(Dataset):
    def __init__(self, src_df, tgt_df):
        # src_df and tgt_df must have cols seqID, rot_xyz tensor [frame_count, max_marker, 3], padding mask [frame_count]
        # check num_seq of src is same as num_seq of tgt
        assert len(src_df) == len(tgt_df)
        # [batch, data_tensor], batch = num_seq, data_tensor [frame, max_marker, 3]
        self.src = [torch.tensor(x, dtype=torch.float32) for x in src_df['rot_xyz_tensor']]
        self.tgt = [torch.tensor(x, dtype=torch.float32) for x in tgt_df['rot_xyz_tensor']]
        self.src_key_padding_mask = [torch.tensor(x, dtype=torch.bool) for x in src_df['padding_mask']]
        self.tgt_key_padding_mask = [torch.tensor(x, dtype=torch.bool) for x in tgt_df['padding_mask']]


    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.src_key_padding_mask[idx], self.tgt_key_padding_mask[idx]


