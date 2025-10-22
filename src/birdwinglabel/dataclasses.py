import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from common_utils import permute_df, padding, simmissing_marker


# create class to turn dataset into torch DataLoader
# depreciated since crossentropy loss is not working
# # use this with crossentropyloss
# class MarkerDataset(Dataset):
#     def __init__(self, dataframe):
#         self.data = dataframe
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         markers = torch.tensor(self.data.iloc[idx]['markers_matrix'], dtype=torch.float32)
#         label = self.data.iloc[idx]['label']
#         label = torch.tensor(label, dtype=torch.long).squeeze()
#         return markers, label


# use this with BCElogitsloss
class EncMarkerDataset(Dataset):
    # formerly HotMarkerDataset
    def __init__(self, dataframe, num_class):
        self.data = dataframe
        self.num_class = num_class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        markers = torch.tensor(self.data.iloc[idx]['rot_xyz'], dtype=torch.float32)
        label = self.data.iloc[idx]['label']
        label = torch.tensor(label, dtype=torch.long).squeeze()
        # One-hot encode: shape [8, 8]
        label = F.one_hot(label, num_classes=self.num_class).float()
        return markers, label



# Autoencoders
class AutoMarkerDataset(Dataset):
    # formerly MarkerTimeIndptDataset
    def __init__(self, src_df, tgt_df, noise:bool, pred:bool = False):
        '''
        src_df: [num_frames, max_marker, 3]
        tgt_df: [num_frames, 8, 3]
        extra markers are currently not supported
        '''
        self.noise = noise
        self.pred = pred
        self.src_df = torch.stack([torch.tensor(x, dtype=torch.float32) for x in src_df['rot_xyz']])
        # add try except for detecting colname 'rot_xyz_mask'
        self.src_mask = torch.stack([torch.tensor(x, dtype=torch.bool) for x in src_df['rot_xyz_mask']])
        if pred is False:
            self.gold_df = torch.stack([torch.tensor(x, dtype=torch.float32) for x in tgt_df['rot_xyz']])
        else:
            self.gold_df = None
        tgt_df_copy = tgt_df
        tgt_df_copy.loc[:, 'rot_xyz'] = tgt_df_copy['rot_xyz'].apply(simmissing_marker)
        tgt_df_copy = permute_df(tgt_df_copy)
        tgt_df_copy.loc[:, ['rot_xyz', 'rot_xyz_mask']] = tgt_df_copy['rot_xyz'].apply(
            lambda x: pd.Series(padding(x, 8), index=['rot_xyz', 'rot_xyz_mask']))
        self.tgt_df = torch.stack([torch.tensor(x, dtype=torch.float32) for x in tgt_df_copy['rot_xyz']])
        self.tgt_mask = torch.stack([torch.tensor(x, dtype=torch.bool) for x in tgt_df_copy['rot_xyz_mask']])
        if noise:
            noise_add = torch.ones_like(self.tgt_df)
            self.tgt_df = self.tgt_df + noise_add * self.tgt_mask.unsqueeze(-1)
        if pred:
            tgt_df.loc[:, ['rot_xyz', 'rot_xyz_mask']] = tgt_df['rot_xyz'].apply(
                lambda x: pd.Series(padding(x, 8), index=['rot_xyz', 'rot_xyz_mask']))
            self.tgt_df = torch.stack([torch.tensor(x, dtype=torch.float32) for x in tgt_df['rot_xyz']])
            self.tgt_mask = torch.stack([torch.tensor(x, dtype=torch.bool) for x in tgt_df['rot_xyz_mask']])



    def __len__(self):
        return len(self.tgt_df)

    def __getitem__(self, idx):
        if self.pred is False:
            return self.src_df[idx], self.tgt_df[idx], self.src_mask[idx], self.tgt_mask[idx], self.gold_df[idx]
        else:
            return self.src_df[idx], self.tgt_df[idx], self.src_mask[idx], self.tgt_mask[idx]

