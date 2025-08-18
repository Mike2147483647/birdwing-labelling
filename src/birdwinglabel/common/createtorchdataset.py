import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from birdwinglabel.common.prepforML import permute_df, padding, simmissing_marker


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
class MarkerTimeIndptDataset(Dataset):
    def __init__(self, src_df, tgt_df, noise:bool):
        '''
        src_df: [num_frames, max_marker, 3]
        tgt_df: [num_frames, 8, 3]
        extra markers are currently not supported
        '''
        self.noise = noise
        self.src_df = torch.stack([torch.tensor(x, dtype=torch.float32) for x in src_df['rot_xyz']])
        self.src_mask = torch.stack([torch.tensor(x, dtype=torch.bool) for x in src_df['rot_xyz_mask']])
        self.gold_df = torch.stack([torch.tensor(x, dtype=torch.float32) for x in tgt_df['rot_xyz']])
        tgt_df_copy = tgt_df
        tgt_df_copy.loc[:, 'rot_xyz'] = tgt_df_copy['rot_xyz'].apply(simmissing_marker)
        tgt_df_copy.loc[:, ['rot_xyz', 'rot_xyz_mask']] = tgt_df_copy['rot_xyz'].apply(
            lambda x: pd.Series(padding(x, 8), index=['rot_xyz', 'rot_xyz_mask']))
        tgt_df_copy = permute_df(tgt_df_copy)
        self.tgt_df = torch.stack([torch.tensor(x, dtype=torch.float32) for x in tgt_df_copy['rot_xyz']])
        self.tgt_mask = torch.stack([torch.tensor(x, dtype=torch.bool) for x in tgt_df_copy['rot_xyz_mask']])
        if noise:
            noise_add = torch.ones_like(self.tgt_df)
            self.tgt_df = self.tgt_df + noise_add * self.tgt_mask.unsqueeze(-1)

    def __len__(self):
        return len(self.tgt_df)

    def __getitem__(self, idx):
        return self.src_df[idx], self.tgt_df[idx], self.src_mask[idx], self.tgt_mask[idx], self.gold_df[idx]






def exp_nonzero_eps(tensor, epsilon=1e-6):
    mask = tensor.abs() > epsilon
    tensor[mask] = torch.exp(tensor[mask])
    tensor[~mask] = 0
    return tensor

class MarkerTimeDptDataset_train(Dataset):
    def __init__(self, src_df, tgt_df, exp:bool = False):
        # src_df and tgt_df must have cols seqID, rot_xyz tensor [frame_count, max_marker, 3], padding mask [frame_count]
        # check num_seq of src is same as num_seq of tgt
        assert len(src_df) == len(tgt_df)
        # [batch, data_tensor], batch = num_seq, data_tensor [frame, max_marker, 3]
        self.src = [torch.tensor(x, dtype=torch.float32) for x in src_df['rot_xyz_tensor']]
        self.tgt = [torch.tensor(x, dtype=torch.float32) for x in tgt_df['rot_xyz_tensor']]
        self.src_key_padding_mask = [torch.tensor(x, dtype=torch.bool) for x in src_df['padding_mask']]
        self.tgt_key_padding_mask = [torch.tensor(x, dtype=torch.bool) for x in tgt_df['padding_mask']]

        self.exp_trans = exp
        if self.exp_trans:
            self.src = [exp_nonzero_eps(x) for x in self.src]
            self.tgt = [exp_nonzero_eps(x) for x in self.tgt]



    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.src_key_padding_mask[idx], self.tgt_key_padding_mask[idx]


class MarkerTimeDptDataset_test(Dataset):
    def __init__(self, src_df, seed_df, gold_df, exp:bool = False):
        # src_df and gold_df must have cols seqID, rot_xyz tensor [frame_count, max_marker, 3], padding mask [frame_count]
        # seed_df must have cols seqID, rot_xyz matrix [num_label, 3], num_labels = 8 for now
        # seed_df must have same number of seq as src_df
        assert len(src_df) == len(seed_df) == len(gold_df)
        # [batch, data_tensor], batch = num_seq, data_tensor [frame, max_marker, 3]
        self.src = [torch.tensor(x, dtype=torch.float32) for x in src_df['rot_xyz_tensor']]
        self.src_key_padding_mask = [torch.tensor(x, dtype=torch.bool) for x in src_df['padding_mask']]
        self.seed = [torch.tensor(x, dtype=torch.float32) for x in seed_df['rot_xyz']]
        self.gold = [torch.tensor(x, dtype=torch.float32) for x in gold_df['rot_xyz_tensor']]
        self.gold_key_padding_mask = [torch.tensor(x, dtype=torch.bool) for x in gold_df['padding_mask']]

        for i in range(len(self.seed)):
            seed_arr = self.seed[i].cpu().numpy()
            gold_first_frame = self.gold[i][0, :8, :].cpu().numpy()
            if not np.allclose(seed_arr, gold_first_frame, atol=1e-6):
                print(
                    f"Warning: seed and gold first frame differ for sequence {i} (max diff: {np.abs(seed_arr - gold_first_frame).max()})")

        self.exp_trans = exp
        if self.exp_trans:
            self.src = [exp_nonzero_eps(x) for x in self.src]
            self.seed = [exp_nonzero_eps(x) for x in self.seed]




    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.src_key_padding_mask[idx], self.seed[idx], self.gold[idx], self.gold_key_padding_mask[idx]





















