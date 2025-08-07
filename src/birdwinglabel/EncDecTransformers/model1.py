import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

import birdwinglabel.dataprocessing.data as data
from birdwinglabel.common import prepforML, createtorchdataset
from birdwinglabel.common.createtorchdataset import MarkerCoordsDataset
from birdwinglabel.common.prepforML import simulate_missing
from birdwinglabel.EncDecTransformers.factories import LinearPosEnc, FrequentialPosEnc, IdentifyMarkerTransformer

# import datasets
df_dir = Path(__file__).parent.parent / "dataprocessing"

src_df = pd.read_pickle(df_dir / "src_df.pkl")
tgt_df = pd.read_pickle(df_dir / "tgt_df.pkl")

# subset to train_src, train_tgt, test_src
seqID = data.get_list_of_seqID(src_df)
train_seqs = [seqID[i] for i in range(0,10)]    # choose here
test_seqs = [seqID[i] for i in range(10,15)]

train_src = data.subset_by_seqID(src_df, train_seqs)
test_src = data.subset_by_seqID(src_df, test_seqs)
train_tgt = data.subset_by_seqID(tgt_df, train_seqs)
test_tgt = data.subset_by_seqID(tgt_df, test_seqs)


# permute train_src, train_tgt, test_src and remove some of the entries in src
def preparation(*dfs):

    def process(df):
        df = prepforML.permute_df(df)
        df = df.copy()
        df['frameID'] = df['frameID'].str.split('_').str[-1].astype(int)
        return df

    return tuple(process(df) for df in dfs)

train_src, train_tgt, test_src, test_tgt = preparation(train_src, train_tgt, test_src, test_tgt)

train_src = simulate_missing(train_src)
test_src = test_src.copy()
test_src = simulate_missing(test_src)

# stack by seqID, each row is [frameID, rot_xyz] or equiv [frameID, num_marker, 3]
# here num_data = number of seq, seq_length = number of frames, feat_dim = nun_marker x 3

def as_seqn_tensor(df, max_marker: int = 32, frame_count: int = 500):
    # df has cols: frameID, rot_xyz, (label,) seqID
    results = []
    for seqID, group in df.groupby('seqID'):
        # get all frameID and find min max
        frame_ids = group['frameID'].values
        min_frame, max_frame = frame_ids.min(), frame_ids.max()
        num_frames = max_frame - min_frame + 1

        # pad rot_xyz to be [max_marker, 3] for all frames
        group['rot_xyz'] = group['rot_xyz'].apply(lambda x: prepforML.padding(x, max_marker))

        # Prepare storage
        rot_xyz_tensor = np.zeros((frame_count, max_marker, 3), dtype=np.float32)
        padding_mask = np.ones(frame_count, dtype=bool)  # True = padding

        # Fill in available frames
        frameid_to_idx = {fid: i for i, fid in enumerate(range(min_frame, max_frame + 1))}
        for _, row in group.iterrows():
            idx = frameid_to_idx[row['frameID']]
            rot_xyz_tensor[idx] = row['rot_xyz']
            padding_mask[idx] = False  # Not padding

        results.append({
            'seqID': seqID,
            'rot_xyz_tensor': rot_xyz_tensor,
            'padding_mask': padding_mask
        })

    # output df with cols: seqID, rot_xyz tensor [frame_count, max_marker, 3], padding mask [frame_count]
    return pd.DataFrame(results)


print(f'before: {train_tgt.iloc[0,1]}')

train_src = as_seqn_tensor(train_src)
train_tgt = as_seqn_tensor(train_tgt)
train_dataset = MarkerCoordsDataset(train_src, train_tgt)
train_dataloader = DataLoader(train_dataset, batch_size = 10)

pos_enc = LinearPosEnc()
model = IdentifyMarkerTransformer(pos_enc=pos_enc, max_marker=32, frame_count=500)
for _, (src, tgt, src_pad_mask, tgt_pad_mask) in enumerate(train_dataloader):
    model(src, tgt, src_pad_mask, tgt_pad_mask)














