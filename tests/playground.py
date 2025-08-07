import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

import birdwinglabel.dataprocessing.data as data
from birdwinglabel.common import prepforML, createtorchdataset
from birdwinglabel.common.prepforML import simulate_missing

# import datasets
df_dir = Path(__file__).parent.parent / "src" / "birdwinglabel" / "dataprocessing"

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


tgt_df, = preparation(tgt_df)

def find_seq_length(df):
    num_seq = df['seqID'].unique()
    seq_size = np.zeros(len(num_seq), dtype=int)
    for i, seqID in enumerate(num_seq):
        group = df[df['seqID'] == seqID]
        frame_ids = group['frameID'].values
        min_frame, max_frame = frame_ids.min(), frame_ids.max()
        seq_size[i] = max_frame - min_frame + 1
    return seq_size

seq_lengths = find_seq_length(tgt_df)
print(f'max seq len: {seq_lengths.max()}')

