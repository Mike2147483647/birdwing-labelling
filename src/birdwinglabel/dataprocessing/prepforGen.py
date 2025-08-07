import pandas as pd
import numpy as np


# load labelled_df (and gold)
labelled_df = pd.read_pickle("labelled_df.pkl")
gold_df = pd.read_pickle("gold_df.pkl")

print(f"sample of labelled: \n{labelled_df.iloc[0:5,0]}")
print(f"sample of gold: \n{gold_df.iloc[0:5,[0,1]]}")

# combine labelled and gold to make labelled have seqID
# create mapping from frameID to seqID using gold_df
frame_to_seq = dict(zip(gold_df['frameID'], gold_df['seqID']))

# map seqID to labelled_df using frameID
labelled_df['seqID'] = labelled_df['frameID'].map(frame_to_seq)

print(labelled_df.info())
print(f"sample of labelled: \n{labelled_df.iloc[0:5,[0,3]]}")
print(f"sample of labelled rot_xyz and label: \n{labelled_df.iloc[0,[1,2]]}")

# src to encoder (eliminate some entries later like EncOnly)
src_df = labelled_df
src_df.to_pickle("src_df.pkl")

# tgt to decoder (actually same as gold_df but just to be sure they are from the same source)
def remove_non_markers(data_point):
    mask = data_point['labels'] != 0
    data_point['labels'] = data_point['labels'][mask]
    data_point['rot_xyz'] = data_point['rot_xyz'][mask]
    return data_point
tgt_df = labelled_df.apply(remove_non_markers, axis=1)
print(f"sample of tgt_df: \n{tgt_df.iloc[0,[1,2]]}")
tgt_df.to_pickle("tgt_df.pkl")

print(f"src_df.info \n{src_df.info()}")
print(f"tgt_df.info \n{tgt_df.info()}")

# make labelled into [seqID, frameID, num_marker, 3] maybe in common and use in EncDec
# [seqID, frameID, num_mark, 3] -> [batch, time, num_marker x 3]
