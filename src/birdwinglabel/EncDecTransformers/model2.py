import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import random

import birdwinglabel.dataprocessing.data as data
from birdwinglabel.common import prepforML, createtorchdataset
from birdwinglabel.common.createtorchdataset import MarkerTimeDptDataset_train, MarkerTimeDptDataset_test, MarkerTimeIndptDataset
from birdwinglabel.common.prepforML import simulate_missing, permute_df, padding
from birdwinglabel.EncDecTransformers.factories import LinearPosEnc, FrequentialPosEnc, \
    IdentifyMarkerTimeDptTransformer, IdentifyMarkerTimeIndptTransformer
from birdwinglabel.common.trainandtest import trainandtest


# import datasets
df_dir = Path(__file__).parent.parent / "dataprocessing"

src_df = pd.read_pickle(df_dir / "src_df.pkl")
tgt_df = pd.read_pickle(df_dir / "tgt_df.pkl")

# subset to train_src, train_tgt, test_src
seqID = data.get_list_of_seqID(src_df)
random.seed(1)
sample_idx = random.sample( range(len(seqID)), 450 )
train_seqs = [seqID[i] for i in sample_idx[0:400]]    # choose here
test_seqs = [seqID[i] for i in sample_idx[400:450]]

train_src = data.subset_by_seqID(src_df, train_seqs)
test_src = data.subset_by_seqID(src_df, test_seqs)
train_tgt = data.subset_by_seqID(tgt_df, train_seqs)
test_tgt = data.subset_by_seqID(tgt_df, test_seqs)

# # Check for each row in train_tgt
# def check_first8labels(df):
#     mask = df['labels'].apply(lambda x: np.array_equal(np.array(x[:8]), np.arange(1, 9)))
#
#     # Print how many rows match
#     print("Rows with first 8 labels as 1-8:", mask.sum())
#     print("Total rows:", len(df))
#
# check_first8labels(src_df)

# simulated real data for src
max_length = 32
train_src = simulate_missing(permute_df(train_src))
test_src = simulate_missing(permute_df(test_src))

train_src[['rot_xyz', 'rot_xyz_mask']] = train_src['rot_xyz'].apply(lambda x: pd.Series(padding(x, max_length)))
test_src[['rot_xyz', 'rot_xyz_mask']] = test_src['rot_xyz'].apply(lambda x: pd.Series(padding(x, max_length)))

print(f'train_src: \n{train_src.iloc[0,1]} \n{train_src.iloc[0,2]}')
print(f'train_mask: {train_src.iloc[0,4]}')

# put into dataset, then dataloader

train_dataset = MarkerTimeIndptDataset(src_df=train_src, tgt_df=train_tgt, noise=True)
test_dataset = MarkerTimeIndptDataset(src_df=test_src, tgt_df=test_tgt, noise=True)

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=10)

# for batch, (src, tgt, src_mask, tgt_mask) in enumerate(train_dataloader):
#     print(f'src: \n{src[0]} \nsrc_mask: \n{src_mask[0]} \ntgt: \n{tgt[0]} \ntgt_mask: \n{tgt_mask[0]}')
#     input()

# for batch, (src, tgt, src_mask, tgt_mask, gold) in enumerate(test_dataloader):
#     print(f'src: \n{src[0]} \nsrc_mask: \n{src_mask[0]} \ntgt: \n{tgt[0]} \ntgt_mask: \n{tgt_mask[0]} \ngold: {gold[0]}')
#     print(f'shape of gold: {gold.shape}')
#     input()

# model


model = IdentifyMarkerTimeIndptTransformer(embed_dim=32, num_head=8 , num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=128)
loss = nn.L1Loss()
optim = torch.optim.AdamW(model.parameters())
trainandtest(loss_fn=loss, optimizer=optim, model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, epochs=50)













