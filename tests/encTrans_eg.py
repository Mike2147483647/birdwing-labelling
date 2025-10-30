import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
import random

from birdwinglabel.common_utils import simulate_missing, permute_df, padding, trainandtest
from birdwinglabel.dataclasses import EncMarkerDataset
from birdwinglabel.model import EncTransformer

# load data
pkl_path = Path(__file__).parent.parent / 'src'/'birdwinglabel'/'dataprocessing'/'labelled_df.pkl'
labelled_df = pd.read_pickle(pkl_path)

# choose subset of data to train on
random.seed(1)
seqID_subset_idx = random.sample(range(len(labelled_df)), 250)
train_idx = seqID_subset_idx[:200]
test_idx = seqID_subset_idx[200:]

train_df = labelled_df.iloc[train_idx]
test_df = labelled_df.iloc[test_idx]

# take a look at the train_df
def train_df_sample(msg):
    print(msg)
    global train_df
    print(f'''
    {train_df.info()}
    train_df sample frameID: {train_df.iloc[0,0]}
    train_df sample rot_xyz: 
    {train_df.iloc[0,1]}
    train_df sample labels: 
    {train_df.iloc[0,2]}
    ''')

train_df_sample('1st')

# preprocessing
def preprocessing_df(df):
    df = simulate_missing(df, 0.1,1)
    df = permute_df(df, 1)
    df[['rot_xyz', 'rot_xyz_mask']] = df['rot_xyz'].apply(lambda x: pd.Series(padding(x, final_length=32)))
    df[['labels', 'labels_mask']] = df['labels'].apply(lambda x: pd.Series(padding(x, final_length=32)))
    return df

train_df = preprocessing_df(train_df)
test_df = preprocessing_df(test_df)
train_df_sample('2nd')

# put into dataloader
train_dataset = EncMarkerDataset(train_df, 9)
test_dataset = EncMarkerDataset(test_df, 9)

train_dataloader = DataLoader(train_dataset,batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=10, shuffle=True)

# train model
model = EncTransformer(embed_dim=32,num_heads=8,mlp_dim=128,num_layers=1,seq_len=32,dropout=0.1,num_class=9)
loss = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.AdamW(model.parameters())
trainandtest(loss_fn=loss, optimizer=optim,model=model,train_dataloader=train_dataloader,test_dataloader=test_dataloader,epochs=1)


