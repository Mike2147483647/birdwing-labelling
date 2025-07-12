import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn

from birdwinglabel.dataprocessing import createtorchdataset, data
from birdwinglabel.dataprocessing.data import full_bilateral_markers

from birdwinglabel.common import trainandtest

from birdwinglabel.Transformers.basic import DecoderOnlyTransformer


# create training and test datasets
seqID_list = data.get_list_of_seqID(full_bilateral_markers)
print(seqID_list[0:50])
train_pd_dataframe = (
    pd.DataFrame(full_bilateral_markers)
    .pipe(data.subset_by_seqID, seqID_list[0:200])
    .pipe(data.create_training)
)
test_pd_dataframe = (
    pd.DataFrame(full_bilateral_markers)
    .pipe(data.subset_by_seqID, seqID_list[200:300])
    .pipe(data.create_training)
)
print(f'{train_pd_dataframe.info()}')

# prepare the torch Datasets from pd dataframes
train_Dataset = createtorchdataset.HotMarkerDataset(train_pd_dataframe,8)
test_Dataset = createtorchdataset.HotMarkerDataset(test_pd_dataframe,8)

# put Datasets into DataLoader objects
batch_size = 50
train_dataloader = DataLoader(train_Dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_Dataset, batch_size=batch_size)

# traning the model
model = DecoderOnlyTransformer(embed_dim=32, num_heads=8, mlp_dim=128, num_layers=12)

loss = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters())
trainandtest.trainandtest(loss, optim, model, train_dataloader, test_dataloader, epochs=8)
