import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn

from birdwinglabel.dataprocessing import createtorchdataset, data
from birdwinglabel.dataprocessing.data import full_bilateral_markers

from birdwinglabel.common import trainandtest

# find device to train nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# create training and test datasets
seqID_list = data.get_list_of_seqID(full_bilateral_markers)
# print(seqID_list[0:50])
train_pd_dataframe = (
    pd.DataFrame(full_bilateral_markers)
    .pipe(data.subset_by_seqID, seqID_list[0:100])
    .pipe(data.create_training)
)
test_pd_dataframe = (
    pd.DataFrame(full_bilateral_markers)
    .pipe(data.subset_by_seqID, seqID_list[200:210])
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

# create model1, MLP1
class MLP1(nn.Module):
    # adapted from https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#hyperparameters 20250703_1745
    def __init__(self, num_layers, num_hidden_feat):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        layers.append(nn.Linear(8*3, num_hidden_feat))
        layers.append(nn.ReLU())
        for i in range(num_layers-1):
            layers.append(nn.Linear(num_hidden_feat,num_hidden_feat))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_hidden_feat, 8*8))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        logits = logits.view(-1, 8, 8)
        return logits


# create instance of MLP1
mlp1 = MLP1(num_layers=6,num_hidden_feat=128)

# feed into automated train and test function
loss = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(mlp1.parameters())
trainandtest.trainandtest(loss, optim, mlp1, train_dataloader, test_dataloader, epochs=10)
