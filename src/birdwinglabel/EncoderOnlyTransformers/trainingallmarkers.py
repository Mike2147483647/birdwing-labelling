import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path

from birdwinglabel.dataprocessing import data
from birdwinglabel.dataprocessing.data import full_bilateral_markers

from birdwinglabel.common import trainandtest, prepforML, createtorchdataset

from birdwinglabel.EncoderOnlyTransformers.basic import IndptLabellingTransformer


# load labelled dataset
pkl_path = Path(__file__).parent.parent / 'dataprocessing' / 'labelled_df.pkl'
labelled_df = pd.read_pickle(pkl_path)

# rename to fit in createtorchdataset
labelled_df.rename(columns={labelled_df.columns[1]: 'markers_matrix', labelled_df.columns[2]: 'label'}, inplace=True)

# sanity check
# print(f'1 Nones in df: {labelled_df.isnull().sum().sum()}')
# print(f'sampled matrix: {labelled_df.iloc[2,1]} \ntype of matrix: {type(labelled_df.iloc[2,1])}')


# set rng for simulating zeros, subsetting data, etc


# set some of the coords matrices, some of their first 8 rows to 0 to simulate missing markers
# def simmissing_row(data_point, seed = 1):
#     RNG = np.random.default_rng(seed=seed)
#     num_of_rows = data_point['markers_matrix'].shape[0]
#     marker_to_zero = RNG.integers(0,num_of_rows,size=1)
#     data_point['markers_matrix'][marker_to_zero] = np.zeros_like(data_point['markers_matrix'][marker_to_zero])
#     data_point['label'][marker_to_zero] = 0
#     return data_point
#
# def simulate_missing(df, portion: float= 0.1, seed = 1):
#     RNG = np.random.default_rng(seed=seed)
#     num_of_data = len(df)
#     rows_to_set_zero = RNG.choice(range(0, num_of_data), size=int(np.floor(num_of_data * portion)),
#                                   replace=False)
#     df.loc[rows_to_set_zero] = df.loc[rows_to_set_zero].apply(simmissing_row, axis=1)
#     return df
# moved to common/prepforML.py

labelled_df = prepforML.simulate_missing(labelled_df, 0.1, 1)

# print(f'labelled_df.iloc[1,1]: \n{labelled_df.iloc[1,1]} \n{type(labelled_df.iloc[1,1])} \n{labelled_df.iloc[1,2]}')
# print(f'2 Nones in df: {labelled_df.isnull().sum().sum()}')


# subset to training and test
seqID_list = data.get_list_of_seqID(full_bilateral_markers)
def get_frameID(idx):

    # Convert seqID_list to a numpy array for faster indexing
    seqID_array = np.array(seqID_list)
    selected_seqIDs = seqID_array[idx]

    # Filter full_bilateral_markers directly using numpy/pandas
    subset_markers = full_bilateral_markers[full_bilateral_markers['seqID'].isin(selected_seqIDs)]

    # Extract frame IDs directly
    frameIDs = data.get_list_of_frameID(subset_markers)
    return frameIDs

# choose sets by index in seqID
train_idx = range(0,200)
test_idx = range(200,250)
train_set = data.subset_by_frameID(labelled_df, get_frameID(train_idx))
test_set = data.subset_by_frameID(labelled_df, get_frameID(test_idx))

# print(f'train set sample: \n{train_set.iloc[2,1]} \nlabel: {train_set.iloc[2,2]}'
#       f' \n type: \n{type(train_set.iloc[2,1])}')
# print(f'Nones in df: {labelled_df.isnull().sum().sum()}')
# randomize the rows of coords matrix and labels of each data point
# def permute(data_point, seed = 1):
#     RNG = np.random.default_rng(seed=seed)
#     num_of_rows = data_point.iloc[1].shape[0]
#     perm = RNG.permutation(num_of_rows)
#     data_point.iloc[1] = data_point.iloc[1][perm]
#     data_point.iloc[2] = data_point.iloc[2][perm]
#     return data_point
#
# def permute_df(df, seed = 1):
#     df = df.apply(permute, axis = 1)
#     return df
# moved to common/prepforML.py

train_set = prepforML.permute_df(train_set)
test_set = prepforML.permute_df(test_set)

# print(f'train set sample: \n{train_set.iloc[2,1]} \n{train_set.iloc[2,2]}')

# pad the matrices to have 32 rows
# def padding(x, final_length = 32):
#     if x.ndim == 1:
#         # Pad 1D vector
#         pad_width = (0, final_length - len(x))
#         return np.pad(x, pad_width, mode='constant')
#     elif x.ndim == 2:
#         # Pad 2D matrix (pad rows)
#         pad_width = ((0, final_length - x.shape[0]), (0, 0))
#         return np.pad(x, pad_width, mode='constant')
#     else:
#         return x  # unchanged if not 1D or 2D
# moved to common

for i in range(1,3):
    train_set.iloc[:,i] = train_set.iloc[:,i].apply(prepforML.padding)
    test_set.iloc[:, i] = test_set.iloc[:, i].apply(prepforML.padding)

# print(f'train set sample: \n{train_set.iloc[2,1]} \n{train_set.iloc[2,2]}')



# # create training and test datasets
# seqID_list = data.get_list_of_seqID(full_bilateral_markers)
# # print(seqID_list[0:50])
# train_pd_dataframe = (
#     pd.DataFrame(full_bilateral_markers)
#     .pipe(data.subset_by_seqID, seqID_list[0:200])
#     .pipe(data.create_training)
# )
# test_pd_dataframe = (
#     pd.DataFrame(full_bilateral_markers)
#     .pipe(data.subset_by_seqID, seqID_list[200:300])
#     .pipe(data.create_training)
# )
# # print(f'{train_pd_dataframe.info()}')
#
# prepare the torch Datasets from pd dataframes
train_Dataset = createtorchdataset.HotMarkerDataset(train_set, 9)
test_Dataset = createtorchdataset.HotMarkerDataset(test_set, 9)

# put Datasets into DataLoader objects
batch_size = 50
train_dataloader = DataLoader(train_Dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_Dataset, batch_size=batch_size)

# traning the model
model = IndptLabellingTransformer(embed_dim=32, num_heads=8, mlp_dim=128, num_layers=3, seq_len=32, num_class=9)

loss = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters())
trainandtest.trainandtest(loss, optim, model, train_dataloader, test_dataloader, epochs=8)