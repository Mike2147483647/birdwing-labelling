import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn



# pad the matrices to have 32 rows, apply it over col of matrix
def padding(x, final_length = 32):
    if x.ndim == 1:
        # Pad 1D vector
        pad_width = (0, final_length - len(x))
        return np.pad(x, pad_width, mode='constant')
    elif x.ndim == 2:
        # Pad 2D matrix (pad rows)
        pad_width = ((0, final_length - x.shape[0]), (0, 0))
        return np.pad(x, pad_width, mode='constant')
    else:
        return x  # unchanged if not 1D or 2D


# simulate missing markers by removing rows and its corresponding labels randomly
# simmissing_row def in global for higher customisability
def simmissing_row(data_point, seed = 1):
    RNG = np.random.default_rng(seed=seed)
    num_of_rows = data_point['markers_matrix'].shape[0]
    marker_to_zero = RNG.integers(0,num_of_rows,size=1)
    data_point['markers_matrix'][marker_to_zero] = np.zeros_like(data_point['markers_matrix'][marker_to_zero])
    data_point['label'][marker_to_zero] = 0
    return data_point

def simulate_missing(df, portion: float= 0.1, seed = 1):
    RNG = np.random.default_rng(seed=seed)
    num_of_data = len(df)
    rows_to_set_zero = RNG.choice(range(0, num_of_data), size=int(np.floor(num_of_data * portion)),
                                  replace=False)
    df.loc[rows_to_set_zero] = df.loc[rows_to_set_zero].apply(simmissing_row, axis=1)
    return df


# randomize the rows of coords matrix and labels of each data point
def permute(data_point, seed = 1):
    # col1 must be list of matrices, col2 must be list of labels
    RNG = np.random.default_rng(seed=seed)
    num_of_rows = data_point.iloc[1].shape[0]
    perm = RNG.permutation(num_of_rows)
    data_point.iloc[1] = data_point.iloc[1][perm]
    data_point.iloc[2] = data_point.iloc[2][perm]
    return data_point

def permute_df(df, seed = 1):
    df = df.apply(permute, seed=seed, axis = 1)
    return df