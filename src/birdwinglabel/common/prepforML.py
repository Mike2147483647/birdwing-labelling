import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn



# pad the matrices to have 32 rows, apply it over col of matrix
def padding(x, final_length = 32):
    if x.ndim == 1:
        padded = np.zeros(final_length, dtype=x.dtype)
        mask = np.ones(final_length, dtype=np.int32)
        length = min(len(x), final_length)
        padded[:length] = x[:length]
        mask[:length] = 0
        return padded, mask
    elif x.ndim == 2:
        padded = np.zeros((final_length, x.shape[1]), dtype=x.dtype)
        mask = np.ones(final_length, dtype=np.int32)
        length = min(x.shape[0], final_length)
        padded[:length, :] = x[:length, :]
        mask[:length] = 0
        return padded, mask
    else:
        raise ValueError("Input must be 1D or 2D numpy array")

# Unpad rows, use with .apply
def unpad_row(row, pad_value=0):
    rot_xyz = np.array(row['rot_xyz'])
    labels = np.array(row['labels'])
    mask = ~(np.all(rot_xyz == pad_value, axis=1))
    return rot_xyz[mask], labels[mask]


# simulate missing markers by removing rows and its corresponding labels randomly
# simmissing_row def in global for higher customisability
def simmissing_row(data_point, seed = 1):
    RNG = np.random.default_rng(seed=seed)
    num_of_rows = data_point.iloc[1].shape[0]
    num_to_zero = RNG.integers(1, num_of_rows + 1)  # random number between 1 and num_of_rows
    marker_to_zero = RNG.choice(num_of_rows, size=num_to_zero, replace=False)
    data_point.iloc[1][marker_to_zero] = np.zeros_like(data_point.iloc[1][marker_to_zero])
    data_point.iloc[2][marker_to_zero] = 0
    return data_point

def simulate_missing(df, portion: float= 0.1, seed = 1):
    df = df.reset_index(drop=True)
    RNG = np.random.default_rng(seed=seed)
    num_of_data = len(df)
    rows_to_set_zero = RNG.choice(range(0, num_of_data), size=int(np.floor(num_of_data * portion)),
                                  replace=False)
    df.loc[rows_to_set_zero] = df.loc[rows_to_set_zero].apply(simmissing_row, axis=1)
    return df

# simulate missing marker for *a* rot_xyz matrix only
def simmissing_marker(coord_matrix, seed = 1):
    RNG = np.random.default_rng(seed=seed)
    num_marker = coord_matrix.shape[0]
    num_to_remain = RNG.integers(1, num_marker+1)
    marker_to_remain = RNG.choice(num_marker, size=num_to_remain, replace=False)
    coord_matrix = coord_matrix[marker_to_remain]
    return coord_matrix

# randomize the rows of coords matrix and labels of each data point
def permute(data_point, seed = 1):
    # col1 must be list of matrices, col2 must be list of labels
    RNG = np.random.default_rng(seed=seed)
    if len(data_point) == 2:
        # Two columns: permute the second column only
        num_of_rows = data_point.iloc[1].shape[0]
        perm = RNG.permutation(num_of_rows)
        data_point.iloc[1] = data_point.iloc[1][perm]
    elif len(data_point) >= 3:
        # Three columns: permute both second and third columns
        num_of_rows = data_point.iloc[1].shape[0]
        perm = RNG.permutation(num_of_rows)
        data_point.iloc[1] = data_point.iloc[1][perm]
        data_point.iloc[2] = data_point.iloc[2][perm]
    return data_point

def permute_df(df, seed = 1):
    df = df.apply(permute, seed=seed, axis = 1)
    return df