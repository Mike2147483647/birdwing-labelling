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



