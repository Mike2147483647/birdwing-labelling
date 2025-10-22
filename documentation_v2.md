<!-- TOC -->
  * [Introduction](#introduction-)
  * [common_utils](#common_utils)
    * [padding](#padding)
    * [unpad_row](#unpad_row)
    * [simmissing_row](#simmissing_row)
    * [simulate_missing](#simulate_missing)
    * [simmissing_marker](#simmissing_marker)
    * [permute](#permute)
    * [permute_df](#permute_df)
  * [dataclasses](#dataclasses)
    * [EncMarkerDataset](#encmarkerdataset)
    * [AutoMarkerDataset](#automarkerdataset)
<!-- TOC -->


## Introduction 
Brief overview of the project goes here. 

The labelled dataframes are expected to contain columns 'seqID', 'frameID', 
and 24 columns of real numbers with colnames containing 'rot_xyz' for every row. 
The unlabelled dataframes are expected to contain columns 'seqID', 'frameID', 
and 3 columns of real numbers with colnames containing 'rot_xyz' for every row. 

For now, this package expects the user to stack the rows which share the same frameID themselves.




## common_utils
Common functions to pre/post-process the datasets for transformer I/O.


### padding
`padding(x, final_length = 32)`

Pads a np array (1D or 2D only) to final_length.

`x`: 1D or 2D np array of floats with dimension (seq_len, ...) \
`final_length`: default = 32, the outmost dimension will be padded to this number \
return: \
`padded`: np array with dimension (final_length, ...) \
`mask`: padding mask with dimension (final_length), where 1 indicates the row is padded


### unpad_row
`unpad_row(row, pad_value=0)`

Unpads a row of a pd.Dataframe with np arrays in each row by detecting which row of the rot_xyz matrix is all 0s.
Is used with pd.Dataframe.apply 

`row`: a row of a pd.Dataframe with columns named 'rot_xyz' and 'labels' \
`pad_value`: the value the functions checks on to identify whether the row is padded \


### simmissing_row
`simmissing_row(data_point, seed = 1)`

Simulates missing markers for a singular data point. It first randomly generates $X$ the number of rows to set to 0,
then randomly selects $X$ rows without replacement to set to 0. 
The second column ('rot_xyz') and third column ('labels') are then set to 0 with same shape.

`data_point`: a row of the pd.Dataframe with second column a 2D np array ('rot_xyz'), and third column a 1D np array ('labels') \
`seed`: the seed for random selection, default = 1
**return**: a row of the pd.Dataframe with same dimensionality as before, but with some rows of the matrix and vector set to 0


### simulate_missing
`simulate_missing(df, portion: float= 0.1, seed = 1)`

Simulates missing markers for the whole dataframe with only a portion of the data being subjected to `simmissing_row`

`df`: a pd.Dataframe with second column a 2D np array ('rot_xyz'), and third column a 1D np array ('labels') \
`portion`: the portion of samples to undergo `simmissing_row` \
`seed`: the seed for random selection, is also passed to `simmissing_row`, default = 1


### simmissing_marker
`simmissing_marker(coord_matrix, seed = 1)`

Simulate missing marker for *a* rot_xyz matrix only. Mainly exists as a more versatile alternative. 
Outputs a size (1,3) zero matrix if error occurs, eg the input is empty.

`coord_matrix`: a 2D numpy array, could work for tensors theoretically. \
`seed`: the seed for random selection, default = 1


### permute
`permute(data_point, seed = 1)`

Permutes the rot_xyz coordinates matrix and the labels simultaneously by row, 
or just the matrix if the column for labels does not exist. 
The second column must be the coordinate matrices ('rot_xyz') and third column must be the labels if it exists.
Accepts a row from a pd.DataFrame (pd.Series) only and is meant to use with `.apply`.

`data_point`: a slice from a pd.DataFrame (pd.Series) \
`seed`: the seed for RNG, default = 1


### permute_df
`permute_df(df, seed = 1)`

Wrapper of `permute` for using on a dataframe directly.

`df`: pd.DataFrame with second column ('rot_xyz') and third column ('labels') (if the third column exists)
`seed`: the seed for RNG, default = 1



## dataclasses

The custom data classes for model training and making predictions of long/multiple sequences which requires batching.


### EncMarkerDataset
`EncMarkerDataset(dataframe, num_class)`

The dataset used for the encoder only transformer model. It turns the 0-8 labelling to one-hot encoded labelling.

`dataframe`: a pd.Dataframe with column 'rot_xyz' ($n \times 3$ coordinate matrix) and 'label' (size $n$ vector) \
`num_class`: number of classes, in the original setting with 8 markers, the number classes is 9

The class is supposed to be passed in a torch dataloader, for example:
```python
from torch.utils.data import DataLoader
foo1 = EncMarkerDataset(df, 9)
foo2 = DataLoader(foo1, batch_size)
```
Then the data can be used in batch, e.g.
```python
for batch, (X, y) in enumerate(foo2):
```
where 
- `X` is a torch.Tensor with dimensions $(\text{batch_size}, n, 3)$, list of coordinate matrices
- `y` will be a torch.Tensor with dimensions $(\text{batch_size}, n, n)$, 
a list of class labels in one-hot encoded fashion 


### AutoMarkerDataset
`AutoMarkerDataset(src_df, tgt_df, noise:bool, pred:bool = False)`

The dataset used for the encoder-decoder transformer model. 
For both `src_df` and `tgt_df`, they must contain columns with names `'rot_xyz'` and `'rot_xyz_mask'`, 
preferably created by [padding](#padding).

`src_df`: pd.Dataframe containing the coordinates of the source, which is passed to the encoder side \
`tgt_df`: pd.Dataframe containing the coordinates of the target, which is passed to the decoder side \
`noise`: boolean, if true, it adds noise to the target \
`pred`: boolean, if true, prediction mode is used, otherwise, training mode is used 












