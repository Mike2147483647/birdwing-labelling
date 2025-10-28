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
    * [trainandtest](#trainandtest)
    * [train_loop](#train_loop)
    * [test_loop](#test_loop-)
    * [train_loop_aut](#train_loop_aut)
    * [test_loop_aut](#test_loop_aut)
  * [dataclasses](#dataclasses)
    * [EncMarkerDataset](#encmarkerdataset)
    * [AutoMarkerDataset](#automarkerdataset)
  * [model](#model)
    * [EncTransformer](#enctransformer)
    * [AutTransformer](#auttransformer)
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


### trainandtest
`trainandtest(loss_fn, optimizer, model, train_dataloader, test_dataloader, 
epochs = 10, log_file=f'{pathlib.Path(sys.argv[0]).stem}_train_log.txt'):`

The go-to function using this package. Automatically trains and tests the model over every epoch.
The log file is named "{name-of-the-running-script}_train_log.txt" by default, 
and is saved completely only after the message "Done!".
The model parameters are saved as "{name-of-the-running-script}_{model-class-name}.pth", 
and is saved only after the message "Done!".

`loss_fn`: instance of loss function to be used during training and testing.\
`optimizer`: instance of optimizer classes from torch, `torch.optim.AdamW` is recommended. \
`train_dataloader`: the torch dataloader object created from `EncMarkerDataset` using the training set. \
`test_dataloader`: the torch dataloader object created from `EncMarkerDataset` using the test set. \
`epochs`: number of epochs of training. \
`log_file`: name of the log file. \


### train_loop
`train_loop(dataloader, model, loss_fn, optimizer)`

The function used to train `EncTransformer`. 
It prints the (average) training loss, the number of training data gone through thus far,
and the total amount of training data as a pseudo progress bar.

`dataloader`: the torch dataloader object created from `EncMarkerDataset`. \
`model`: instance of `EncTransformer`. \
`loss_fn`: loss function to be used during training, instance of `torch.nn.BCEWithLogitsLoss` is recommended. \
`optimizer`: instance of optimizer classes from torch, `torch.optim.AdamW` is recommended. \


### test_loop 
`test_loop(dataloader, model, loss_fn)`

The function used to evaluate the performance of `EncTransformer`.
It prints the test loss (average over all entries), accuracy per entry, test loss average over batch, \
accuracy per marker: number of true positive (true positive rate), \
accuracy per frame and the fraction in integers.

`dataloader`: the torch dataloader object created from `EncMarkerDataset`. \
`model`: instance of `EncTransformer`. \
`loss_fn`: instance of loss function to be used during testing, only supports `torch.nn.BCEWithLogitsLoss`.\


### train_loop_aut
`train_loop_aut(dataloader, model, loss_fn, optimizer, current_epoch, epochs)`

The function used to train `AutTransformer`.
Saves the covariance of marker coordinates in a tgt_marker $\times$ 3 square matrix, 
which can be used later as a reference matching with the observed markers with predicted markers.

`dataloader`: the torch dataloader object created from `AutoMarkerDataset`. \
`model`: instance of `AutTransformer`. \ 
`loss_fn`: instance of loss function to be used during training. \
`optimizer`: instance of optimizer classes from torch, `torch.optim.AdamW` is recommended. \
`current_epoch`: used internally in `trainandtest`. \
`epochs`: used internally in `trainandtest`, number of epochs to be trained. \


### test_loop_aut
`test_loop_aut(dataloader, model, loss_fn)`

The function used to evaluate the performance of `AutTransformer`.
It prints the test loss (average over all frames), with actual values of the loss and total number of frames, 
and the framewise within 5\%, 10\%, 20\% error accuracy, all with actual fraction.

`dataloader`: the torch dataloader object created from `AutoMarkerDataset`. \
`model`: instance of `AutTransformer`. \ 
`loss_fn`: instance of loss function to be used during testing. \



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
Note that the label is implicitly indicated in the target by the order of the markers.

`src_df`: pd.Dataframe containing the coordinates of the source, which is passed to the encoder side \
`tgt_df`: pd.Dataframe containing the coordinates of the target, which is passed to the decoder side \
`noise`: boolean, if true, it adds noise to the target (unused)\
`pred`: boolean, if true, prediction mode is used, otherwise, training mode is used 

The methods `__init__` and `__getitem__` required by the torch dataloader class does different things under two modes.

In training mode, `__getitem__` gives outputs 
`self.src_df[idx], self.tgt_df[idx], self.src_mask[idx], self.tgt_mask[idx], self.gold_df[idx]`.
where `self.src_df` and `self.src_mask` are from the columns `'rot_xyz'` and `'rot_xyz_mask'` of src_df,
and `self.gold_df` is from column `'rot_xyz'` of tgt_df directly.
`self.tgt_df` and `self.tgt_mask` are created by passing a copy of tgt_df to
`simmissing_marker`, `permute_df` and `padding(.,final_length = 8)`.

In prediction mode, `__getitem__` gives outputs 
`self.src_df[idx], self.tgt_df[idx], self.src_mask[idx], self.tgt_mask[idx]`,
where `self.src_df` and `self.src_mask` are from the columns `'rot_xyz'` and `'rot_xyz_mask'` of src_df,
and `self.tgt_df` and `self.tgt_mask` are obtained by applying `padding(.,final_length = 8)` on tgt_df.

So, for now, when training, only feed data with all 8 markers present in tgt_df; 
when predicting, keep tgt_df unpadded and the class will pad it automatically.

(Potentially will make use of the tgt_pad_mask to train with less markers present,
may need another padding function to fill in missing markers, since currently the padding function will only append, not insert.)

The class is supposed to be passed in a torch dataloader, for example:
```python
from torch.utils.data import DataLoader
foo1 = AutoMarkerDataset(src_df, tgt_df, noise = False, pred = False)
foo2 = DataLoader(foo1, batch_size)
```
then you can extract by e.g.
```python
for batch, (src, tgt, src_mask, tgt_mask, gold) in enumerate(foo2):
```



## model

This section includes the machine learning models. 
After initialisation, data can be fed into the models for training or predictions.
Caution: the model parameters have to be loaded afterwards.

### EncTransformer
`EncTransformer(embed_dim: int, num_heads: int, mlp_dim: int, num_layers: int, seq_len: int = 8, dropout: float=0.1, num_class:int = 8)`

The model used for identifying the markers directly. 
It cannot be extended to identify extra markers but have higher accuracy.

`embed_dim` (int): the size of the embedding dimension. It embeds the 3D coordinates to a higher dimensional space. \
`num_heads` (int): the number of heads in the multihead attention block used in the transformer architecture. \
`mlp_dim` (int): the dimension of the fully connected layer inside the transformer architecture. \
Usually a multiple of number of heads. \
`num_layers` (int): the number of layers of the encoder. \
`seq_len` (int): the number of markers (observed). \
`dropout` (float): the probability of dropping out the output of neurons as a measure of regularization. \
`num_class` (int): the number of classes to be identified. \

`EncTransformer.forward(inputs: torch.Tensor)` or `EncTransformer(inputs: torch.Tensor)`

Passes the inputs through the model. 
The output is a tensor with dimension (batch_size, seq_len, num_class).

`inputs` (torch.Tensor): a tensor with dimension (batch_size, seq_len, 3), 
i.e. a list of the marker 3D coordinates matrices.


### AutTransformer
`AutTransformer(embed_dim:int,
num_head: int = 1,
num_encoder_layers: int = 1,
num_decoder_layers: int = 1,
dim_feedforward: int = 4,
dropout: float=0.1,
coord_dim:int = 3,
tgt_marker:int = 8,
src_marker:int = 32,
fc_in_embed:bool = True,
norm_embed:bool = False
):`

The model used for predicting the 3D coordinates of all the markers by combining 
the results of `EncTransformer` (target) and the observed coordinates (source).
The predicted coordinates are matched to the observed coordinates for labelling afterwards.
Note that the label is implicitly indicated in the target by the order of the markers.

`num_head` (int): the number of heads in the multihead attention block used in the transformer architecture. \
`num_encoder_layers` (int): number of encoder layers in the transformer architecture. \
`num_decoder_layers` (int): number of decoder layers in the transformer architecture. \
`dim_feedforward` (int): the dimension of the fully connected layer inside the transformer architecture. 
`dropout` (float): the probability of dropping out the output of neurons as a measure of regularization. \
`coord_dim` (int): the dimension of the markers, usually 3. \
`tgt_marker` (int): number of markers in the target. \
`src_marker` (int): number of markers in the source. \
`fc_in_embed` (bool): usage of using fully connected layer as the embedding before feeding to the transformer, 
else use linear layer. Default: `True`. \
`norm_embed` (bool): normalize the outputs of the embedding layer before feeding to the transformer. Default: `False`. \

`AutTransformer.forward(src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor = None)`

Passes the source and target to obtain predictions.

`src` (torch.Tensor): the source tensor with dimension (batch_size, src_marker, 3),
the tensor of the observed coordinates. \
`tgt` (torch.Tensor): the target tensor with dimension (batch_size, tgt_marker, 3),
the tensor of the coordinates labelled as valid marker by `EncTransformer`. \
`src_mask` (torch.Tensor): 1 indicates the marker is padded, 0-1 matrix with dimension (batch_size, src_marker) .\
`tgt_mask` (torch.Tensor): 1 indicates the marker is padded, 0-1 matrix with dimension (batch_size, tgt_marker).\










