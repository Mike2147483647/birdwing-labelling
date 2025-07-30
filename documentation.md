<!-- TOC -->
  * [Introduction <a name='introduction'></a>](#introduction-a-nameintroductiona)
  * [common <a name='common'></a>](#common-a-namecommona)
    * [common.prepforML <a name='common.prepforML'></a>](#commonprepforml-a-namecommonprepformla)
    * [common.createtorchdataset <a name='common.createtorchdataset'></a>](#commoncreatetorchdataset-a-namecommoncreatetorchdataseta)
    * [common.trainandtest <a name="common.trainandtest"></a>](#commontrainandtest-a-namecommontrainandtesta)
  * [dataprocessing <a name="dataprocessing"></a>](#dataprocessing-a-namedataprocessinga)
    * [dataprocessing.crossreference](#dataprocessingcrossreference)
    * [dataprocessing.data](#dataprocessingdata)
    * [dataprocessing.labellingcrossrefdata](#dataprocessinglabellingcrossrefdata)
<!-- TOC -->


## Introduction <a name='introduction'></a>
This is documentation of birdwinglabel. Most of the error produced from incorrect dimension/name of array when using 
common, MLP and Transformer can be solved by reshaping the dataframe into `col0: 'frameID'` `col1: 'markers_matrix'` `col2: 'label'`

## common <a name='common'></a>
common assets. Run the code below before running examples in this section.
```python
import numpy as np
from birdwinglabel.common import trainandtest, prepforML, createtorchdataset
```

### common.prepforML <a name='common.prepforML'></a>

`padding(x, final_length = 32)`

Pads `x` have length `final_length`. `x` can be 1D or 2D only, best used with np arrays.

```
# Example: 
matrix = np.random.rand(2,3)
print(f'matrix: \n{matrix}')
matrix = prepforML.padding(matrix, 4)
print(f'matrix after padding: \n{matrix}')
```

`simmissing_row(data_point, seed = 1)`

Simulate situations of missing markers in raw data when the fully labelled dataset is used for training.
A row is randomly chosen in the matrix to be removed (set to 0), and its corresponding label is also set to 0.
Requires the data point to have `col1: 'markers_matrix'` `col2: 'label'`. Please use `simulate_missing` for daily usage.

`simulate_missing(df, portion: float= 0.1, seed = 1)`

Simulate situations of missing markers in raw data when the fully labelled dataset is used for training.
Applies `simmissing_row` to randomly chosen row of the input dataframe. 
`portion` controls the amount of datapoints (rows) chosen to have missing markers. 

`permute(data_point, seed = 1)`

Randomize the rows of coordinates matrix and labels of each data point, so that the model does not learn to label any coordinates in an increasing order
Not recommend to use this directly. Please use `permute_df` instead.

`permute_df(df, seed = 1)`

Wrapper for `permute` used for dataframes.


### common.createtorchdataset <a name='common.createtorchdataset'></a>

Create custom objects to pass in `torch.utils.data.DataLoader`. For example,

```python
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from birdwinglabel.common.createtorchdataset import MarkerDataset, HotMarkerDataset

matrices = [np.random.rand(5,3) for _ in range(5)]
labels = [np.random.randint(0,9,5) for _ in range(5)]

data = pd.DataFrame(data = {'markers_matrix': matrices,
                            'label': labels
                           }
                    )
dataset = HotMarkerDataset(data, 9)
dataloader = DataLoader(dataset, batch_size=50)
```


`MarkerDataset(dataframe)`   

`MarkerDataset` is a custom DataLoader. It is a child of `torch.utils.data.dataset`, 
which is then passed into `torch.nn.Module` and its subclasses. In this package, it prepares the Dataloader for training
with loss function `nn.CrossEntropyLoss`.

`dataframe` must have `col1: 'markers_matrix'` `col2: 'label'`. For now, it only supports 8 classes, i.e. dataset with no missing markers.

`HotMarkerDataset(dataframe, num_class)`

`MarkerDataset` is a custom DataLoader. It is a child of `torch.utils.data.dataset`, 
which is then passed into `torch.nn.Module` and its subclasses. In this package, it prepares the Dataloader for training
with loss function `nn.BCEWithLogitsLoss`.

`dataframe` must have `col1: 'markers_matrix'` `col2: 'label'`. `num_class` should be number of markers on bird + 1.

### common.trainandtest <a name="common.trainandtest"></a>

`train_loop(dataloader, model, loss_fn, optimizer)`

Used internally in `trainandtest`. For each batch in dataloader, it computes the model output and calculates the loss. 
It then goes through backpropagation.

Parameters:
- dataloader: object of class `torch.utils.data.DataLoader`
- model: object of class `torch.nn.Module`
- loss_fn: loss function in `torch.nn`
- optimizer: `torch.optim` object, preferably instance of `optim.Adam`

`test_loop(dataloader, model, loss_fn)`

Used internally in `trainandtest`. It evaluates the inputs contained in `dataloader` with `model` and gives a prediction 
by taking the argmax of each row of the `num_marker` $\times$ `num_labels` matrix. 
It then computes the test set's
1. accuracy per entry in the `num_marker` $\times$ `num_labels` matrix
2. average loss (in logits)
3. accuracy per marker
4. accuracy per frame

Parameters:
- dataloader: object of class `torch.utils.data.DataLoader`
- model: object of class `torch.nn.Module`
- loss_fn: loss function in `torch.nn`

`trainandtest(loss_fn, optimizer, model, train_dataloader, test_dataloader, epochs = 10, log_file='train_log.txt')`

The go-to function in training models with this package. It will run `train_loop` and `test_loop` 
for the number of `epochs` times. It is advised to create instances for `optimizer`, `model`, `train_dataloader` and
`test_dataloader` beforehand. Remember to use the same specifications of the model when running the trained model for
labelling.

Parameters:
- loss_fn: loss function in `torch.nn`
- optimizer: `torch.optim` object, preferably instance of `optim.Adam`
- model: object of class `torch.nn.Module`
- train_dataloader: object of class `torch.utils.data.DataLoader`, which is created from the training set
- test_dataloader: object of class `torch.utils.data.DataLoader`, which is created from the test set
- epochs: number of rounds of gradient descent, length of training



## dataprocessing <a name="dataprocessing"></a>

Most of the files are not intended for normal package use, instead they create `.pkl` files to provide access of 
preprocessed data to model training. Exceptions are the helper functions in `dataprocessing/data`.

### dataprocessing.crossreference

Not intended to be used externally. However, since the data is not included in the package, it is required to run this
file at least once. It creates a unlabelled set by comparing the FullBilateral dataset and the FullNoLabel
dataset, so that the unlabelled set is a subset of FullNolabel, which contains all the markers and frames in FullBilateral.
The unlabelled set has 3 columns, with 152878 non-null rows:

1. frameID
2. rot_xyz, matrix of n x 3, where n is the number of markers observed in the frame.
  It is filtered from FullNoLabels. The order of the row in each matrix is sorted according to FullBilateral.
3. rot_xyz_mask, is a vector of length n, with n described as above. It is a vector of indicator that 
  whether that row of coordinates exists in FullBilateral.

This unlabelled set is then output as `unlabelled_grouped_df.pkl`

The other output from this file is `gold_df.pkl`. It is essentially FullBilateral, but with an extra column
 `col 26 rot_xyz_matrix`, which is stacking col 2-25 of FullBilateral into a 8 (number of labels) x 3 (xyz coordinates) 
matrix.


### dataprocessing.data

Most of the functions in this file are helper functions or wrapper of pandas functions that makes the code more 
expressive. Here is a quick list of these functions, as most of their names are self explanatory:

- `get_list_of_seqID(bird_data)`. `bird_data` is a pandas `DataFrame` that contains column `seqID`.
- `get_list_of_frameID(bird_data)`. `bird_data` is a pandas `DataFrame` that contains column `frameID`.
- `subset_by_seqID(bird_data, seq_id_list)`. `bird_data` is a pandas `DataFrame` that contains column `seqID`. 
  `seq_id_list` is a list of seqID, in most situations from `get_list_of_seqID`.
- `subset_by_frameID(bird_data, frame_id_list)`. `bird_data` is a pandas `DataFrame` that contains column `frameID`.
  `frame_id_list` is a list of frameID, in most situations from `get_list_of_frameID`.

`unlabelled_seqID()` 

Used internally for checks.

`stack_matrix(unlabelled_df)`

Used within `dataprocessing`. For unlabelled dataset (FullNoLabel), filter and stack rot_xyz into n (no of markers) x 3 
matrix. Outputs a dataframe with col0 frameID, col1 rot_xyz (the matrices).

`create_training(labelled_data, seed = 1)`

Depreciated because it only adapts the fully labelled training case.


### dataprocessing.labellingcrossrefdata

Continues what `crossreference` did. Not intended to be used from any other files and is only meant to be run as main. 
It outputs `labelled_df.pkl` and have different checks in process of creating `labelled_df.pkl`.









