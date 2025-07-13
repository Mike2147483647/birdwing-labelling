import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path

import birdwinglabel.dataprocessing.data as data
from birdwinglabel.dataprocessing.data import full_no_labels

from birdwinglabel.common import prepforML
from birdwinglabel.Transformers.basic import DecoderOnlyTransformer


# subset and prepare the unlabelled dataset for transformer
unlabelled_seqID = data.unlabelled_seqID()
unlabelled_df = data.subset_by_seqID(full_no_labels, [unlabelled_seqID[0]])
unlabelled_df = data.stack_matrix(unlabelled_df)
unlabelled_df.iloc[:,1] = unlabelled_df.iloc[:,1].apply(prepforML.padding)
print(f'unlabelled_df.info() \n{unlabelled_df.info()}')

# separate frameID and rot_xyz because torch recommend inputting np arrays instead of list of np arrays
unlabelled_frameID = unlabelled_df['frameID']
unlabelled_matrix = np.stack(unlabelled_df.iloc[:,1].to_numpy())
unlabelled_tensor = torch.tensor(unlabelled_matrix, dtype=torch.float32)
# unlabelled_tensor has dim [num_of_frames,32,3]
print(f'unlabelled_tensor.shape: \n{unlabelled_tensor.shape}')

# set up the transformer
model = DecoderOnlyTransformer(embed_dim=32, num_heads=8, mlp_dim=128, num_layers=3, seq_len=32, num_class=9)

model.load_state_dict(torch.load('DecoderOnlyTransformer_weights.pth'))
model.eval()

# pass unlabelled data to transformer
pred_label_tensor = model.forward(unlabelled_tensor)
# pred_label_tensor has dim [num_of_frames, 32, 9] where each row is in terms of logits (-inf,inf)
# pred_label_tensor = (torch.sigmoid(pred_label_tensor) > 0.5).float()
pred_label_matrix = pred_label_tensor.argmax(dim=2)
# pred_label_matrix has dim [num_of_frames, 32]

labelled_df = pd.DataFrame({
    'frameID': unlabelled_frameID,
    'rot_xyz': list(unlabelled_matrix),  # Each entry is a [32,3] matrix
    'labels': list(pred_label_matrix)   # Each entry is a [32] vector
})

def unpad_row_apply(row, pad_value=0):
    rot_xyz = np.array(row['rot_xyz'])
    labels = np.array(row['labels'])
    mask = ~(np.all(rot_xyz == pad_value, axis=1))
    return rot_xyz[mask], labels[mask]

labelled_df[['rot_xyz', 'labels']] = labelled_df.apply(
    unpad_row_apply, axis=1, result_type='expand'
)

# print(f'frameID: {labelled_df.iloc[2,0]} \n coordinates:{labelled_df.iloc[2,1]}\n labels: {labelled_df.iloc[2,2]}\n'
#       f'unpad_coords: {labelled_df.iloc[2,3]} \nunpad_labels: {labelled_df.iloc[2,4]}')


# output as file
labelled_df.to_pickle('Transformer_labelled_df.pkl')

