import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path

import birdwinglabel.dataprocessing.data as data

from birdwinglabel.common import prepforML
from birdwinglabel.EncoderOnlyTransformers.basic import IndptLabellingTransformer
from birdwinglabel.dataprocessing.data import unlabelled_seqID, full_no_labels

# get data
unlabelled_seqID = unlabelled_seqID()
working_df = data.subset_by_seqID(full_no_labels, [unlabelled_seqID[2]])
working_df = data.stack_matrix(working_df)
print(working_df.info())

# subset the sequence to label
working_df.iloc[:,1] = working_df.iloc[:,1].apply(lambda x: prepforML.padding(x,32)[0])
print(f'working_df sample: {working_df.iloc[2]} \n{working_df.iloc[2,1]} ')

labelling_seq_tensor = torch.tensor(working_df.iloc[:,1], dtype=torch.float32)

# set up the transformer
model = IndptLabellingTransformer(embed_dim=32, num_heads=8, mlp_dim=128, num_layers=3, seq_len=32, num_class=9)

model.load_state_dict(torch.load('ET_32m_more_data_weights.pth', map_location='cuda'))
model.eval()

# pass unlabelled data to transformer
pred_label_tensor = model.forward(labelling_seq_tensor)
# pred_label_tensor has dim [num_of_frames, 32, 9] where each row is in terms of logits (-inf,inf)
# pred_label_tensor = (torch.sigmoid(pred_label_tensor) > 0.5).float()
pred_label_matrix = pred_label_tensor.argmax(dim=2)
# pred_label_matrix has dim [num_of_frames, 32]



labelled_df = pd.DataFrame({
    'frameID': working_df['frameID'],
    'rot_xyz': [arr.numpy() for arr in labelling_seq_tensor],  # Each entry is a [32,3] matrix
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

print(f'frameID: {labelled_df.iloc[2,0]} \n coordinates:{labelled_df.iloc[2,1]}\n labels: {labelled_df.iloc[2,2]}\n')
      # f'unprocessed_frameID: {labelling_seq_df.iloc[2,0]} \nunprocessed_coords: {labelling_seq_df.iloc[2,1]} \nunprocessed_labels: {labelling_seq_df.iloc[2,2]}')

def sort_by_label(row):
    rot_xyz = np.array(row['rot_xyz'])
    labels = np.array(row['labels'])
    # Mask for labels 1-8
    mask_1_8 = (labels >= 1) & (labels <= 8)
    mask_0 = (labels == 0)
    # Indices for 1-8 sorted
    idx_1_8 = np.argsort(labels[mask_1_8])
    # Indices for 0's (order doesn't matter)
    idx_0 = np.arange(np.sum(mask_0))
    # Concatenate indices
    sorted_rot_xyz = np.concatenate([rot_xyz[mask_1_8][idx_1_8], rot_xyz[mask_0][idx_0]])
    sorted_labels = np.concatenate([labels[mask_1_8][idx_1_8], labels[mask_0][idx_0]])
    return sorted_rot_xyz, sorted_labels

# labelled_df[['rot_xyz', 'labels']] = labelled_df.apply(
#     sort_by_label, axis=1, result_type='expand'
# )

# output as file
labelled_df.to_pickle('Transformer_labelled_df.pkl')

