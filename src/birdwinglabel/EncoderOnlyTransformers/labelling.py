import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path

import birdwinglabel.dataprocessing.data as data

from birdwinglabel.common import prepforML
from birdwinglabel.EncoderOnlyTransformers.basic import IndptLabellingTransformer



def enc_labelling(raw_df, model:IndptLabellingTransformer, model_param_path):
    '''
    raw_df: pd.DataFrame with col 'frameID', 'rot_xyz'
        'rot_xyz' entries are matrices of [num_marker, 3]
    model: IndptLabellingTransformer
    model_param_path: path to .pth where the trained params are stored
    max_length: int, pad length
    '''
    max_length = model.seq_len

    # Permute 'rot_xyz' column
    df = raw_df.copy()
    df = prepforML.permute_df(df)

    # Pad 'rot_xyz' to max_length
    df['rot_xyz'] = df['rot_xyz'].apply(lambda x: prepforML.padding(x, max_length)[0])

    # Convert to tensor
    rot_xyz_tensor = torch.tensor(np.stack(df['rot_xyz'].values), dtype=torch.float32)

    # Load model parameters
    model.load_state_dict(torch.load(model_param_path, map_location='cuda'))
    model.eval()

    # Forward pass
    with torch.no_grad():
        pred_label_tensor = model(rot_xyz_tensor)
    pred_label_matrix = pred_label_tensor.argmax(dim=2)

    # Build labelled DataFrame
    labelled_df = pd.DataFrame({
        'frameID': df['frameID'],
        'rot_xyz': [arr.numpy() for arr in rot_xyz_tensor],
        'labels': [arr.numpy() for arr in pred_label_matrix]
    })

    # Unpad rows using prepforML.unpad_row
    labelled_df[['rot_xyz', 'labels']] = labelled_df.apply(
        prepforML.unpad_row, axis=1, result_type='expand'
    )

    return labelled_df


if __name__ == '__main__':
    # get data
    data_dir = Path(__file__).parent.parent / 'dataprocessing'
    working_df = pd.read_pickle(data_dir / 'src_df.pkl')
    seqID_list = data.get_list_of_seqID(working_df)
    labelling_seq_df = data.subset_by_seqID(working_df, [seqID_list[48]])

    model1 = IndptLabellingTransformer(embed_dim=32, num_heads=8, mlp_dim=128, num_layers=3, seq_len=32, num_class=9)
    model1_param_path = Path(__file__).parent / 'ET_32m_more_data_weights.pth'
    out_seq_df = enc_labelling(labelling_seq_df, model1, model1_param_path)
    print(f'''
        out_seq_df ID: {out_seq_df.iloc[0, 0]}
        out_seq_df rot_xyz: {out_seq_df.iloc[0, 1]}
        out_seq_df label: {out_seq_df.iloc[0, 2]}
    ''')

    from birdwinglabel.visualisation.plot3d import plot_sequence
    plot_sequence(out_seq_df)














