import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import norm
import random

from birdwinglabel.EncDecTransformers.factories import IdentifyMarkerTimeIndptTransformer
from birdwinglabel.dataprocessing.data import subset_by_seqID, get_list_of_seqID
from birdwinglabel.common import prepforML
from birdwinglabel.EncoderOnlyTransformers.basic import IndptLabellingTransformer
from birdwinglabel.EncoderOnlyTransformers.labelling import enc_labelling
from birdwinglabel.visualisation.plot3d import plot_sequence


# prep for model 2
def remove_non_marker(data_point):
    '''
    data_point: pd.Series with col 'frameID', 'rot_xyz', 'labels'
    rot_xyz: matrix [max_length, 3]
    labels: torch.tensor vector of int 0-8 [max_length]
    '''
    rot_xyz = data_point['rot_xyz']
    labels = data_point['labels']
    frameID = data_point['frameID']
    # Create mask for labels not equal to 0
    mask = labels != 0
    # Filter rot_xyz and labels
    filtered_rot_xyz = rot_xyz[mask]
    filtered_labels = labels[mask]
    # Sort by label ascending
    sort_idx = np.argsort(filtered_labels)
    sorted_rot_xyz = filtered_rot_xyz[sort_idx]
    sorted_labels = filtered_labels[sort_idx]
    # Return as pd.Series
    return pd.Series({'frameID': frameID, 'rot_xyz': sorted_rot_xyz, 'labels': sorted_labels})



def autoenc_predict(src, tgt, model: IdentifyMarkerTimeIndptTransformer, model_param_path):
    '''
    src_df: pd.DataFrame with col 'frameID', 'rot_xyz'
        'rot_xyz' entries are matrices of [num_marker, 3]
    tgt_df: pd.DataFrame with col 'frameID', 'rot_xyz'
        'rot_xyz' entries are matrices of [num_marker, 3], num_marker <= 8
    model: IdentifyMarkerTimeIndptTransformer
    model_param_path: path to .pth where the trained params are stored
    '''
    from birdwinglabel.common.createtorchdataset import MarkerTimeIndptDataset
    from torch.utils.data import DataLoader

    src_max_length = model.src_marker
    tgt_max_length = model.tgt_marker

    # Prepare src DataFrame
    in_src_df = src.copy()
    in_src_df = prepforML.permute_df(in_src_df)
    in_src_df[['rot_xyz', 'rot_xyz_mask']] = in_src_df['rot_xyz'].apply(
        lambda x: pd.Series(prepforML.padding(x, src_max_length))
    )

    # # Prepare tgt DataFrame
    in_tgt_df = tgt.copy()
    # in_tgt_df = prepforML.permute_df(in_tgt_df)
    # in_tgt_df['rot_xyz'] = in_tgt_df['rot_xyz'].apply(prepforML.simmissing_marker)
    # in_tgt_df[['rot_xyz', 'rot_xyz_pad_mask']] = in_tgt_df['rot_xyz'].apply(
    #     lambda x: pd.Series(prepforML.padding(x, tgt_max_length))
    # )

    # Create dataset and dataloader
    dataset = MarkerTimeIndptDataset(in_src_df, in_tgt_df, noise=False, pred=True)
    dataloader = DataLoader(dataset, batch_size=10)

    # Load model parameters
    device = 'cuda'
    model.load_state_dict(torch.load(model_param_path, map_location=device))
    model.eval()
    model.to(device)

    all_preds = []
    frame_ids = []

    with torch.no_grad():
        for batch in dataloader:
            src_tensor, tgt_tensor, src_pad_mask, tgt_pad_mask, _ = batch
            src_tensor = src_tensor.to(device)
            tgt_tensor = tgt_tensor.to(device)
            src_pad_mask = src_pad_mask.to(device)
            tgt_pad_mask = tgt_pad_mask.to(device)

            pred = model(src_tensor, tgt_tensor, src_pad_mask, tgt_pad_mask)
            # pred: [batch, 8, 3]
            all_preds.append(pred.cpu())
            # Collect frameIDs for this batch
            start_idx = len(frame_ids)
            end_idx = start_idx + src_tensor.shape[0]
            frame_ids.extend(tgt.iloc[start_idx:end_idx]['frameID'].values)

    # Concatenate predictions
    all_preds = torch.cat(all_preds, dim=0)
    out_df = pd.DataFrame({
        'frameID': frame_ids,
        'rot_xyz': [arr.numpy() for arr in all_preds]
    })
    return out_df

def test_acc(pred_df, gold_df):
    abs_errors = []
    rel_errors = []
    # for each row of pred and gold
    for idx in range(len(pred_df)):
        pred = pred_df.iloc[idx]['rot_xyz']  # shape: [8, 3]
        gold = gold_df.iloc[idx]['rot_xyz']  # shape: [num_marker, 3], num_marker <= 8

        # obtain rot_xyz of pred dim: [8,3] and gold dim: [num_marker,3] slice to [:8,3]
        num_marker = min(pred.shape[0], gold.shape[0])
        pred = pred[:num_marker]
        gold = gold[:num_marker]

        # compute L2loss for each row of the matrices [8]
        l2_error = np.linalg.norm(pred - gold, axis=1)  # [num_marker]
        gold_l2 = np.linalg.norm(gold, axis=1)
        # compute relative error to gold [8]
        rel_error = l2_error / (gold_l2 + 1e-8)

        abs_errors.append(l2_error)
        rel_errors.append(rel_error)

    # output dataframe with col0: 'abs_error' [num_row, 8], 'rel_error' [num_row, 8]
    return pd.DataFrame({'abs_error': abs_errors, 'rel_error': rel_errors})

def autoenc_label(raw_df, pred_df, sample_variance_path, tol: float = 0.05):
    '''
    :param raw_df: col0 'frameID', col1 'rot_xyz' dim: [num_marker,3]
    :param pred_df: col0 'frameID', col1 'rot_xyz' dim: [8,3]
    :param sample_variance_path: path to sample variance dim: [8,3]
    :param tol: lower bound of probability to label as marker
    :return: col0 'frameID', col1 'rot_xyz' dim: [num_marker,3], col2 'label' dim: [num_marker]
    '''

    # Load sample variance [8,3]
    sample_variance = np.load(sample_variance_path)  # shape: [8, 3]
    std = np.sqrt(sample_variance)  # [8, 3]

    raw_df = raw_df.reset_index(drop=True)
    out_rows = []
    for idx, row in raw_df.iterrows():
        frameID = row['frameID']
        raw_rot_xyz = row['rot_xyz']  # [num_marker, 3]
        pred_rot_xyz = pred_df.iloc[idx]['rot_xyz']  # [8, 3]

        num_marker = raw_rot_xyz.shape[0]
        num_classes = pred_rot_xyz.shape[0]

        # Compute |pred - raw| for all combinations
        diff = np.abs(raw_rot_xyz[:, None, :] - pred_rot_xyz[None, :, :])  # [num_marker, 8, 3]

        # Compute two-sided tail probability for each axis using per-class std

        # Broadcast std: [1, 8, 3]
        prob = 2 * (1 - norm.cdf(diff, loc=0, scale=std[None, :, :]))  # [num_marker, 8, 3]

        # Take min over axis=2 (axes), so [num_marker, 8]
        min_prob = np.median(prob, axis=2)
        if idx == 1:

            print(f'min_prob: {min_prob}')

        # Greedy assignment without replacement
        labels = np.zeros(num_marker, dtype=int)
        assigned_classes = set()
        for _ in range(min(num_marker, num_classes)):
            mask = np.ones_like(min_prob, dtype=bool)
            for c in assigned_classes:
                mask[:, c] = False
            masked_probs = np.where(mask, min_prob, -1)
            i, j = np.unravel_index(np.argmax(masked_probs), masked_probs.shape)
            if min_prob[i, j] < tol:
                break
            labels[i] = j + 1
            assigned_classes.add(j)
            min_prob[i, :] = -1

        out_rows.append({'frameID': frameID, 'rot_xyz': raw_rot_xyz, 'labels': labels})

    return pd.DataFrame(out_rows)


def autoenc_label_hungarian(raw_df, pred_df, sample_variance_path, tol: float = 0.05):
    # Hungarian algorithm, finds the optimal assignment that maximizes the total score (probability)
    from scipy.stats import norm
    from scipy.optimize import linear_sum_assignment

    sample_variance = np.load(sample_variance_path)
    std = np.sqrt(sample_variance)

    raw_df = raw_df.reset_index(drop=True)
    out_rows = []
    for idx, row in raw_df.iterrows():
        frameID = row['frameID']
        raw_rot_xyz = row['rot_xyz']
        pred_rot_xyz = pred_df.iloc[idx]['rot_xyz']

        num_marker = raw_rot_xyz.shape[0]
        num_classes = pred_rot_xyz.shape[0]

        diff = np.abs(raw_rot_xyz[:, None, :] - pred_rot_xyz[None, :, :])
        prob = 2 * (1 - norm.cdf(diff, loc=0, scale=std[None, :, :]))
        min_prob = np.median(prob, axis=2)

        # Use negative probabilities as cost (since Hungarian minimizes)
        cost_matrix = -min_prob
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        labels = np.zeros(num_marker, dtype=int)
        for i, j in zip(row_ind, col_ind):
            if min_prob[i, j] >= tol:
                labels[i] = j + 1  # 1-based label
            # else: leave as 0

        out_rows.append({'frameID': frameID, 'rot_xyz': raw_rot_xyz, 'labels': labels})

    return pd.DataFrame(out_rows)



if __name__ == '__main__':
    # path to data
    data_dir = Path(__file__).parent.parent / 'dataprocessing'
    src_df = pd.read_pickle(data_dir / "src_df.pkl")
    tgt_df = pd.read_pickle(data_dir / "tgt_df.pkl")

    seqID_list = get_list_of_seqID(src_df)
    random.seed(1)
    sample_idx = random.sample(seqID_list, 250)
    unsampled_seqIDs = list(set(seqID_list) - set(sample_idx))
    test_df = subset_by_seqID(src_df, [unsampled_seqIDs[0]])
    test_answer_df = subset_by_seqID(tgt_df, [unsampled_seqIDs[0]])

    model1 = IndptLabellingTransformer(embed_dim=32, num_heads=8, mlp_dim=128, num_layers=3, seq_len=32, num_class=9)
    model1_path = Path(__file__).parent.parent / 'EncoderOnlyTransformers' / 'ET_32m_more_data_weights.pth'
    model1_label_df = enc_labelling(test_df, model1, model1_path)

    np.set_printoptions(suppress=True, precision=4)
    print(f'''
        model_1_df ID: {model1_label_df.iloc[0, 0]}
        model_1_df rot_xyz: {model1_label_df.iloc[0, 1]}
        model_1_df label: {model1_label_df.iloc[0, 2]}
        model_1_df colnames: {model1_label_df.columns}
    ''')

    model1_label_df_filtered = model1_label_df.apply(remove_non_marker, axis=1)
    print(f'''
        model1_label_df_filtered rot_xyz: {model1_label_df_filtered.iloc[0, 1]}
        model1_label_df_filtered labels: {model1_label_df_filtered.iloc[0, 2]}
        model1_label_df_filtered colnames: {model1_label_df.columns}
    ''')

    model2 = IdentifyMarkerTimeIndptTransformer(embed_dim=32, num_head=8 , num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=128)
    model2_param_path = Path(__file__).parent / 'model2_weights.pth'
    model2_pred_df = autoenc_predict(test_df, model1_label_df_filtered, model2, model2_param_path)
    print(f'''
            test_df frameID: {test_df.iloc[0, 0]}
            test_df rot_xyz: {test_df.iloc[0, 1]}
            test_df labels: {test_df.iloc[0, 2]}
            test_df colnames: {test_df.columns}
    ''')
    print(f'''
    test_answer_df rot_xyz: {test_answer_df.iloc[0, 0]}
    test_answer_df rot_xyz: {test_answer_df.iloc[0, 1]}
    test_answer_df labels: {test_answer_df.iloc[0, 2]}
    test_answer_df colnames: {test_answer_df.columns}
    ''')
    print(f'''
        model2_pred_df frameID: {model2_pred_df.iloc[0,0]}
        model2_pred_df rot_xyz: {model2_pred_df.iloc[0,1]}
    ''')
    test_result = test_acc(model2_pred_df, test_df)
    print(f'test_result sample: \nabs: {test_result.iloc[0,0]} \nrel:{test_result.iloc[0,1]}')
    input()

    model2_var_path = Path(__file__).parent / 'model2_sample_variance.npy'
    model2_label_df = autoenc_label_hungarian(test_df, model2_pred_df, model2_var_path, tol = 0.01)
    print(f'''
model2_label_df frameID: {model2_label_df.iloc[1,0]}
model2_label_df rot_xyz: {model2_label_df.iloc[1,1]}
model2_label_df labels: {model2_label_df.iloc[1,2]}
test_df frameID: {test_df.iloc[1,0]}
test_df rot_xyz: {test_df.iloc[1,1]}
test_df labels: {test_df.iloc[1,2]}
''')

    plot_sequence(model1_label_df,200)
    plot_sequence(model2_label_df,200)






























