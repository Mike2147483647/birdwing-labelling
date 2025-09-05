import numpy as np
import pandas as pd
import torch
from pathlib import Path

from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from scipy.stats import multivariate_normal
import random

from birdwinglabel.EncDecTransformers.factories import IdentifyMarkerTimeIndptTransformer
from birdwinglabel.dataprocessing.data import subset_by_seqID, get_list_of_seqID
from birdwinglabel.common import prepforML
from birdwinglabel.EncoderOnlyTransformers.basic import IndptLabellingTransformer
from birdwinglabel.EncoderOnlyTransformers.labelling import enc_labelling
from birdwinglabel.visualisation.plot3d import plot_sequence
from labelling import remove_non_marker, autoenc_predict, autoenc_label_per_marker, autoenc_label_per_entry


def test_labelling_accuracy(
    source_df, model1, model1_path, model2, model2_path, model2_cov_path,
        tolerances=[0.0001,0.0005,0.001,0.005], label_mode='marker'
):
    # # Step 1: Run model1 to get initial labels
    # from birdwinglabel.EncoderOnlyTransformers.labelling import enc_labelling
    # from labelling import remove_non_marker, autoenc_predict, autoenc_label_per_marker

    model1_label_df = enc_labelling(source_df, model1, model1_path)
    model1_label_df_filtered = model1_label_df.apply(remove_non_marker, axis=1)

    # Step 2: Run model2 to get predictions
    model2_pred_df = autoenc_predict(source_df, model1_label_df_filtered, model2, model2_path)

    # Step 3: Run label assignment for each tolerance
    results = {}
    for tol in tolerances:
        if label_mode == 'marker':
            label_func = autoenc_label_per_marker
        elif label_mode == 'entry':
            label_func = autoenc_label_per_entry
        else:
            raise ValueError("label_mode must be 'marker' or 'entry'")

        model2_label_df = label_func(
            source_df, model2_pred_df, model2_cov_path, tol=tol, hungarian=True
        )

        marker_correct = 0
        total_markers = 0
        frame_correct = 0
        num_frames = len(source_df)

        for i in range(num_frames):
            gold_labels = source_df.iloc[i]['labels']
            pred_labels = model2_label_df.iloc[i]['labels']
            n = min(len(gold_labels), len(pred_labels))
            marker_correct += (gold_labels[:n] == pred_labels[:n]).sum()
            total_markers += n
            if np.array_equal(gold_labels[:n], pred_labels[:n]):
                frame_correct += 1

        marker_accuracy = marker_correct / total_markers if total_markers > 0 else 0
        frame_accuracy = frame_correct / num_frames if num_frames > 0 else 0

        results[tol] = {
            'marker_accuracy': marker_accuracy,
            'frame_accuracy': frame_accuracy
        }
        print(f"Tolerance {tol}: Marker accuracy {marker_accuracy:.4f}, Frame accuracy {frame_accuracy:.4f}")

    return results





if __name__ == '__main__':

    # path to data
    data_dir = Path(__file__).parent.parent / 'dataprocessing'
    src_df = pd.read_pickle(data_dir / "src_df.pkl")
    tgt_df = pd.read_pickle(data_dir / "tgt_df.pkl")

    seqID_list = get_list_of_seqID(src_df)
    random.seed(1)
    sample_idx = random.sample(seqID_list, 250)
    unsampled_seqIDs = list(set(seqID_list) - set(sample_idx))
    sample_seqID_list = sample_idx[200:250]
    test_df = subset_by_seqID(src_df, sample_seqID_list)
    print(f'test_df info: {test_df.info()}')

    modelET = IndptLabellingTransformer(embed_dim=32, num_heads=8, mlp_dim=128, num_layers=3, seq_len=32, num_class=9)
    modelET_path = Path(__file__).parent.parent / 'EncoderOnlyTransformers' / 'ET_32m_more_data_weights.pth'

    modelT = IdentifyMarkerTimeIndptTransformer(embed_dim=32, num_head=8, num_encoder_layers=3, num_decoder_layers=3,
                                                dim_feedforward=128, fc_in_embed=True, pos_enc=False)
    modelT_param_path = Path(__file__).parent / 'model2_fc_embed_20epoch_weights.pth'
    modelT_var_path = Path(__file__).parent / 'model2_fc_embed_20epoch_sample_covariance.npy'

    acc_marker = test_labelling_accuracy(test_df, modelET, modelET_path, modelT, modelT_param_path, modelT_var_path, label_mode='marker')
    acc_entry = test_labelling_accuracy(test_df, modelET, modelET_path, modelT, modelT_param_path, modelT_var_path, label_mode='entry')
    pd.DataFrame(acc_marker).to_csv('acc_marker_df.csv', index=False)
    pd.DataFrame(acc_entry).to_csv('acc_entry_df.csv', index=False)
    # # starts here
    # model1_label_df = enc_labelling(test_df, model1, model1_path)
    #
    # np.set_printoptions(suppress=True, precision=4)
    # model1_label_df_filtered = model1_label_df.apply(remove_non_marker, axis=1)
    #
    # model2_pred_df = autoenc_predict(test_df, model1_label_df_filtered, model2, model2_param_path)
    #
    # model2_label_df = autoenc_label_per_marker(test_df, model2_pred_df, model2_var_path, tol=0.001, hungarian=True)
    # # ends here
    #
    # plot_sequence(model1_label_df, 200)
    # plot_sequence(model2_label_df, 200)




















