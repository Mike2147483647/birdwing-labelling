import numpy as np
import pandas as pd

import pathlib
import sys
import contextlib

from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from scipy.stats import multivariate_normal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader









# pad the matrices to have 32 rows, apply it over col of matrix
def padding(x, final_length = 32):
    '''
    :param x: 1D or 2D np array of floats with dimension (seq_len, ...)
    :param final_length: default = 32, the outmost dimension will be padded to this number
    :return: np array with dimension (final_length, ...), padding mask with dimension (final_length), where 1 indicates the row is padded
    '''
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
    df.loc[rows_to_set_zero] = df.loc[rows_to_set_zero].apply(lambda row: simmissing_row(row, seed=seed), axis=1)
    return df


# simulate missing marker for *a* rot_xyz matrix only
def simmissing_marker(coord_matrix, seed = 1):
    RNG = np.random.default_rng(seed=seed)
    try:
        num_marker = coord_matrix.shape[0]
        num_to_remain = RNG.integers(1, num_marker+1)
        marker_to_remain = RNG.choice(num_marker, size=num_to_remain, replace=False)
        coord_matrix = coord_matrix[marker_to_remain]
        return coord_matrix
    except Exception as e:
        return np.zeros((1, 3))


# randomize the rows of coords matrix and labels of each data point
def permute(data_point, seed = 1):
    # col1 must be list of matrices, col2 must be list of labels
    RNG = np.random.default_rng(seed=seed)
    try:
        if len(data_point) >= 3 and data_point.index[2] in ['label', 'labels']:
            num_of_rows = data_point.iloc[1].shape[0]
            perm = RNG.permutation(num_of_rows)
            data_point.iloc[1] = data_point.iloc[1][perm]
            data_point.iloc[2] = data_point.iloc[2][perm]
        elif len(data_point) == 2:
            num_of_rows = data_point.iloc[1].shape[0]
            perm = RNG.permutation(num_of_rows)
            data_point.iloc[1] = data_point.iloc[1][perm]
        return data_point
    except Exception as e:
        print("Error in permute with data_point:")
        print(f'{data_point.iloc[0]}\n{data_point.iloc[1]}\n{data_point.iloc[2]}')
        print("Exception:", e)
        return data_point

def permute_df(df, seed = 1):
    df = df.apply(permute, seed=seed, axis = 1)
    return df


###################################################################################################
# train and test
###################################################################################################

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers

    model.train()

    # X: inputs ; y: target
    for batch, (X, y) in enumerate(dataloader):
        # debug check
        # print(f"y shape: {y.shape}, y dtype: {y.dtype}, y sample: {y[:5]}")
        # print(f"X shape: {X.shape}, X dtype: {X.dtype}, X sample: {X[:2]}")

        # Compute prediction and loss
        pred = model(X)
        # print(f"pred shape: {pred.shape}, pred dtype: {pred.dtype}, X sample: {pred[:2]}")
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # visualisation of progress
        if batch % 100 == 0:
            loss, current = loss.item(), batch * X.shape[0] + len(X)    # X.shape[0] is batch size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, total_labels, true_positive, actual_positive,total_frame_correct = 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # print(f"sample y: {y[0]}")
            # print(f"sample pred: {pred[0]}")

            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                # y: [batch, num_marker, num_labels], pred: [batch, num_marker, num_labels]
                max_indices = torch.argmax(pred, dim=2)  # shape: [batch, num_marker]
                print(f"sample max_indices: {max_indices[0]} \nsample y: {y[0]}")
                pred_labels = F.one_hot(max_indices, num_classes=pred.shape[2])  # shape: [batch, num_marker, num_labels]
                pred_labels = pred_labels.float()
                # print(f"sample pred_labels: {pred_labels[0]}")

                # accuracy per entry
                correct += (pred_labels == y).sum().item()
                total_labels += y.numel()

                # accuracy per label (row)
                true_positive += ((pred_labels == 1) & (y == 1)).sum().item()
                actual_positive += (y == 1).sum().item()

                # accuracy per frame
                batch_correct = torch.isclose(pred_labels, y)
                # print(f"sample batch_correct shape: {batch_correct.shape}")
                frame_correct = batch_correct.all(dim=(1, 2))
                # print(f"sample frame_correct shape: {frame_correct.shape}")
                total_frame_correct += frame_correct.sum()
            else:
                raise NotImplementedError("Unsupported loss function for performance calculation.")

    # print(f"pred shape: {pred.shape}, pred dtype: {pred.dtype}, pred sample: {pred[:5]}")
    # print(f"y shape: {y.shape}, y dtype: {y.dtype}, y sample: {y[:5]}")

    test_loss /= num_batches
    accuracy = 100 * correct / total_labels
    true_positive_rate = 100 * true_positive / actual_positive if actual_positive > 0 else 0
    accuracy_per_frame = total_frame_correct / size
    print(f"Test Error: \nAccuracy per entry: {accuracy:>6.3f}%, Avg loss: {test_loss:>8f}")
    print(f"Accuracy per marker: {true_positive} ({true_positive_rate:>6.3f}%)")
    print(f"Accuracy per frame: {accuracy_per_frame:>6.3f} ({total_frame_correct} / {size})")



def trainandtest(loss_fn, optimizer, model, train_dataloader, test_dataloader, epochs = 10, log_file=f'{pathlib.Path(sys.argv[0]).stem}_train_log.txt'):

    # train and test dataloader are instance of DataLoader using train and test data
    # prepare class to output console ouput in txt while keeping the usual console output

    from birdwinglabel.model import AutTransformer, EncTransformer


    class Tee:
        def __init__(self, file):
            self.file = file
            self.stdout = sys.stdout

        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)

        def flush(self):
            self.file.flush()
            self.stdout.flush()

    with open(log_file, 'w') as f, contextlib.redirect_stdout(Tee(f)):
        print("Model architecture:\n", model)
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            if isinstance(model, AutTransformer):
                train_loop_aut(train_dataloader, model, loss_fn, optimizer, t+1, epochs)
                test_loop_aut(test_dataloader, model, loss_fn)
            elif isinstance(model, EncTransformer):
                train_loop(train_dataloader, model, loss_fn, optimizer)
                test_loop(test_dataloader, model, loss_fn)
            else:
                raise NotImplementedError("Unsupported model for calculation.")

        torch.save(model.state_dict(), f'{pathlib.Path(sys.argv[0]).stem}_{model.__class__.__name__}_weights.pth')
        print("Done!")


def train_loop_aut(dataloader, model, loss_fn, optimizer, current_epoch, epochs):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layer
    model.train()
    update_interval = max(1, size // 10)

    all_diffs = []

    for batch, (src, tgt, src_mask, tgt_mask, gold) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(src, tgt, src_mask, tgt_mask)
        loss = loss_fn(pred, gold)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # sample variance of each entry in training set
        if current_epoch == epochs:
            diff = pred - gold  # [batch, 8, 3]
            all_diffs.append(diff.detach().cpu())

        # visualisation of progress
        current = batch * src.shape[0] + len(src)
        if current % update_interval < src.shape[0]:
            percent = int(100 * current / size)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}] ({percent}%)")

    if current_epoch == epochs and all_diffs:
        diffs_cat = torch.cat(all_diffs, dim=0)  # [total_samples, 8, 3]
        diffs_flat = diffs_cat.reshape(diffs_cat.shape[0], -1)  # [total_samples, 24]
        covariance = np.cov(diffs_flat.numpy(), rowvar=False)  # [24, 24]
        np.save(f'{pathlib.Path(sys.argv[0]).stem}_sample_covariance.npy', covariance)




def test_loop_aut(dataloader, model, loss_fn):
    model.eval()
    num_frame = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    loss, within_5, within_10, within_20 = 0, 0, 0, 0
    # print(f'check test dataloader iter {next(iter(dataloader))}')

    with torch.no_grad():
        for batch, (src, tgt, src_mask, tgt_mask, gold) in enumerate(dataloader):
            batch_size = gold.shape[0]
            pred = model(src, tgt, src_mask, tgt_mask)  # [batch, 8, 3]
            loss += loss_fn(pred, gold) * batch_size    # loss_fn uses default 'mean' mode, so multiply by batch size to get sum

            # Compute per-marker L2 norm
            l2_error = torch.norm(pred - gold, p=2, dim=-1)  # [batch, 8]
            gold_l2 = torch.norm(gold, p=2, dim=-1)
            # .clamp(min=1e-8))  # avoid div by zero
            rel_error = l2_error / gold_l2  # relative error [batch, 8]

            # debug
            # if batch == 1:
            #     print(f'src sample: {src[0]} \nsrc_mask sample: {src_mask[0]} \ntgt sample: {tgt[0]}')
            #     print(f'pred sample: {pred[0]} \ngold sample: {gold[0]} \nrelative error: {rel_error[0]}')

            # Compute max relative error per frame (across all markers)
            frame_max_error = rel_error.max(dim=-1).values  # [batch]

            within_5 += (frame_max_error < 0.05).sum().item()
            within_10 += (frame_max_error < 0.10).sum().item()
            within_20 += (frame_max_error < 0.20).sum().item()



    print(f'''
    test loss avg over frame: {loss / num_frame} 
    Proportion of frames that has <5% error: {within_5 / num_frame:.6f}
    Proportion of frames that has <10% error: {within_10 / num_frame:.6f}
    Proportion of frames that has <20% error: {within_20 / num_frame:.6f}
    ''')


###########################################################################
# labelling
###########################################################################


def enc_labelling(raw_df, model, model_param_path):
    '''
    raw_df: pd.DataFrame with col 'frameID', 'rot_xyz'
        'rot_xyz' entries are matrices of [num_marker, 3]
    model: EncTransformer
    model_param_path: path to .pth where the trained params are stored
    max_length: int, pad length
    '''
    max_length = model.seq_len

    # Permute 'rot_xyz' column
    df = raw_df.copy()
    df = permute_df(df)

    # Pad 'rot_xyz' to max_length
    df['rot_xyz'] = df['rot_xyz'].apply(lambda x: padding(x, max_length)[0])

    # Convert to tensor
    rot_xyz_tensor = torch.tensor(np.stack(df['rot_xyz'].values), dtype=torch.float32)

    # Load model parameters
    model.load_state_dict(torch.load(model_param_path, map_location='cuda'))
    model.eval()

    # Forward pass
    with torch.no_grad():
        pred_label_tensor = model(rot_xyz_tensor)
    pred_label_matrix = pred_label_tensor.argmax(dim=2)     # convert to numerical labelling

    # Build labelled DataFrame
    labelled_df = pd.DataFrame({
        'frameID': df['frameID'],
        'rot_xyz': [arr.numpy() for arr in rot_xyz_tensor],
        'labels': [arr.numpy() for arr in pred_label_matrix]
    })

    # Unpad rows using unpad_row
    labelled_df[['rot_xyz', 'labels']] = labelled_df.apply(
        unpad_row, axis=1, result_type='expand'
    )

    return labelled_df


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


def autoenc_predict(src, tgt, model, model_param_path):
    '''
    src_df: pd.DataFrame with col 'frameID', 'rot_xyz'
        'rot_xyz' entries are matrices of [num_marker, 3]
    tgt_df: pd.DataFrame with col 'frameID', 'rot_xyz'
        'rot_xyz' entries are matrices of [num_marker, 3], num_marker <= 8
    model: AutTransformer
    model_param_path: path to .pth where the trained params are stored
    '''
    from birdwinglabel.dataclasses import AutoMarkerDataset
    from torch.utils.data import DataLoader

    src_max_length = model.src_marker
    tgt_max_length = model.tgt_marker

    # Prepare src DataFrame
    in_src_df = src.copy()
    in_src_df = permute_df(in_src_df)
    in_src_df[['rot_xyz', 'rot_xyz_mask']] = in_src_df['rot_xyz'].apply(
        lambda x: pd.Series(padding(x, src_max_length))
    )

    # # Prepare tgt DataFrame
    in_tgt_df = tgt.copy()
    # in_tgt_df = prepforML.permute_df(in_tgt_df)
    # in_tgt_df['rot_xyz'] = in_tgt_df['rot_xyz'].apply(prepforML.simmissing_marker)
    # in_tgt_df[['rot_xyz', 'rot_xyz_pad_mask']] = in_tgt_df['rot_xyz'].apply(
    #     lambda x: pd.Series(prepforML.padding(x, tgt_max_length))
    # )

    # Create dataset and dataloader
    dataset = AutoMarkerDataset(in_src_df, in_tgt_df, noise=False, pred=True)
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
            src_tensor, tgt_tensor, src_pad_mask, tgt_pad_mask = batch
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
    euclid_errors = []
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

        euclid_errors.append(l2_error)
        rel_errors.append(rel_error)

    # output dataframe with col0: 'euclid_error' [num_row, 8], 'rel_error' [num_row, 8]
    return pd.DataFrame({'euclid_error': euclid_errors, 'rel_error': rel_errors})

def autoenc_label_per_entry(raw_df, pred_df, sample_cov_path, tol: float = 0.05, hungarian: bool = True):
    '''
    :param raw_df: col0 'frameID', col1 'rot_xyz' dim: [num_marker,3]
    :param pred_df: col0 'frameID', col1 'rot_xyz' dim: [8,3]
    :param sample_cov_path: path to sample covariance dim: [24,24]
    :param tol: lower bound of probability to label as marker
    :param hungarian: if True, use Hungarian algorithm for assignment
    :return: col0 'frameID', col1 'rot_xyz' dim: [num_marker,3], col2 'label' dim: [num_marker]
    '''
    sample_cov = np.load(sample_cov_path)  # shape: [24, 24]
    sample_variance = np.array(
        [sample_cov[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3].diagonal() for i in range(8)])  # [8, 3]
    std = np.sqrt(sample_variance)  # [8, 3]

    raw_df = raw_df.reset_index(drop=True)
    out_rows = []
    for idx, row in raw_df.iterrows():
        frameID = row['frameID']
        raw_rot_xyz = row['rot_xyz']  # [num_marker, 3]
        pred_rot_xyz = pred_df.iloc[idx]['rot_xyz']  # [8, 3]

        num_marker = raw_rot_xyz.shape[0]
        num_classes = pred_rot_xyz.shape[0]

        diff = np.abs(raw_rot_xyz[:, None, :] - pred_rot_xyz[None, :, :])  # [num_marker, 8, 3]
        prob = 2 * (1 - norm.cdf(diff, loc=0, scale=std[None, :, :]))  # [num_marker, 8, 3]
        min_prob = np.median(prob, axis=2)  # [num_marker, 8]

        labels = np.zeros(num_marker, dtype=int)
        if hungarian:
            # Use Hungarian algorithm for optimal assignment
            cost_matrix = -min_prob
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                if min_prob[i, j] >= tol:
                    labels[i] = j + 1  # 1-based label
        else:
            # Greedy assignment without replacement
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


def autoenc_label_per_marker(raw_df, pred_df, sample_cov_path, tol: float = 0.05, hungarian: bool = True):
    '''
    :param raw_df: DataFrame with 'frameID', 'rot_xyz' [num_marker,3]
    :param pred_df: DataFrame with 'frameID', 'rot_xyz' [8,3]
    :param sample_cov_path: path to sample covariance [24,24]
    :param tol: lower bound of probability to label as marker
    :param hungarian: if True, use Hungarian algorithm for assignment
    :return: DataFrame with 'frameID', 'rot_xyz', 'labels'
    '''
    sample_cov = np.load(sample_cov_path)  # [24, 24]
    cov_blocks = [sample_cov[i*3:(i+1)*3, i*3:(i+1)*3] for i in range(8)]  # list of 8 [3,3] arrays

    raw_df = raw_df.reset_index(drop=True)
    out_rows = []
    for idx, row in raw_df.iterrows():
        frameID = row['frameID']
        raw_rot_xyz = row['rot_xyz']  # [num_marker, 3]
        pred_rot_xyz = pred_df.iloc[idx]['rot_xyz']  # [8, 3]

        num_marker = raw_rot_xyz.shape[0]
        num_classes = pred_rot_xyz.shape[0]

        # Compute probability matrix [num_marker, 8]
        prob_matrix = np.zeros((num_marker, num_classes))
        for i in range(num_marker):
            for j in range(num_classes):
                mean = pred_rot_xyz[j]  # [3]
                cov = cov_blocks[j]     # [3,3]
                prob_matrix[i, j] = multivariate_normal.pdf(raw_rot_xyz[i], mean=mean, cov=cov)

        labels = np.zeros(num_marker, dtype=int)
        if hungarian:
            # Use Hungarian algorithm for optimal assignment
            cost_matrix = -prob_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                if prob_matrix[i, j] >= tol:
                    labels[i] = j + 1
        else:
            # Greedy assignment without replacement
            assigned_classes = set()
            for _ in range(min(num_marker, num_classes)):
                mask = np.ones_like(prob_matrix, dtype=bool)
                for c in assigned_classes:
                    mask[:, c] = False
                masked_probs = np.where(mask, prob_matrix, -1)
                i, j = np.unravel_index(np.argmax(masked_probs), masked_probs.shape)
                if prob_matrix[i, j] < tol:
                    break
                labels[i] = j + 1
                assigned_classes.add(j)
                prob_matrix[i, :] = -1

        out_rows.append({'frameID': frameID, 'rot_xyz': raw_rot_xyz, 'labels': labels})

    return pd.DataFrame(out_rows)












