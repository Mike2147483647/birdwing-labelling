import numpy as np
import torch
import torch.nn.functional as F
import pathlib
import sys
import contextlib

from birdwinglabel.model import EncTransformer, AutTransformer








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