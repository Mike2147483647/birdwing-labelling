import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pathlib
import sys
import contextlib

from birdwinglabel.EncDecTransformers.factories import IdentifyMarkerTimeDptTransformer, IdentifyMarkerTimeIndptTransformer


# code adapted from https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html 20250703_1555

# loss_fn from nn and optimizer from torch.optim.SGD() eg
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


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

            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                # y: [batch, num_marker], pred: [batch, num_marker, num_labels]
                pred_labels = pred.argmax(-1)
                correct += (pred_labels == y).sum().item()
                total_labels += y.numel()

            elif isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
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
            if isinstance(model, IdentifyMarkerTimeDptTransformer):
                train_loop_aut_seq(train_dataloader, model, loss_fn, optimizer)
                test_loop_aut_seq(test_dataloader, model, loss_fn)
            elif isinstance(model, IdentifyMarkerTimeIndptTransformer):
                train_loop_aut(train_dataloader, model, loss_fn, optimizer, t+1, epochs)
                test_loop_aut(test_dataloader, model, loss_fn)
            else:
                train_loop(train_dataloader, model, loss_fn, optimizer)
                test_loop(test_dataloader, model, loss_fn)
        print("Done!")
        torch.save(model.state_dict(), f'{pathlib.Path(sys.argv[0]).stem}_{model.__class__.__name__}_weights.pth')


def train_loop_aut_seq(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layer
    model.train()
    update_interval = max(1, size // 10)

    for batch, (src, tgt, src_pad_mask, tgt_pad_mask) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(src, tgt, src_pad_mask, tgt_pad_mask)
        # print(f"pred shape: {pred.shape}, pred dtype: {pred.dtype}, X sample: {pred[:2]}")
        # Mask: [batch, frame_count]
        not_padded = ~tgt_pad_mask
        # pred, tgt: [batch, frame_count, num_marker, 3]
        loss = loss_fn(pred[not_padded], tgt[not_padded])

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # visualisation of progress
        current = batch * src.shape[0] + len(src)
        if current % update_interval < src.shape[0]:
            percent = int(100 * current / size)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}] ({percent}%)")


def test_loop_aut_seq(dataloader, model: IdentifyMarkerTimeDptTransformer, loss_fn):
    model.eval()
    num_seq = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    total_frames, within_5, within_10, within_20 = 0, 0, 0, 0
    # print(f'check test dataloader iter {next(iter(dataloader))}')

    for batch, (src, src_pad_mask, seed, gold, gold_pad_mask) in enumerate(dataloader):
        # print(f'in loop now')
        print(f'seed values: {seed}')
        # compute prediction: [batch_size, tgt_length, num_label, 3]
        pred = model.generate_sequence(seed_tgt=seed,src=src,src_key_padding_mask=src_pad_mask)
        print(f'shape of pred: {pred.shape}')
        # compare with gold also [batch_size, tgt_length, num_label, 3], but only compare with non padded entries
        # gold_pad_mask: [batch_size, frame_count], True for padding
        not_padded = ~gold_pad_mask  # [batch_size, frame_count]
        print(f'shape of not_padded indicator: {not_padded.shape}')

        # only [batch_size, tgt_length, 8, 3] is needed since we padded 8 to num_label
        pred = pred[:, :, :8, :]
        gold = gold[:, :, :8, :]

        # only non padded frames are needed
        pred = pred[not_padded]
        gold = gold[not_padded]

        # transform pred to normal scale instead of exp
        if dataloader.dataset.exp_trans:
            pred = torch.log(pred.clamp_min(1e-6))

        # Compute loss only on non-padded frames
        loss = loss_fn(pred, gold)
        test_loss += loss.item()

        print(f'pred first frame: {pred[0,:,:]}')
        print(f'gold first frame: {gold[ 0, :, :]}')

        # Compute per-marker L2 norm
        l2_error = torch.norm(pred - gold, p=2, dim=-1)  # [batch, frame_count, num_marker]
        gold_l2 = torch.norm(gold, p=2, dim=-1)
                   # .clamp(min=1e-8))  # avoid div by zero
        rel_error = l2_error / gold_l2  # relative error


        print(f'gold_l2 shape: {gold_l2.shape} \ngold_l2 value: {gold_l2}')
        print(f'l2_error shape: {l2_error.shape} \nl2_error value: {l2_error}')

        # Compute max relative error per frame (across all markers)
        frame_max_error = rel_error.max(dim=-1).values  # [batch, frame_count]


        # Mask out padded frames
        # frame_max_error = frame_max_error[not_padded]
        print(f'shape of frame_max_error: {frame_max_error.shape}')
        print(f'frame_max_error: {frame_max_error}')

        total_frames += frame_max_error.numel()
        within_5 += (frame_max_error < 0.05).sum().item()
        within_10 += (frame_max_error < 0.10).sum().item()
        within_20 += (frame_max_error < 0.20).sum().item()


    print(f'''
    total number of frames testing: {total_frames}
    number of frames <5% error: {within_5}
    number of frames <10% error: {within_10}
    number of frames <20% error: {within_20}
    ''')

    print(f'''
    Average loss per sequence: {test_loss / num_seq:.6f}
    Average loss per frame: {test_loss / total_frames:.6f}
    Proportion of frames that has <5% error: {within_5 / total_frames:.6f}
    Proportion of frames that has <10% error: {within_10 / total_frames:.6f}
    Proportion of frames that has <20% error: {within_20 / total_frames:.6f}
    ''')
    # input(f'press Enter to move onto next epoch: ')

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








