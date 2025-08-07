import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn



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
                # print(f"sample max_indices: {max_indices[0]}")
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
    print(f"Test Error: \nAccuracy per entry: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f}")
    print(f"Accuracy per marker: {true_positive} ({true_positive_rate:>0.1f}%)")
    print(f"Accuracy per frame: {accuracy_per_frame:>0.1f} ({total_frame_correct} / {size})")



def trainandtest(loss_fn, optimizer, model, train_dataloader, test_dataloader, epochs = 10, log_file='train_log.txt'):

    # train and test dataloader are instance of DataLoader using train and test data
    # prepare class to output console ouput in txt while keeping the usual console output
    import sys
    import contextlib

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
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")
        torch.save(model.state_dict(), f'{model.__class__.__name__}_weights.pth')


def train_loop_aut(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layer
    model.train()

    # X: inputs ; y: target
    for batch, (src, tgt, src_pad_mask, tgt_pad_mask) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(src, tgt, src_pad_mask, tgt_pad_mask)
        # print(f"pred shape: {pred.shape}, pred dtype: {pred.dtype}, X sample: {pred[:2]}")
        loss = loss_fn(pred, tgt)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # visualisation of progress
        if batch % 100 == 0:
            loss, current = loss.item(), batch * X.shape[0] + len(X)    # X.shape[0] is batch size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



