import pandas as pd
import numpy as np
import torch
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
    test_loss, correct, total_labels = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                # y: [batch, 8], pred: [batch, 8, 8]
                pred_labels = pred.argmax(-1)
                correct += (pred_labels == y).sum().item()
                total_labels += y.numel()
            elif isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                # y: [batch, 8, 8], pred: [batch, 8, 8]
                pred_labels = (torch.sigmoid(pred) > 0.5).float()
                correct += (pred_labels == y).sum().item()
                total_labels += y.numel()
            else:
                raise NotImplementedError("Unsupported loss function for accuracy calculation.")

    # print(f"pred shape: {pred.shape}, pred dtype: {pred.dtype}, pred sample: {pred[:5]}")
    # print(f"y shape: {y.shape}, y dtype: {y.dtype}, y sample: {y[:5]}")

    test_loss /= num_batches
    accuracy = 100 * correct / total_labels
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")



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

