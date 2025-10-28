import torch
from torch import nn

# find device to train nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")