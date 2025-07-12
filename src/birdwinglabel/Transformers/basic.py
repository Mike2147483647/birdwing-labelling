import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn

from birdwinglabel.dataprocessing import createtorchdataset, data
from birdwinglabel.dataprocessing.data import full_bilateral_markers

from birdwinglabel.common import trainandtest

# find device to train nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# # create training and test datasets
# seqID_list = data.get_list_of_seqID(full_bilateral_markers)
# print(seqID_list[0:50])
# train_pd_dataframe = (
#     pd.DataFrame(full_bilateral_markers)
#     .pipe(data.subset_by_seqID, seqID_list[0:200])
#     .pipe(data.create_training)
# )
# test_pd_dataframe = (
#     pd.DataFrame(full_bilateral_markers)
#     .pipe(data.subset_by_seqID, seqID_list[200:300])
#     .pipe(data.create_training)
# )
# print(f'{train_pd_dataframe.info()}')
#
# # prepare the torch Datasets from pd dataframes
# train_Dataset = createtorchdataset.HotMarkerDataset(train_pd_dataframe,8)
# test_Dataset = createtorchdataset.HotMarkerDataset(test_pd_dataframe,8)
#
# # put Datasets into DataLoader objects
# batch_size = 50
# train_dataloader = DataLoader(train_Dataset, batch_size=batch_size)
# test_dataloader = DataLoader(test_Dataset, batch_size=batch_size)
#

# adapted from course notes example
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float=0.1):

        # Input x Tensor of shape [seq_len, batch_size, embed_dim]
        # Output: Tensor of shape [seq_len, batch_size, embed_dim]

        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None=None) -> torch.Tensor:
        # x shape: [seq_len, batch_size, embed_dim]
        z, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + self.dropout(z)) # layer norm + residual connection
        mlp_output = self.mlp(x)
        x = self.ln2(x + self.dropout(mlp_output)) # layer norm + residual connection
        return x # [seq_len, batch_size, embed_dim]


# create embedding from 3d coordinates the embedding dimensions
class BirdEmbedding(nn.Module):
    def __init__(self, input_dim=3, embed_dim=32):
        super().__init__()
        # project 3d coords to embed_dim
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, 3]
        x = self.proj(x)  # [batch_size, seq_len, embed_dim]
        return x


# adapted from course notes example
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, num_layers: int, seq_len: int = 8, dropout: float=0.1, num_class:int = 8):
        super().__init__()

        self.num_class = num_class
        self.seq_len = seq_len
        self.embed = BirdEmbedding(3, embed_dim=embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.flatten = nn.Flatten(start_dim=1)  # flatten on 2nd,3rd dim
        self.out = nn.Linear(seq_len * embed_dim, seq_len * num_class)



    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        # turn input [batch_size, seq_len, 3] into embedding [batch_size, seq_len, embed_dim], seq_len = 8
        x = self.embed(inputs)

        # Transpose for MultiheadAttention: [seq_len, batch_size, embed_dim]
        x = x.transpose(0, 1).contiguous()

        # notes: no need for attn_mask in our settings since we model the bird shape as undirect graph
        for layer in self.layers:
            x = layer(x)

        x = self.ln(x)
        # print(f'normalized x shape: {x.shape}')
        x = x.transpose(0, 1)  # embedding transpose to [batch_size, seq_len, embed_dim]
        x = self.flatten(x)
        logits = self.out(x)    # mapped into 8 class prob x 8 markers, flattened
        logits = logits.view(inputs.size(0), self.seq_len, self.num_class)  # unflatten
        return logits


# model = DecoderOnlyTransformer(embed_dim=32, num_heads=8, mlp_dim=128, num_layers=12)
#
# loss = nn.BCEWithLogitsLoss()
# optim = torch.optim.Adam(model.parameters())
# trainandtest.trainandtest(loss, optim, model, train_dataloader, test_dataloader, epochs=8)
