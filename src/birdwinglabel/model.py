import torch
from torch import nn

# find device to train nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class BirdEmbedding(nn.Module):
    def __init__(self, input_dim=3, embed_dim=32):
        super().__init__()
        # project 3d coords to embed_dim
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, 3]
        x = self.proj(x)  # [batch_size, seq_len, embed_dim]
        return x

class EncTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, num_layers: int, seq_len: int = 8, dropout: float=0.1, num_class:int = 8):
        super().__init__()

        self.num_class = num_class
        self.seq_len = seq_len
        self.embed = BirdEmbedding(3, embed_dim=embed_dim)

        self.encLayer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encLayer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim)
        self.flatten = nn.Flatten(start_dim=1)  # flatten on 2nd,3rd dim
        self.out = nn.Linear(seq_len * embed_dim, seq_len * num_class)



    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        # turn input [batch_size, seq_len, 3] into embedding [batch_size, seq_len, embed_dim], seq_len = 8
        x = self.embed(inputs)

        # Transpose for MultiheadAttention: [seq_len, batch_size, embed_dim]
        x = x.transpose(0, 1).contiguous()

        x = self.encoder(x)

        x = x.transpose(0, 1)  # embedding transpose to [batch_size, seq_len, embed_dim]
        x = self.flatten(x)
        logits = self.out(x)    # mapped into 8 class prob x 8 markers, flattened
        logits = logits.view(inputs.size(0), self.seq_len, self.num_class)  # unflatten
        return logits


class FCEmbedding(nn.Module):
    def __init__(self, num_marker:int ,in_dim=3, out_dim=32, norm:bool = True):
        super().__init__()
        # project 3d coords to embed_dim
        self.num_marker = num_marker
        self.out_dim = out_dim
        self.proj = nn.Linear(num_marker*in_dim, num_marker*out_dim)
        self.flatten = nn.Flatten(start_dim=1)
        self.norm = norm
        self.ln = nn.LayerNorm(self.out_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, 3]
        batch_size = x.size(0)
        x = self.flatten(x)     # x: [batch_size, seq_len * 3]
        x = self.proj(x)  # [batch_size, seq_len * embed_dim]
        x = x.view(batch_size, self.num_marker, self.out_dim)
        if self.norm:
            x = self.ln(x)
        return x

class LinEmbedding(nn.Module):
    def __init__(self, in_dim=3, out_dim=32, norm:bool = True):
        super().__init__()
        # project 3d coords to embed_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = norm
        self.ln = nn.LayerNorm(self.out_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, 3]
        x = self.proj(x)  # [batch_size, seq_len, embed_dim]
        if self.norm:
            x = self.ln(x)
        return x


class AutTransformer(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 num_head: int = 1,
                 num_encoder_layers: int = 1,
                 num_decoder_layers: int = 1,
                 dim_feedforward: int = 4,
                 dropout: float=0.1,
                 coord_dim:int = 3,
                 tgt_marker:int = 8,
                 src_marker:int = 32,
                 fc_in_embed:bool = True,
                 norm_embed:bool = False
                 ):
        super().__init__()
        self.num_head = num_head
        self.coord_dim = coord_dim
        self.tgt_marker = tgt_marker
        self.src_marker = src_marker
        self.flatten = nn.Flatten(start_dim=1)
        self.norm_embed = norm_embed

        self.src_in_embed_layer = FCEmbedding(num_marker=src_marker, in_dim=coord_dim, out_dim=embed_dim, norm = self.norm_embed)
        self.tgt_in_embed_layer = FCEmbedding(num_marker=tgt_marker, in_dim=coord_dim, out_dim=embed_dim, norm = self.norm_embed)
        self.fc_in_embed = fc_in_embed

        self.src_in_embed_layer_alt = LinEmbedding(coord_dim, embed_dim, norm = self.norm_embed)
        self.tgt_in_embed_layer_alt = LinEmbedding(coord_dim, embed_dim, norm = self.norm_embed)
        self.out_embed_layer = FCEmbedding(num_marker=tgt_marker, in_dim=embed_dim, out_dim=coord_dim,norm=False)
        self.transformer = nn.Transformer(d_model=embed_dim,
                                          nhead=num_head,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)


    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor = None):
        '''
        :param src: dim: [batch, max_marker, 3]
        :param tgt: dim: [batch, 8, 3]
        :param src_mask: [batch, max_marker]
        :param tgt_mask: [batch, 8]
        :return: [batch, 8, 3]
        '''
        if self.fc_in_embed:
            src_embedded = self.src_in_embed_layer(src)     # [batch, max_marker, embed_dim]
            tgt_embedded = self.tgt_in_embed_layer(tgt)     # [batch, 8, embed_dim]
        else:
            src_embedded = self.src_in_embed_layer_alt(src)
            tgt_embedded = self.tgt_in_embed_layer_alt(tgt)

        output = self.transformer(
            src=src_embedded,
            tgt=tgt_embedded,
            src_key_padding_mask=src_mask,  # invert mask: True for padding
            tgt_key_padding_mask=tgt_mask if tgt_mask is not None else None
        )
        # [batch, 8, embed_dim]
        output = self.out_embed_layer(output)        # [batch, 8, embed_dim]
        return output

