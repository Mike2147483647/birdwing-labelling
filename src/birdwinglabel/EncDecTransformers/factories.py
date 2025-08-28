import torch
from sympy.logic.boolalg import Boolean
from torch import nn
import math


# find device to train nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# linear learnt position encoding
class LinearPosEnc(nn.Module):
    def __init__(self, mode: int = 0, max_marker: int = 32, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.mode = mode
        self.linear = nn.Linear(max_marker * 3, max_marker * 3)
        # mode 0 (default): all posEnc are learnt
        # mode 1: add 100 per frame
        # otherwise: nothing

    def forward(self, x: torch.Tensor):
        # x: [batch_size, frame_count, max_marker, 3]
        batch_size, frame_count, max_marker, xyz = x.shape
        x = x.reshape(batch_size, frame_count, max_marker * xyz)
        # x: [batch_size, frame_count, max_marker * 3]

        if self.mode == 0:
            x = self.linear(x)
            return self.dropout(x)
        elif self.mode == 1:
            offset = torch.arange(frame_count, device=x.device).view(1, frame_count, 1) * 100
            x = x + offset
            return x


# frequential posEnc from Attention is all you need, code from https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch accessed 20250806_1901
class FrequentialPosEnc(nn.Module):

    def __init__(self, max_marker: int = 32, xyz: int = 3, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        d_model = max_marker * xyz

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, frame_count, max_marker, 3]
        batch_size, frame_count, max_marker, xyz = x.shape
        x = x.reshape(batch_size, frame_count, max_marker * xyz)
        # x: [batch_size, frame_count, max_marker * 3]

        x = x.transpose(0, 1).contiguous()
        # x: [frame_count, batch_size, max_marker * 3]
        x = x + self.pe[:x.size(0)]
        x = x.transpose(0, 1).contiguous()
        # x: [batch_size, frame_count, max_marker * 3]
        return self.dropout(x)


class IdentifyMarkerTimeDptTransformer(nn.Module):
    def __init__(
            self,
            pos_enc: nn.Module,
            max_marker: int,
            frame_count: int,
            num_head:int = 1,
            num_encoder_layers:int = 1,
            num_decoder_layers:int = 1,
            dim_feedforward:int = 4,
            dropout: float = 0.1,
            backward: Boolean = False
    ):
        super().__init__()
        self.pos_enc = pos_enc
        self.transformer = nn.Transformer(
            d_model=max_marker*3,
            nhead=num_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.frame_count = frame_count
        self.max_marker = max_marker
        self.mask = self.transformer.generate_square_subsequent_mask(self.frame_count)
        self.backward = backward
        if self.backward:
            self.mask = torch.flip(self.mask, dims=[0,1])



    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
        # src, tgt: [batch_size, frame_count, max_marker, 3]
        batch_size, frame_count, num_label, xyz = tgt.shape
        # pos enc
        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt)
        # src, tgt: [batch_size, frame_count, max_marker * 3]
        # transformer

        # combine causal mask and padding mask
        # reverse order of padding indicator if backward causal
        if self.backward:
            src_key_padding_mask = torch.flip(src_key_padding_mask, dims=[1])
            tgt_key_padding_mask = torch.flip(tgt_key_padding_mask, dims=[1])

        # Convert key padding mask to float and expand to [batch_size, 1, seq_len]
        src_key_padding = src_key_padding_mask.unsqueeze(1).to(torch.float32) * float('-inf')
        tgt_key_padding = tgt_key_padding_mask.unsqueeze(1).to(torch.float32) * float('-inf')

        # Add to self.mask (broadcast to [batch_size, seq_len, seq_len])
        src_combined_mask = self.mask.unsqueeze(0) + src_key_padding
        tgt_combined_mask = self.mask.unsqueeze(0) + tgt_key_padding

        output = self.transformer(
            src=src,
            tgt=tgt,
            src_mask = src_combined_mask,
            tgt_mask = tgt_combined_mask,
            src_is_causal=True, 
            tgt_is_causal=True
        )
        # output: [batch_size, frame_count, max_marker * 3]
        # reshape

        output = output.reshape(batch_size, frame_count, num_label, xyz)
        return output
        # output: [batch_size, frame_count, 8, 3]


    def generate_sequence(self, seed_tgt: torch.Tensor, src, src_key_padding_mask):
        # seed_tgt: [batch_size, num_label, 3]
        # num_label should be same as training

        tgt_length = self.frame_count
        batch_size, num_label, xyz = seed_tgt.shape

        # pad seed so that its [max_marker, 3]
        padded_seed = torch.zeros((batch_size, self.max_marker, xyz), dtype=seed_tgt.dtype, device=seed_tgt.device)
        padded_seed[:, :num_label, :] = seed_tgt

        tgt_tensor = torch.zeros((batch_size, tgt_length, self.max_marker, 3), dtype=torch.float32)
        tgt_padding_mask = torch.zeros(tgt_length, dtype=torch.float32)     # dummy mask

        tgt_tensor[:,0,:,:] = padded_seed.to(torch.float32)
        for current_time in range(1,tgt_length):
            static_pred = self.forward(
                src = src,
                src_key_padding_mask = src_key_padding_mask,
                tgt = tgt_tensor,
                tgt_key_padding_mask= tgt_padding_mask
            )
            tgt_tensor[:,current_time, :,:] = static_pred[:,current_time,:,:]
        # tgt_tensor [batch_size, tgt_length, num_label, 3]
        return tgt_tensor


class BirdEmbedding(nn.Module):
    def __init__(self, num_marker:int ,in_dim=3, out_dim=32):
        super().__init__()
        # project 3d coords to embed_dim
        self.num_marker = num_marker
        self.out_dim = out_dim
        self.proj = nn.Linear(num_marker*in_dim, num_marker*out_dim)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        # x: [batch_size, seq_len, 3]
        batch_size = x.size(0)
        x = self.flatten(x)     # x: [batch_size, seq_len * 3]
        x = self.proj(x)  # [batch_size, seq_len * embed_dim]
        x = x.view(batch_size, self.num_marker, self.out_dim)
        return x


class IdentifyMarkerTimeIndptTransformer(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 num_head: int = 1,
                 num_encoder_layers: int = 1,
                 num_decoder_layers: int = 1,
                 dim_feedforward: int = 4,
                 dropout: float=0.1,
                 coord_dim:int = 3,
                 tgt_marker:int = 8,
                 src_marker:int = 32
                 ):
        super().__init__()
        self.num_head = num_head
        self.coord_dim = coord_dim
        self.tgt_marker = tgt_marker
        self.src_marker = src_marker
        self.flatten = nn.Flatten(start_dim=1)
        self.src_in_embed_layer = BirdEmbedding(num_marker=src_marker, in_dim=coord_dim, out_dim=embed_dim)
        self.tgt_in_embed_layer = BirdEmbedding(num_marker=tgt_marker, in_dim=coord_dim, out_dim=embed_dim)
        self.out_embed_layer = BirdEmbedding(num_marker=tgt_marker, in_dim=embed_dim, out_dim=coord_dim)
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

        src_embedded = self.src_in_embed_layer(src)     # [batch, max_marker, embed_dim]
        tgt_embedded = self.tgt_in_embed_layer(tgt)     # [batch, 8, embed_dim]
        output = self.transformer(
            src=src_embedded,
            tgt=tgt_embedded,
            src_key_padding_mask=src_mask,  # invert mask: True for padding
            tgt_key_padding_mask=tgt_mask if tgt_mask is not None else None
        )
        # [batch, 8, embed_dim]
        output = self.out_embed_layer(output)        # [batch, 8, embed_dim]
        return output














