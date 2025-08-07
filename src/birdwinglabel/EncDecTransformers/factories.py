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

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

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


class IdentifyMarkerTransformer(nn.Module):
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
        self.mask = self.transformer.generate_square_subsequent_mask(frame_count)
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























