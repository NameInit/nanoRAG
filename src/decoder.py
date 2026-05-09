import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .utils import get_positional_encoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout(attn1))

        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        out2 = self.layernorm2(out1 + self.dropout(attn2))

        ffn_output = self.ffn(out2)
        out3 = self.layernorm3(out2 + self.dropout(ffn_output))

        return out3


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_pos, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = get_positional_encoding(max_pos, d_model)
        
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dff, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.size(1)
        
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:, :seq_len, :].to(x.device)
        
        x = self.dropout(x)

        for layer in self.dec_layers:
            x = layer(x, enc_output, look_ahead_mask, padding_mask)

        return x
