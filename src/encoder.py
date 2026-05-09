import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .utils import get_positional_encoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_output = self.mha(x, x, x, mask) 
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_pos, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = get_positional_encoding(max_pos, d_model)
        
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        seq_len = x.size(1)
        
        x = self.embedding(x) 
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        x += self.pos_encoding[:, :seq_len, :].to(x.device)
        
        x = self.dropout(x)

        for layer in self.enc_layers:
            x = layer(x, mask)

        return x