import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 input_vocab_size, target_vocab_size, max_pos, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, max_pos, dropout)
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size, max_pos, dropout)

        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, tar, enc_mask, look_ahead_mask, dec_mask):
        enc_output = self.encoder(inp, enc_mask) 
        
        dec_output = self.decoder(tar, enc_output, look_ahead_mask, dec_mask)
        
        final_output = self.final_layer(dec_output)
        
        return final_output
