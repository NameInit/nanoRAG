import torch
import torch.nn as nn
import numpy as np

def get_positional_encoding(max_seq_len, d_model):
    pos_enc = np.zeros((max_seq_len, d_model))
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

    return torch.FloatTensor(pos_enc).unsqueeze(0)

def create_masks(inp, tar):
    enc_padding_mask = (inp == 0).unsqueeze(1).unsqueeze(2).float()
    
    dec_padding_mask = (inp == 0).unsqueeze(1).unsqueeze(2).float()

    size = tar.size(1)
    look_ahead_mask = torch.triu(torch.ones(size, size), diagonal=1).to(tar.device)
    
    dec_target_padding_mask = (tar == 0).unsqueeze(1).unsqueeze(2).float()
    combined_mask = torch.max(look_ahead_mask, dec_target_padding_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask