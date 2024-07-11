import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model) # pe是一个max_seq_length*d_model的矩阵
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # position是一个max_seq_length*1的矩阵
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]