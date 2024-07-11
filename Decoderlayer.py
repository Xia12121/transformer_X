import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from multi-heads import MultiHeadAttention
from feed-forward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        self_attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attention_output))
        cross_attention_output = self.cross_attention(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attention_output))
        feed_forward_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(feed_forward_output))
        return x