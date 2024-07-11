import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        # 这一步用于初始化参数，参数d_model是输入的维度，num_heads是头的数量

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model) # w_o是输出的线性变换

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attention_scores = torch.matmul(Q, K.T) / math.sqrt(self.d_k)
        # T是转置操作，这里是为了计算Q和K的点积
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            # 这里是为了实现mask操作，将mask为0的位置的attention_scores设置为一个很小的数
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        output = torch.matmul(attention_probs, V)# 这里是计算attention_probs和V的点积
        return output

    def split_heads(self, x):
        batch_size, seq_len,d_model = x.size() # x.size()返回的是一个元组，元组的第一个元素是batch_size，第二个元素是seq_len，第三个元素是d_model
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)# 这里是将x的最后一个维度分成两个维度，一个是num_heads，一个是d_k
        
    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        # 这里是将x的num_heads和d_k维度合并成一个维度

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))
        # 这里是将Q，K，V分别通过W_Q，W_K，W_V进行线性变换，然后通过split_heads函数进行分割
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.combine_heads(output)
        output = self.W_O(output)
        return output