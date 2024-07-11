import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)# 这里是两个线性变换，第一个是d_model到d_ff，第二个是d_ff到d_model
        # 输入是d_model，输出是d_model，中间是d_ff
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))