""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn
import torch


class GCA(nn.Module):
    """ Global Channel Attention used to replace Fully Connection
    """
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        
        self.conv = nn.Conv2d(in_features, in_features, 3, 1, 1, groups=in_features, bias=False)# k=3, s=1, p=1
        self.norm1 = nn.BatchNorm2d(in_features)
        self.act = nn.ReLU6(inplace=True)
        
        self.linear = nn.Conv2d(in_features, in_features,1,1,0)
        self.norm2 = nn.BatchNorm2d(in_features)
    def forward(self, x):
        
        #x = x + self.norm2(self.linear(self.act(self.norm1(self.conv(x)))))
        x = x + self.linear(self.act(self.norm2(self.conv(self.norm1(x)))))
        return x

