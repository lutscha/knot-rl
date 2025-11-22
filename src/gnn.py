import torch
from torch import nn
import math
import numpy as np
import torch_geometric as pyg

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, n=10000):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.n = n

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(n) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
       
        num_nodes = x.shape[1]//2
        
        node_embeddings = torch.zeros(
            (num_nodes, 2, self.d_model),
            device=x.device,
            dtype=self.pe.dtype
        )

        node_embeddings[x[0], x[1]] = self.pe[:num_nodes*2]

        return node_embeddings.flatten(1)

