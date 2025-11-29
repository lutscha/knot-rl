import math
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import softmax as scatter_softmax


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for nodes in a graph diagram representation.

    It treats the Dowker sequence positions as temporal positions and encodes 
    them using sine and cosine functions. The encoding follows the standard 
    formula:

    $$
    P(k, 2i) = \\sin\\left(\\frac{k}{n^{2i/d_{model}}}\\right)
    $$
    $$
    P(k, 2i+1) = \\cos\\left(\\frac{k}{n^{2i/d_{model}}}\\right)
    $$

    Args:
        d_model (int): Dimension of the model (embedding size). Note that since each
        node appears twice, their final embedding will be of size 2 * d_model as the
        over and under encodings are concantenated. d_model should be even.
        max_len (int, optional): Maximum length of Dowker sequence. Should be 
            2 * max crossings expected. Defaults to 500.
        n (int, optional): Base for the positional encoding frequency. 
            Defaults to 10000.

    Attributes:
        pe (torch.Tensor): The learnable positional encoding buffer of shape 
            (max_len, d_model).
    """
    def __init__(self, d_model, max_len=500, n=10000):
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
        """Encodes the input tensor using the pre-computed positional buffer.

        Args:
            x (torch.Tensor): Input tensor of shape (2 * num_nodes, 2). 
                - Column 0: Positions in the Dowker sequence.
                - Column 1: Over/under information (0 for over, 1 for under).
                
                Example input for Trefoil:
                [[0, 0],
                 [1, 1],
                 [2, 0],
                 [0, 1],
                 [1, 0],
                 [2, 1]]

        Returns:
            torch.Tensor: Flattened node embeddings of shape (num_nodes, 2 * d_model).
        """

        num_nodes = x.shape[0]//2
        
        node_embeddings = torch.zeros(
            (num_nodes, 2, self.d_model),
            device=x.device,
            dtype=self.pe.dtype
        )

        node_embeddings[x[:, 0], x[:, 1]] = self.pe[:num_nodes*2]

        return node_embeddings.flatten(1)

class KnotData(Data):
    '''
    Custom Data class for knot diagrams, extending PyG's Data class.
    Inherits all standard attributes and methods, with custom concatenation behavior for
    neighbor_index, making sure it concatenates along dim=0.

    Args:
        x (torch.Tensor): Node feature matrix. (num_nodes, input_dim)
        neighbor_index (torch.Tensor): Indices of the 4 distinct neighbors for each node. (num_nodes, 4)
        mask (torch.Tensor): Mask tensor for actions at each node. (num_nodes, M)
    '''
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'neighbor_index':
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

def distribution_from_logits(logits, ptr):
    """
    Args:
        logits: (N_total, M) 
        ptr: (B + 1) Graph pointers (e.g. [0, 5, 12...])
    """
    N_total, M = logits.shape
    logits_flat = logits.view(-1)
    
    ptr_scaled = ptr * M 
    
    probs_flat = scatter_softmax(logits_flat, ptr=ptr_scaled)
    
    # TODO: see what format the MCTS will ask for the policies
    probs_list = [
        probs_flat[ptr_scaled[i] : ptr_scaled[i+1]] 
        for i in range(len(ptr_scaled) - 1)
    ]
    
    return probs_flat, probs_list