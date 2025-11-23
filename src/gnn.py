import torch
from torch import nn
import torch.nn.functional as F
import math 
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool

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
        over and under encodings are concantenated.
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
            x (torch.Tensor): Input tensor of shape (2, num_nodes). 
                - Row 0: Positions in the Dowker sequence.
                - Row 1: Over/under information (0 for over, 1 for under).
                
                Example input for Trefoil:
                [[0, 1, 2, 0, 1, 2],
                 [0, 1, 0, 1, 0, 1]]

        Returns:
            torch.Tensor: Flattened node embeddings of shape (num_nodes, 2 * d_model).
        """

        num_nodes = x.shape[1]//2
        
        node_embeddings = torch.zeros(
            (num_nodes, 2, self.d_model),
            device=x.device,
            dtype=self.pe.dtype
        )

        node_embeddings[x[0], x[1]] = self.pe[:num_nodes*2]

        return node_embeddings.flatten(1)

class KnotAttention(nn.Module):
    """
    Pure pytorch implementation of message passing attention mechanism for knot diagrams.
    """
    def __init__(self, input_dim, d_k, d_v=None, heads=2, num_neighbors=5):
        super().__init__()

        self.input_dim = input_dim
        self.d_k = d_k
        self.d_v = d_v if d_v is not None else input_dim // heads
        self.heads = heads
        self.num_neighbors = num_neighbors

        # TODO: biases?
        self.w_q = nn.Parameter(torch.empty(heads, input_dim, d_k))
        
        self.w_k = nn.Parameter(torch.empty(heads, num_neighbors, input_dim, d_k))
        self.w_v = nn.Parameter(torch.empty(heads, num_neighbors, input_dim, self.d_v))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

    def forward(self, x, adjacency_matrix):
        """
        x: (Num_Nodes, Input_Dim)
        adjacency_matrix: (Num_Neighbors, Num_Nodes)
        """
        
        Q = torch.einsum('nd,hdk->hnk', x, self.w_q)

        x_neighbors = x[adjacency_matrix] # (Num_Neighbors, Num_Nodes, Input_Dim)

        K = torch.einsum('hidk,ind->hink', x_neighbors, self.w_k)

        # TODO: implement rest and fix above
        
        return

class ValueGCN(nn.Module):
    """
    GCN-based model for predicting the value function of a knot diagram state.
    """
    def __init__(self, input_dim, hidden_dim):
        super(ValueGCN, self).__init__()

        self.pe_encoder = PositionalEncoding(d_model=input_dim)

        self.conv1 = GCNConv(input_dim*2, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Graph connectivity (2, num_edges)
            batch: Batch vector mapping each node to a specific graph in the batch
                   (num_nodes,) - needed for pooling
        """
        x = self.pe_encoder(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, None)

        x = self.classifier(x)
        
        return x

