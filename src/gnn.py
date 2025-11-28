import torch
from torch import nn
import torch.nn.functional as F
import math 
from torch_geometric.nn import GCNConv, global_mean_pool

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
    Multi-head attention mechanism for knot diagrams. This module has a separate
    key and value projection for each neighbor position in the adjacency matrix.
    In particular, the key and value embeddigns of over-out, over-in, under-out,
    and under-in neighbors are all distinct.

    Args:
        input_dim (int): Dimension of input node features.
        d_k (int): Dimension of the key/query vectors.
        heads (int, optional): Number of attention heads. Defaults to 2. The number
            of heads must divide the input_dim since d_v = input_dim / heads.
    """
    def __init__(self, input_dim, d_k, heads=2):
        super().__init__()

        if input_dim % heads != 0:
            raise ValueError("Input dimension must be divisible by number of heads.")
        
        self.input_dim = input_dim
        self.d_k = d_k
        self.d_v = input_dim // heads
        self.heads = heads

        self.w_q = nn.Parameter(torch.empty(heads, input_dim, d_k))
        self.w_k = nn.Parameter(torch.empty(heads, 5, input_dim, d_k))
        self.w_v = nn.Parameter(torch.empty(heads, 5, input_dim, self.d_v))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

    def forward(self, x, adjacency_matrix):
        """
        Multi-head attention forward pass.
        
        Args:
            x: (num_nodes, input_dim)
            adjacency_matrix: (4, num_Nndes)

        Returns:
            Z: (num_nodes, Heads * d_v)
        """
        
        N = x.shape[0]

        # (4, num_nodes, input_dim)
        real_neighbors = x[adjacency_matrix]

        # (1, num_nodes, input_dim)
        self_node = x.unsqueeze(0)
        
        # (5, num_nodes, input_dim)
        x_neighbors = torch.cat([self_node, real_neighbors], dim=0)

        Q = torch.einsum('nd,hdk->hnk', x, self.w_q)
        K = torch.einsum('rnd,hrdk->hrnk', x_neighbors, self.w_k)
        V = torch.einsum('rnd,hrdv->hrnv', x_neighbors, self.w_v)

        A = torch.einsum('hnk,hrnk->hrn', Q, K)/math.sqrt(self.d_k)
        A = F.softmax(A, dim=1)

        Z = torch.einsum('hrn,hrnv->hnv', A, V)

        Z = Z.permute(1, 0, 2).reshape(N, -1)

        return Z

class KnotTransformerLayer(nn.Module):
    """
    Single layer of a Transformer model tailored for knot diagrams.

    This layer implements a relational self-attention mechanism where nodes 
    attend to their neighbors based on specific edge types over/under and in/out,
    followed by a standard position-wise feed-forward network.

    Args:
        input_dim (int): Dimension of the input node features.
            Must be divisible by `heads` to ensure dimension alignment.
        d_k (int): Dimension of the key/query/value vectors per head.
        heads (int, optional): Number of attention heads. Defaults to 2.
        d_ff (int, optional): Hidden dimension of the Feed-Forward network. 
            Defaults to 4 * input_dim.
    """

    def __init__(self, input_dim, d_k, heads=2, d_ff=None):
        super().__init__()

        self.d_ff = 4*input_dim if d_ff is None else d_ff

        self.attention = KnotAttention(input_dim, d_k, heads=heads)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, input_dim)
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x, adjacency_matrix):
        """
        Forward pass of the Knot Transformer Layer.

        The forward pass consists of:
        1. Relational Multi-Head Attention
        2. Residual Connection + Layer Normalization
        3. Position-wise Feed-Forward Network
        4. Residual Connection + Layer Normalization

        Args:
            x (torch.Tensor): Input node features.
                Shape: `(num_nodes, input_dim)`
            adjacency_matrix (torch.Tensor): Indices of the 4 distinct neighbors 
                for each node (excluding self-loop).
                Shape: `(4, num_nodes)`

        Returns:
            torch.Tensor: Updated node features preserving input shape.
                Shape: `(num_nodes, input_dim)`
        """
        attn_output = self.attention(x, adjacency_matrix)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class KnotMultiTransformerClassifier(nn.Module):
    """
    A deep Relational Transformer model for classifying knot diagrams (e.g., Trivial vs. Non-Trivial).

    This model integrates domain-specific positional encodings with a stack of 
    KnotTransformer layers. It aggregates local topological features into a 
    global representation to make a binary classification decision.

    The architecture follows:
    1. **Embedding:** Encodes Dowker sequence positions, over/under and in/out information.
       Note: The encoder produces 2 embeddings per node and
       concatenates them. Therefore, `input_dim` must be even.
    2. **Transformer Stack:** $L$ layers of Relational Multi-Head Attention and 
       Feed-Forward networks.
    3. **Global Pooling:** Adaptive Average Pooling to collapse variable node 
       counts into a fixed-size vector.
    4. **Classifier Head:** A MLP projecting the global vector to logits.

    Args:
        num_layers (int): The number of Transformer layers to stack.
        input_dim (int): The total dimension of the node embeddings ($d_{model}$).
            **Constraint:** Must be divisible by `lcm(2, heads)`. 
            - Divisible by 2 because the encoder concatenates two halves.
            - Divisible by `heads` to split features evenly in attention.
        d_k (int): The dimension of the query/key vectors per head.
        heads (int, optional): Number of attention heads. Defaults to 2.
        d_ff (int, optional): Dimension of the Feed-Forward internal layer. 
            Defaults to 4 * input_dim.
        max_encode_len (int, optional): Max length for positional encoding. Defaults to 500.
        n_encode (int, optional): Frequency base for positional encoding. Defaults to 10000.
    """
    def __init__(self, num_layers, input_dim, d_k, heads=2, d_ff=None, max_encode_len=500, n_encode=10000):
        super().__init__()
        
        # --- Constraints Check ---
        if input_dim % 2 != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by 2 for the PositionalEncoding concatenation.")
        if input_dim % heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by heads ({heads}) for Attention splitting.")

        self.encoder = PositionalEncoding(d_model=input_dim//2, max_len=max_encode_len, n=n_encode)

        self.layers = nn.ModuleList([
            KnotTransformerLayer(input_dim, d_k, heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 2*input_dim),
            nn.ReLU(),
            nn.Linear(2*input_dim, 2) # 0 = Non-Trivial, 1 = Trivial
        )
    
    def forward(self, x, adjacency_matrix):
        """"
        Processes a raw knot diagram representation into classification logits.

        Args:
            x (torch.Tensor): Raw node specifications. 
                Shape: `(2, num_nodes)`
                - Row 0: Dowker sequence positions.
                - Row 1: Over/Under/Sign information (depending on Embedding impl).
            adjacency_matrix (torch.Tensor): Indices of the 4 distinct neighbors 
                for each node.
                Shape: `(4, num_nodes)`

        Returns:
            torch.Tensor: Unnormalized logits for classification.
                Shape: `(1, 2)` (assuming batch size 1 for single graph inference)
                - Index 0: Score for Class 0 (e.g., Non-Trivial)
                - Index 1: Score for Class 1 (e.g., Trivial)
        """
        x = self.encoder(x)

        for layer in self.layers:
            x = layer(x, adjacency_matrix)
            
        global_representation = self.gap(x.unsqueeze(0).permute(0, 2, 1)).squeeze(-1)
        
        logits = self.classifier(global_representation)
        
        return logits

class ValueGCN(nn.Module):
    """
    GCN-based model for binary classification of knot diagrams.
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
        
        return F.sigmoid(x)

