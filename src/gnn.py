import torch
from torch import nn
import torch.nn.functional as F
import math
from torch_geometric.nn import global_mean_pool

class BatchPositionalEncoding(nn.Module):
    """
    Implements positional encoding for nodes in a graph diagram representation.

    It treats the Dowker sequence positions as temporal positions and encodes 
    them using sine and cosine functions. It takes a batch of Dowker sequences
    and the batch_ptr vector. The encoding follows the standard formula:

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
    def __init__(self, d_model, max_len=2000, n=10000):
        super().__init__()

        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(n) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    def forward(self, dowker, ptr):
        """
        Encodes a batch of disjoint Dowker sequences into node embeddings.

        This method handles the conversion from global batch indices back to local 
        sequence positions to retrieve the correct positional encodings. It then 
        scatters these encodings into a node-centric format, concatenating the 
        'over' and 'under' embeddings for each node.

        Args:
            dowker (torch.Tensor): The batched Dowker sequence tensor.
                Shape: (2 * N_total, 2)
                - Column 0: Global Node Indices. Because of the custom `__inc__` 
                  in KnotData, these indices are cumulative across the batch 
                  (e.g., if Graph 1 has 5 nodes, Node 0 of Graph 2 is index 5).
                - Column 1: Over/Under flag (0 for Over, 1 for Under).
            ptr (torch.Tensor): The batch pointer tensor from PyG.
                Shape: (Batch_Size + 1,)
                Contains the cumulative count of nodes in the batch (e.g., [0, N1, N1+N2, ...]).
                Used to determine the start and end of each graph in the dense sequence.

        Returns:
            torch.Tensor: The computed embeddings for all nodes in the batch.
                Shape: (N_total, 2 * d_model)
                Each row corresponds to a specific node (in global order) and contains
                the concatenation of its 'Over' positional encoding and 'Under' 
                positional encoding.
        """
        
        N = dowker.size(0) // 2
        
        dowker_starts = ptr[:-1] * 2

        nodes_per_graph = ptr[1:] - ptr[:-1]
        rows_per_graph = nodes_per_graph * 2

        # Provide output_size to prevent CPU-GPU sync
        shifts = dowker_starts.repeat_interleave(
            rows_per_graph,
            output_size=2*N
        )
        
        global_seq = torch.arange(2*N, device=dowker.device)
        local_seq = global_seq - shifts
        
        pe_vecs = self.pe[local_seq]
        
        out_flat = torch.empty(N * 2, self.d_model, device=dowker.device)
        
        # Instead of 2D indexing out[dowker[:,0], dowker[:,1]], we calculate the 1D index.
        # Index = Node_ID * 2 + Over_Under_Flag (0 or 1)
        flat_indices = dowker[:, 0] * 2 + dowker[:, 1]
        
        out_flat[flat_indices] = pe_vecs
        
        # Memory layout [Node0_Over, Node0_Under, Node1_Over...] matches expected flatten behavior
        return out_flat.view(N, 2 * self.d_model)

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
            adjacency_matrix: (num_nodes, 4)

        Returns:
            Z: (num_nodes, Heads * d_v)
        """
        
        N = x.shape[0]

        # (num_nodes, 4, input_dim)
        real_neighbors = x[adjacency_matrix]

        # (num_nodes, 1, input_dim)
        self_node = x.unsqueeze(1)
        
        # (num_nodes, 5, input_dim)
        x_neighbors = torch.cat([self_node, real_neighbors], dim=1)

        Q = torch.einsum('nd,hdk->hnk', x, self.w_q)
        K = torch.einsum('nrd,hrdk->hnrk', x_neighbors, self.w_k)
        V = torch.einsum('nrd,hrdv->hnrv', x_neighbors, self.w_v)

        A = torch.einsum('hnk,hnrk->hnr', Q, K)/math.sqrt(self.d_k)
        A = F.softmax(A, dim=1)

        Z = torch.einsum('hnr,hnrv->hnv', A, V)

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
        d_k (int): Dimension of the query and key vectors per head.
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
                Shape: `(num_nodes, 4)`
        Returns:
            torch.Tensor: Updated node features preserving input shape.
                Shape: `(num_nodes, input_dim)`
        """

        attn_output = self.attention(x, adjacency_matrix)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class AlphaKnot(nn.Module):
    """
    Policy and value network for MCTS. It utilizes the KnotTransformerLayer for
    a common embedding module on which the policy and value heads operate.

    The model processes a batch of disjoint graphs, applies a Transformer-based
    GNN to extract node embeddings, aggregates them for a graph-level value prediction,
    and projects them for node-level action logits.

    Args:
        model_dim (int): Dimension of the node initial embeddings (input features).
            It must satisfy lcm(2, heads) | model_dim due to embedding size considerations.
        d_k (int): Dimension of the key and query vectors in the Multi-Head Attention mechanism.
        transformer_layers (int, optional): Number of stacked KnotTransformerLayer blocks.
            Defaults to 4.
        heads (int, optional): Number of attention heads. Defaults to 2.
            The number of heads should divide the input_dim since d_v = input_dim // heads.
        moves (int, optional): Size of the action space per node (number of output classes).
            Defaults to 14.
        d_ff (int, optional): Dimension of the feed-forward network within the transformer.
            If None, defaults to 4 * model_dim.
        value_dim (int, optional): Hidden dimension size for the Value Head MLP.
            If None, defaults to model_dim * 2.
        policy_dim (int, optional): Hidden dimension size for the Policy Head MLP.
            If None, defaults to model_dim * 2.
    """
    def __init__(self, model_dim, d_k, transformer_layers=4, heads=2, moves = 14, d_ff=None, value_dim=None, policy_dim=None):
        super().__init__()
        
        self.value_dim = value_dim if value_dim is not None else model_dim*2
        self.policy_dim = policy_dim if policy_dim is not None else model_dim*2

        self.encoder = BatchPositionalEncoding(model_dim//2)

        self.transformer_pass = nn.ModuleList([
            KnotTransformerLayer(input_dim=model_dim, d_k=d_k, heads=heads, d_ff=d_ff)
            for _ in range(transformer_layers)
        ])

        self.value_head = nn.Sequential(
            nn.Linear(model_dim, self.value_dim),
            nn.ReLU(),
            nn.Linear(self.value_dim, 1)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(model_dim, self.policy_dim),
            nn.ReLU(),
            nn.Linear(self.policy_dim, moves)
        )

    def forward(self, batch):
        """
        Forward pass for a batch of disjoint graphs.

        Args:
            batch (torch_geometric.data.Batch): A PyG Batch object of KnotData instances.
            It should contain:
                - x (Tensor): Node features of shape (N_total, model_dim).
                - neighbor_index (Tensor): Adjacency/Neighbor indices used by the transformer.
                - mask (BoolTensor): Action mask of shape (N_total, moves). 
                  True indicates an invalid move that should be masked out.
                - batch (LongTensor): Batch vector of shape (N_total,) mapping each 
                  node to its graph index.

        Returns:
            Tuple[Tensor, Tensor]:
                - logits (Tensor): Node-level action logits of shape (N_total, moves).
                  Invalid moves are masked with -inf.
                - values (Tensor): Graph-level value estimates of shape (Batch_Size,).
        """

        dowker = batch.dowker
        neighbor_index = batch.neighbor_index
        mask = batch.mask
        batch_index = batch.batch
        batch_ptr = batch.ptr

        x = self.encoder(dowker, batch_ptr)

        for layer in self.transformer_pass:
            x = layer(x, neighbor_index)
        
        graph_embedding = global_mean_pool(x, batch_index)

        values = self.value_head(graph_embedding).squeeze(-1)

        logits = self.policy_head(x)
        
        logits = logits.masked_fill(mask, float('-inf'))

        return logits, values
