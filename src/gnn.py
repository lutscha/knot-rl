import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

def mean_pool_index_add(x: torch.Tensor, batch_sizes: torch.Tensor) -> torch.Tensor:
    """
    Efficiently averages embeddings based on variable-length batch segments using index_add_.

    Args:
        x (torch.Tensor): The input tensor of stacked embeddings with shape 
            (N_total, d), where N_total is the sum of all elements in batch_sizes.
        batch_sizes (torch.Tensor): A 1D tensor containing the size of each 
            batch segment (N1, N2, ..., Nb). Shape is (b,).

    Returns:
        torch.Tensor: The pooled embeddings with shape (b, d), where row i 
            is the mean of the corresponding segment in x.
    """
    indices = torch.repeat_interleave(
        torch.arange(len(batch_sizes), device=batch_sizes.device), 
        batch_sizes
    )

    b, d = len(batch_sizes), x.shape[1]
    summed = torch.zeros(b, d, device=x.device, dtype=x.dtype)
    summed.index_add_(0, indices.to(x.device), x)

    output = summed / batch_sizes.to(x.device).view(-1, 1).clamp(min=1e-9)
    
    return output

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
    def __init__(self, input_dim: int, d_k: int, heads: int = 2) -> None:
        """
        Initialize the multi-head attention module.
        
        Args:
            input_dim: Dimension of input node features.
            d_k: Dimension of the key/query vectors.
            heads: Number of attention heads. Defaults to 2. The number of heads
                must divide the input_dim since d_v = input_dim / heads.
                
        Raises:
            ValueError: If input_dim is not divisible by heads.
        """
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

    def reset_parameters(self) -> None:
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Multi-head attention forward pass.
        
        Args:
            x: Input node features.
                Shape: (num_nodes, input_dim)
            adjacency_matrix: Indices of the 4 distinct neighbors for each node.
                Shape: (num_nodes, 4)
                Each row contains [over_out, over_in, under_out, under_in] neighbor indices.

        Returns:
            Output node features after multi-head attention.
                Shape: (num_nodes, heads * d_v)
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
            Must be divisible by heads to ensure dimension alignment.
        d_k (int): Dimension of the query and key vectors per head.
        heads (int, optional): Number of attention heads. Defaults to 2.
        d_ff (int, optional): Hidden dimension of the Feed-Forward network. 
            Defaults to 4 * input_dim.
    """

    def __init__(
        self, 
        input_dim: int, 
        d_k: int, 
        heads: int = 2, 
        d_ff: Optional[int] = None
    ) -> None:
        """
        Initialize the transformer layer.
        
        Args:
            input_dim: Dimension of the input node features.
                Must be divisible by heads to ensure dimension alignment.
            d_k: Dimension of the query and key vectors per head.
            heads: Number of attention heads. Defaults to 2.
            d_ff: Hidden dimension of the Feed-Forward network.
                If None, defaults to 4 * input_dim.
        """
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

    def forward(
        self, 
        x: torch.Tensor, 
        adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Knot Transformer Layer.

        The forward pass consists of:
        1. Relational Multi-Head Attention
        2. Residual Connection + Layer Normalization
        3. Position-wise Feed-Forward Network
        4. Residual Connection + Layer Normalization

        Args:
            x: Input node features.
                Shape: (num_nodes, input_dim)
            adjacency_matrix: Indices of the 4 distinct neighbors for each node
                (excluding self-loop).
                Shape: (num_nodes, 4)
                Each row contains [over_out, over_in, under_out, under_in] neighbor indices.
                
        Returns:
            Updated node features preserving input shape.
                Shape: (num_nodes, input_dim)
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
            It must satisfy heads | model_dim due to embedding size considerations.
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
    def __init__(
        self,
        model_dim: int,
        d_k: int,
        transformer_layers: int = 4,
        heads: int = 2,
        moves: int = 10,
        d_ff: Optional[int] = None,
        value_dim: Optional[int] = None,
        policy_dim: Optional[int] = None,
    ) -> None:
        """
        Initialize the AlphaKnot model.
        
        Args:
            model_dim: Dimension of the node initial embeddings (input features).
                It must satisfy heads | model_dim due to embedding size considerations.
            d_k: Dimension of the key and query vectors in the Multi-Head Attention mechanism.
            transformer_layers: Number of stacked KnotTransformerLayer blocks.
                Defaults to 4.
            heads: Number of attention heads. Defaults to 2.
                The number of heads should divide the input_dim since d_v = input_dim // heads.
            moves: Size of the action space per node (number of output classes).
                Defaults to 14.
            d_ff: Dimension of the feed-forward network within the transformer.
                If None, defaults to 4 * model_dim.
            value_dim: Hidden dimension size for the Value Head MLP.
                If None, defaults to model_dim * 2.
            policy_dim: Hidden dimension size for the Policy Head MLP.
                If None, defaults to model_dim * 2.
        """
        super().__init__()
        
        self.value_dim = value_dim if value_dim is not None else model_dim*2
        self.policy_dim = policy_dim if policy_dim is not None else model_dim*2

        self.linear = nn.Linear(6, model_dim)

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

    # TODO: update so it concats x and neighbor_index
    def forward(self, x, neighbor_index, batch_sizes) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a batch of disjoint graphs.

        Args:
            x (torch.Tensor): Initial (N,6) embedding of the batch.
                Shape: (N_total, 6)
            neighbor_index (torch.Tensor): Neighbor index tensor.
                Shape: (N_total, 4)
            batch_sizes (torch.Tensor): A 1D tensor containing the size of each 
                batch segment (N1, N2, ..., Nb). Shape is (b,).

        Returns:
            Tuple containing:
                - logits: Node-level action logits.
                    Shape: (N_total, moves)
                    Invalid moves are masked with -inf.
                - values: Graph-level value estimates.
                    Shape: (batch_size,)
        """

        x = self.linear(x)

        for layer in self.transformer_pass:
            x = layer(x, neighbor_index)

        graph_embedding = mean_pool_index_add(x, batch_sizes)

        values = self.value_head(graph_embedding).squeeze(-1)

        logits = self.policy_head(x)

        return logits, values

