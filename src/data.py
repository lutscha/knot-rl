import torch
from typing import Any
from torch_geometric.data import Data
from torch_geometric.utils import softmax as scatter_softmax


class KnotData(Data):
    """
    Custom Data class for knot diagrams.
    
    Extends PyTorch Geometric's Data class to handle knot-specific attributes
    with custom batching behavior for 'neighbor_index' and 'dowker' fields.
    
    Attributes:
        dowker (torch.Tensor): Dowker sequence representation of the knot.
            Shape: (2 * num_nodes, 2)
            - Column 0: Node indices
            - Column 1: Over/Under flag (0 for Over, 1 for Under)
        neighbor_index (torch.Tensor): Adjacency matrix indices.
            Shape: (num_nodes, 4)
            Each row contains indices for [over_out, over_in, under_out, under_in] neighbors.
        mask (torch.Tensor): Action mask indicating valid/invalid moves.
            Shape: (num_nodes, moves)
            True indicates an invalid move.
        num_nodes (int): Number of nodes (crossings) in the knot diagram.
    """
    
    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any):
        """
        Specify concatenation dimension for batching. Makes sure that the neighbor_index 
        and dowker tensors are concatenated along the first dimension.
        """
        if key == 'neighbor_index' or key == 'dowker':
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any):
        """
        Specify increment for indices when batching.
        Increments the dowker tensor by the number of nodes in the previous graph.
        """
        if key == 'dowker':
            return torch.tensor([[self.num_nodes, 0]])
        return super().__inc__(key, value, *args, **kwargs)


def distribution_from_logits(
    logits: torch.Tensor, 
    ptr: torch.Tensor
):
    """
    Convert logits of possible moves on each graph of a batch into probability distributions.
    
    Computes softmax over all (node, move) pairs per graph, normalizing each graph's
    policy distribution separately.
    
    Args:
        logits: Raw action logits for all nodes in the batch.
            Shape: (N_total, M) where N_total is total nodes across all graphs,
            and M is the number of moves per node.
        ptr: Graph pointer tensor indicating boundaries between graphs in the batch.
            Shape: (batch_size + 1,)
            Contains cumulative node counts: [0, N1, N1+N2, N1+N2+N3, ...]

    TODO: decide exactly in which form to return the outputs
    Returns:
        Tuple containing:
            - probs_flat: Flattened probability distribution over all actions.
                Shape: (N_total * M,)
                Probabilities are normalized per graph.
            - probs_list: List of probability distributions, one per graph.
                Each element has shape (N_i * M,) where N_i is the number of nodes
                in graph i.
                
    Example:
        >>> logits = torch.randn(10, 14)  # 10 nodes, 14 moves
        >>> ptr = torch.tensor([0, 5, 10])  # 2 graphs with 5 nodes each
        >>> probs_flat, probs_list = distribution_from_logits(logits, ptr)
        >>> len(probs_list)  # 2 graphs
        
    """
    N_total, M = logits.shape
    logits_flat = logits.view(-1)
    
    ptr_scaled = ptr * M 
    
    probs_flat = scatter_softmax(logits_flat, ptr=ptr_scaled)
    
    probs_list = [
        probs_flat[ptr_scaled[i] : ptr_scaled[i+1]] 
        for i in range(len(ptr_scaled) - 1)
    ]
    
    return probs_flat, probs_list