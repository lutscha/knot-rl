import torch
from torch_geometric.data import Data
from torch_geometric.utils import softmax as scatter_softmax

class KnotData(Data):
    """
    Custom Data class for knot diagrams.
    """
    
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'neighbor_index' or key == 'dowker':
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'dowker':
            return torch.tensor([[self.num_nodes, 0]])


        
        return super().__inc__(key, value, *args, **kwargs)

def distribution_from_logits(logits, ptr):
    """
    Converts the logits of possible moves on each graph of a batch
    into a list of distributions on the graphs.
    Args:
        logits: (N_total, M) 
        ptr: (B + 1) Graph pointers (e.g. [0, 5, 12...])

    Returns:
        TODO: decide exactly in which form to return the outputs
        
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