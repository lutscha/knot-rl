import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from collections import deque
import random
from torch_geometric.data import Batch
from torch_geometric.utils import scatter
from .data import KnotData


class ReplayBuffer:
    """
    Replay buffer for storing (KnotData, VisitDistribution, Outcome) tuples.
    
    The buffer stores experience tuples collected from MCTS:
    - KnotData: The state (knot diagram) at a particular point in the search
    - VisitDistribution: The policy distribution from MCTS visit counts, 
      flattened to match logits shape (N_total, moves)
    - Outcome: The final value/reward from the MCTS search (scalar)
    
    Args:
        capacity (int): Maximum number of experience tuples to store.
            When full, oldest entries are removed (FIFO).
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
    
    def add(self, knot_data: KnotData, visit_distribution: torch.Tensor, outcome: float):
        """
        Add an experience tuple to the replay buffer.
        
        Args:
            knot_data (KnotData): The knot diagram state.
            visit_distribution (torch.Tensor): Policy distribution from MCTS.
                Shape: (num_nodes, moves) - flattened probability distribution
                over all (node, move) pairs for this graph.
            outcome (float): The final value/reward from MCTS (typically in [-1, 1]).
        """
        self.buffer.append((knot_data, visit_distribution, outcome))
    
    def sample(self, batch_size: int) -> List[Tuple[KnotData, torch.Tensor, float]]:
        """
        Sample a batch of experience tuples uniformly at random.
        
        Args:
            batch_size (int): Number of samples to return.
            
        Returns:
            List of (KnotData, VisitDistribution, Outcome) tuples.
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return the current size of the replay buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all entries from the replay buffer."""
        self.buffer.clear()


def compute_loss(
    model: nn.Module,
    batch: Batch,
    visit_distributions: torch.Tensor,
    outcomes: torch.Tensor,
    policy_weight: float = 1.0,
    value_weight: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute policy and value losses for a batch of experiences.
    
    Args:
        model (nn.Module): The AlphaKnot model.
        batch (Batch): Batched KnotData instances.
        visit_distributions (torch.Tensor): Target policy distributions from MCTS.
            Shape: (N_total, moves) - flattened over all nodes in batch.
        outcomes (torch.Tensor): Target values from MCTS.
            Shape: (batch_size,) - one value per graph in batch.
        policy_weight (float): Weight for policy loss. Defaults to 1.0.
        value_weight (float): Weight for value loss. Defaults to 1.0.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_loss: Combined weighted loss
            - policy_loss: Cross-entropy loss between predicted and target policies
            - value_loss: MSE loss between predicted and target values
    """
    logits, values = model(batch)
    
    batch_index = batch.batch
    moves = logits.shape[1]
    
    logits_flat = logits.view(-1)
    visit_dist_flat = visit_distributions.view(-1)
    segment_index = batch_index.repeat_interleave(moves)  # (N_total * moves,)
    
    # compute log_softmax per graph using numerically stable log-sum-exp trick
    # log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    # this avoids the numerical issues
    
    max_per_graph = scatter(logits_flat, segment_index, dim=0, reduce='max')
    max_per_action = max_per_graph[segment_index]
    
    exp_shifted = torch.exp(logits_flat - max_per_action)
    sum_exp_per_graph = scatter(exp_shifted, segment_index, dim=0, reduce='sum')
    log_sum_exp_per_action = torch.log(sum_exp_per_graph[segment_index])
    log_probs_flat = logits_flat - max_per_action - log_sum_exp_per_action
    
    # compute cross-entropy element-wise: -target * log(predicted)
    ce_per_action = -visit_dist_flat * log_probs_flat
    
    # sum CE per graph
    ce_per_graph = scatter(ce_per_action, segment_index, dim=0, reduce='sum')
    
    policy_loss = ce_per_graph.mean()
    value_loss = nn.functional.mse_loss(values, outcomes)
    
    # total loss
    total_loss = policy_weight * policy_loss + value_weight * value_loss
    
    return total_loss, policy_loss, value_loss