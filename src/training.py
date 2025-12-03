import torch
import torch.nn as nn


from torch.nn.functional import mse_loss
from typing import List, Tuple, Optional, Dict
from collections import deque
import random
from torch_geometric.data import Batch
from torch_geometric.utils import scatter
from data import KnotData
from gnn import AlphaKnot

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
        # detach tensors to ensure they don't keep graph history alive in RAM
        visit_distribution = visit_distribution.detach().cpu()
        self.buffer.append((knot_data, visit_distribution, outcome))
    
    def sample(self, batch_size: int) -> List[Tuple[KnotData, torch.Tensor, float]]:
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
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
    batch_index = batch.batch  # Shape: (N_total_nodes,)
    moves = logits.shape[1]

    # logsumexp
    logits_flat = logits.reshape(-1)
    visit_dist_flat = visit_distributions.reshape(-1)
    segment_index = batch_index.repeat_interleave(moves)

    max_per_graph = scatter(logits_flat, segment_index, dim=0, reduce='max')
    max_per_action = max_per_graph[segment_index]
    exp_shifted = torch.exp(logits_flat - max_per_action)
    sum_exp_per_graph = scatter(exp_shifted, segment_index, dim=0, reduce='sum')
    log_sum_exp_per_action = torch.log(sum_exp_per_graph[segment_index]) # epsilon for safety here?
    
    log_probs_flat = logits_flat - max_per_action - log_sum_exp_per_action

    ce_per_action = -visit_dist_flat * log_probs_flat
    ce_per_graph = scatter(ce_per_action, segment_index, dim=0, reduce='sum')
    policy_loss = ce_per_graph.mean()
    
    value_loss = mse_loss(values.view(-1), outcomes.view(-1))
    
    total_loss = policy_weight * policy_loss + value_weight * value_loss
    
    return total_loss, policy_loss, value_loss

def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    device: torch.device,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    gradient_clipping: bool = True
) -> Tuple[float, float, float]:
    """
    Perform a single training step. Model should already be on device and in training.
    """
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0, 0.0
    
    experiences = replay_buffer.sample(batch_size)
    
    knot_data_list, visit_distributions_list, outcomes_list = zip(*experiences)
    
    batch = Batch.from_data_list(list(knot_data_list))
    batch = batch.to(device)
    
    visit_distributions = torch.cat(visit_distributions_list, dim=0).to(device)
    z_outcomes = torch.tensor(outcomes_list, dtype=torch.float32, device=device)
    
    optimizer.zero_grad()
    
    total_loss, policy_loss, value_loss = compute_loss(
        model, batch, visit_distributions, z_outcomes, policy_weight, value_weight
    )

    total_loss.backward()
    
    if gradient_clipping:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
    optimizer.step()
    
    return total_loss.item(), policy_loss.item(), value_loss.item()

def run_test():
    print("Initializing components...")
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    buffer = ReplayBuffer(capacity=1000)
    model = AlphaKnot(model_dim=64, d_k=16, transformer_layers=4, heads=2, moves=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    model = model.to(device)
    model.train()

    # train loop

if __name__ == "__main__":
    run_test()