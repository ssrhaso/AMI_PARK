import torch
import numpy as np

class CEMPlanner:
    """
    Standard CEM Planner (Clean & Stable)
    """
    def __init__(self, world_model, cost_function, scalers, 
                 horizon=10, num_samples=1000, num_iterations=5, device='cpu'):
        self.world_model = world_model
        self.cost_function = cost_function
        self.scalers = scalers
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.device = device
        self.action_dim = 2
    
    def plan(self, current_state):
        current_state = torch.tensor(current_state, dtype=torch.float32, device=self.device)
        
        # Scalers
        state_mean = torch.tensor(self.scalers['state'].mean_, device=self.device, dtype=torch.float32)
        state_std = torch.tensor(self.scalers['state'].scale_, device=self.device, dtype=torch.float32)
        action_mean = torch.tensor(self.scalers['action'].mean_, device=self.device, dtype=torch.float32)
        action_std = torch.tensor(self.scalers['action'].scale_, device=self.device, dtype=torch.float32)
        
        # Initialize Distribution (Standard)
        mu = torch.zeros(self.horizon, self.action_dim, device=self.device)
        sigma = torch.ones(self.horizon, self.action_dim, device=self.device)
        
        best_elite_sequences = None
        
        for iteration in range(self.num_iterations):
            # Sample
            noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=self.device)
            action_seqs = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise
            action_seqs = torch.clamp(action_seqs, -2.0, 2.0)
            
            # Denormalize
            action_seqs_raw = action_seqs * action_std + action_mean
            action_seqs_raw = torch.clamp(action_seqs_raw, -1.0, 1.0)
            
            # Rollout
            costs = torch.zeros(self.num_samples, device=self.device)
            states = current_state.unsqueeze(0).expand(self.num_samples, -1)
            
            for t in range(self.horizon):
                actions_t = action_seqs_raw[:, t, :]
                
                states_norm = (states - state_mean) / (state_std + 1e-8)
                actions_norm = (actions_t - action_mean) / (action_std + 1e-8)
                
                with torch.no_grad():
                    next_states_norm = self.world_model(states_norm, actions_norm)
                
                next_states = next_states_norm * state_std + state_mean
                
                # Cost
                step_costs = self.cost_function.get_cost(next_states, actions_t)
                costs += step_costs
                states = next_states
            
            # Elites
            elite_count = max(1, int(0.1 * self.num_samples))
            _, elite_idx = torch.topk(costs, elite_count, largest=False)
            elite_seqs = action_seqs[elite_idx]
            best_elite_sequences = elite_seqs
            
            # Update
            mu_new = elite_seqs.mean(dim=0)
            sigma_new = elite_seqs.std(dim=0) + 0.1
            
            momentum = 0.8
            mu = momentum * mu + (1 - momentum) * mu_new
            sigma = momentum * sigma + (1 - momentum) * sigma_new
            
        # Return best action
        best_action_norm = best_elite_sequences[0, 0, :]
        best_action = best_action_norm * action_std + action_mean
        best_action = torch.clamp(best_action, -1.0, 1.0)
        
        return best_action.cpu().numpy()
