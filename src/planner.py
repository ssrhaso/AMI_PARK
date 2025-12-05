import torch
import numpy as np


class CEMPlanner:
    """Cross-Entropy Method for trajectory planning"""
    
    def __init__(self, world_model, cost_function, scalers, 
                 horizon=10, num_samples=1000, num_iterations=3, device='cpu'):
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
        
        state_mean = torch.tensor(self.scalers['state'].mean_, device=self.device, dtype=torch.float32)
        state_std = torch.tensor(self.scalers['state'].scale_, device=self.device, dtype=torch.float32)
        action_mean = torch.tensor(self.scalers['action'].mean_, device=self.device, dtype=torch.float32)
        action_std = torch.tensor(self.scalers['action'].scale_, device=self.device, dtype=torch.float32)
        
        mu = torch.zeros(self.horizon, self.action_dim, device=self.device)
        sigma = torch.ones(self.horizon, self.action_dim, device=self.device)
        
        best_elite_sequences = None
        
        for iteration in range(self.num_iterations):
            noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=self.device)
            action_seqs = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise
            action_seqs = torch.clamp(action_seqs, -2.0, 2.0)
            
            action_seqs_raw = action_seqs * action_std.unsqueeze(0).unsqueeze(0) + action_mean.unsqueeze(0).unsqueeze(0)
            action_seqs_raw = torch.clamp(action_seqs_raw, -1.0, 1.0)
            
            costs = torch.zeros(self.num_samples, device=self.device)
            states = current_state.unsqueeze(0).expand(self.num_samples, -1)
            
            for t in range(self.horizon):
                actions_t = action_seqs_raw[:, t, :]
                
                states_norm = (states - state_mean) / (state_std + 1e-8)
                actions_norm = (actions_t - action_mean) / (action_std + 1e-8)
                
                with torch.no_grad():
                    next_states_norm = self.world_model(states_norm, actions_norm)
                
                next_states = next_states_norm * state_std + state_mean
                next_states = torch.clamp(next_states, -20.0, 20.0)
                
                step_costs = self.cost_function.get_cost(next_states, actions_t)
                costs += step_costs
                
                states = next_states
            
            elite_count = max(1, int(0.1 * self.num_samples))
            _, elite_idx = torch.topk(costs, elite_count, largest=False)
            elite_seqs = action_seqs[elite_idx]
            best_elite_sequences = elite_seqs
            
            mu_new = elite_seqs.mean(dim=0)
            sigma_new = elite_seqs.std(dim=0) + 0.01
            
            mu = 0.9 * mu + 0.1 * mu_new
            sigma = 0.9 * sigma + 0.1 * sigma_new
        
        best_action_norm = best_elite_sequences[0, 0, :]
        best_action = best_action_norm * action_std + action_mean
        best_action = torch.clamp(best_action, -1.0, 1.0)
        
        return best_action.cpu().numpy()
