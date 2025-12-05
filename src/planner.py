import torch
import numpy as np

class CEMPlanner:
    def __init__(
        self, 
        world_model, 
        cost_function, 
        scalers, 
        horizon=10, 
        num_samples=1000, 
        iterations=5, 
        device="cpu"
    ):
        self.world_model = world_model
        self.cost_function = cost_function
        self.scalers = scalers
        self.horizon = horizon
        self.num_samples = num_samples
        self.iterations = iterations
        self.device = device
        self.action_dim = 2

    def normalize_batch(self, states, actions):
        """Normalize batch of states and actions"""
        state_mean = torch.tensor(self.scalers['state'].mean_, device=self.device, dtype=torch.float32)
        state_scale = torch.tensor(self.scalers['state'].scale_, device=self.device, dtype=torch.float32)
        action_mean = torch.tensor(self.scalers['action'].mean_, device=self.device, dtype=torch.float32)
        action_scale = torch.tensor(self.scalers['action'].scale_, device=self.device, dtype=torch.float32)
        
        states_norm = (states - state_mean) / state_scale
        actions_norm = (actions - action_mean) / action_scale
        return states_norm, actions_norm

    def denormalize_delta(self, delta_norm):
        """Convert normalized delta to raw delta"""
        delta_mean = torch.tensor(self.scalers['delta'].mean_, device=self.device, dtype=torch.float32)
        delta_scale = torch.tensor(self.scalers['delta'].scale_, device=self.device, dtype=torch.float32)
        return delta_norm * delta_scale + delta_mean

    def plan(self, current_state):
        """Plan using Delta Dynamics"""
        
        # 1. Setup tensors for Scalers
        state_mean = torch.tensor(self.scalers['state'].mean_, device=self.device, dtype=torch.float32)
        state_scale = torch.tensor(self.scalers['state'].scale_, device=self.device, dtype=torch.float32)
        action_mean = torch.tensor(self.scalers['action'].mean_, device=self.device, dtype=torch.float32)
        action_scale = torch.tensor(self.scalers['action'].scale_, device=self.device, dtype=torch.float32)

        # 2. Initialize Distribution (Mean=0, Std=1)
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.ones(self.horizon, self.action_dim, device=self.device)

        # 3. Prepare Initial State (Raw)
        current_state_raw = torch.tensor(current_state, device=self.device, dtype=torch.float32)
        
        for i in range(self.iterations):
            # Sample Actions (Normalized Space)
            action_seqs = torch.normal(
                mean.expand(self.num_samples, -1, -1),
                std.expand(self.num_samples, -1, -1)
            )
            action_seqs = torch.clamp(action_seqs, -1.0, 1.0)
            
            # Convert to Raw Actions for Cost/Dynamics
            action_seqs_raw = action_seqs * action_scale + action_mean
            action_seqs_raw = torch.clamp(action_seqs_raw, -1.0, 1.0)
            
            costs = torch.zeros(self.num_samples, device=self.device)
            state_raw = current_state_raw.expand(self.num_samples, -1)
            
            for t in range(self.horizon):
                action_raw = action_seqs_raw[:, t, :]
                
                # Prepare inputs for Model (Normalize)
                state_norm = (state_raw - state_mean) / state_scale
                action_norm = (action_raw - action_mean) / action_scale
                
                with torch.no_grad():
                    # Predict DELTA (Normalized)
                    delta_norm = self.world_model(state_norm, action_norm)
                
                # Denormalize Delta
                delta_raw = self.denormalize_delta(delta_norm)
                
                # INTEGRATE DYNAMICS: Next = Current + Delta
                next_state_raw = state_raw + delta_raw
                
                # Calculate Cost
                step_cost = self.cost_function.get_cost(next_state_raw, action_raw)
                costs += step_cost
                
                # Update state for next step
                state_raw = next_state_raw
            
            # Elite Selection
            k = max(1, int(0.1 * self.num_samples))
            top_costs, top_indices = torch.topk(costs, k, largest=False)
            top_actions = action_seqs[top_indices]
            
            # Update Distribution
            new_mean = top_actions.mean(dim=0)
            new_std = top_actions.std(dim=0)
            
            mean = 0.7 * new_mean + 0.3 * mean
            std = 0.7 * new_std + 0.3 * std
            
        # Return best action (Raw)
        best_action_raw = action_seqs_raw[top_indices[0], 0, :].cpu().numpy()
        return best_action_raw
