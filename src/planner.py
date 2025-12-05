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

    def denormalize_delta(self, delta_norm):
        """Convert normalized delta to raw delta"""
        delta_mean = torch.tensor(
            self.scalers['delta'].mean_,
            device=self.device,
            dtype=torch.float32
        )
        delta_scale = torch.tensor(
            self.scalers['delta'].scale_,
            device=self.device,
            dtype=torch.float32
        )
        return delta_norm * delta_scale + delta_mean

    def plan(self, current_state):
        """Plan using Delta Dynamics"""
        
        # Setup scalers
        state_mean = torch.tensor(self.scalers['state'].mean_, device=self.device, dtype=torch.float32)
        state_scale = torch.tensor(self.scalers['state'].scale_, device=self.device, dtype=torch.float32)
        action_mean = torch.tensor(self.scalers['action'].mean_, device=self.device, dtype=torch.float32)
        action_scale = torch.tensor(self.scalers['action'].scale_, device=self.device, dtype=torch.float32)

        # Initialize Distribution
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.ones(self.horizon, self.action_dim, device=self.device)

        # Prepare Initial State
        current_state_raw = torch.tensor(current_state, device=self.device, dtype=torch.float32)
        
        for i in range(self.iterations):
            # Sample Actions (Normalized Space)
            action_seqs = torch.normal(
                mean.expand(self.num_samples, -1, -1),
                std.expand(self.num_samples, -1, -1)
            )
            action_seqs = torch.clamp(action_seqs, -2.0, 2.0)  # Allow wider range
            
            # Convert to Raw Actions
            action_seqs_raw = action_seqs * action_scale + action_mean
            action_seqs_raw = torch.clamp(action_seqs_raw, -1.0, 1.0)
            
            costs = torch.zeros(self.num_samples, device=self.device)
            state_raw = current_state_raw.expand(self.num_samples, -1)
            
            for t in range(self.horizon):
                action_raw = action_seqs_raw[:, t, :]
                
                # Normalize for model
                state_norm = (state_raw - state_mean) / state_scale
                action_norm = (action_raw - action_mean) / action_scale
                
                with torch.no_grad():
                    # Model predicts DELTA
                    delta_norm = self.world_model(state_norm, action_norm)
                
                # Denormalize delta
                delta_raw = self.denormalize_delta(delta_norm)
                
                # INTEGRATE: Next = Current + Delta
                next_state_raw = state_raw + delta_raw
                
                # Clip to valid range
                next_state_raw = torch.clamp(next_state_raw, -20.0, 20.0)
                
                # Calculate cost
                step_cost = self.cost_function.get_cost(next_state_raw, action_raw)
                costs += step_cost
                
                state_raw = next_state_raw
            
            # Elite selection
            k = max(1, int(0.1 * self.num_samples))
            top_costs, top_indices = torch.topk(costs, k, largest=False)
            top_actions = action_seqs[top_indices]
            
            # Update distribution
            new_mean = top_actions.mean(dim=0)
            new_std = top_actions.std(dim=0) + 0.01  # Add small noise for exploration
            
            mean = 0.5 * new_mean + 0.5 * mean
            std = 0.5 * new_std + 0.5 * std
            
        # Return best action
        best_action_raw = action_seqs_raw[top_indices, 0, :].cpu().numpy()
        return best_action_raw
