""" PLANNER MODULE  - USING CEM WITH NORMALISED DATA TO PLAN ACTIONS """
import torch
import numpy as np


class CEMPlanner:
    def __init__(
        self,
        world_model,
        cost_function,
        scalers,
        horizon : int = 30,
        num_samples : int = 256,
        iterations : int = 4,
        device : str = "cpu",
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
        
        state_mean = torch.tensor(
            self.scalers['state'].mean_,
            device=self.device,
            dtype=torch.float32
        )
        state_scale = torch.tensor(
            self.scalers['state'].scale_,
            device=self.device,
            dtype=torch.float32
        )
        action_mean = torch.tensor(
            self.scalers['action'].mean_,
            device=self.device,
            dtype=torch.float32
        )
        action_scale = torch.tensor(
            self.scalers['action'].scale_,
            device=self.device,
            dtype=torch.float32
        )
        
        states_norm = (states - state_mean) / state_scale
        actions_norm = (actions - action_mean) / action_scale
        
        return states_norm, actions_norm

    def denormalize_action(self, action_norm):
        """Convert normalized action to raw space"""
        
        action_mean = torch.tensor(
            self.scalers['action'].mean_,
            device=self.device,
            dtype=torch.float32
        )
        action_scale = torch.tensor(
            self.scalers['action'].scale_,
            device=self.device,
            dtype=torch.float32
        )
        action_raw = action_norm * action_scale + action_mean
        action_raw = torch.clamp(action_raw, -1.0, 1.0)
        return action_raw


    
    def normalize(self, state, action):
        """CONVERT RAW STATE AND ACTION TO NORMALISED TENSORS"""
        
        state_mean = torch.tensor(
            self.scalers['state'].mean_,  # ← CHANGED from .mean to .mean_
            device=self.device,
            dtype=torch.float32
        )
        state_scale = torch.tensor(
            self.scalers['state'].scale_,  # ← CHANGED from .scale to .scale_
            device=self.device,
            dtype=torch.float32
        )
        action_mean = torch.tensor(
            self.scalers['action'].mean_,  # ← CHANGED from .mean to .mean_
            device=self.device,
            dtype=torch.float32
        )
        action_scale = torch.tensor(
            self.scalers['action'].scale_,  # ← CHANGED from .scale to .scale_
            device=self.device,
            dtype=torch.float32
        )
        
        state_norm = (state - state_mean) / state_scale
        action_norm = (action - action_mean) / action_scale
        
        return state_norm, action_norm

    
    def denormalize_state(self, state_norm):
        """Convert normalized prediction tensor to raw state"""
        
        state_mean = torch.tensor(
            self.scalers['state'].mean_,  # ← CHANGED from .mean to .mean_
            device=self.device,
            dtype=torch.float32
        )
        state_scale = torch.tensor(
            self.scalers['state'].scale_,  # ← CHANGED from .scale to .scale_
            device=self.device,
            dtype=torch.float32
        )
        
        state = state_norm * state_scale + state_mean
        
        return state

        
    def plan(self, current_state):
        """Plan actions using CEM and world model (Delta Dynamics Version)"""
        
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.ones(self.horizon, self.action_dim, device=self.device)
        
        # Normalize initial state ONCE
        current_state_raw = torch.tensor(current_state, device=self.device, dtype=torch.float32)
        state_mean_t = torch.tensor(self.scalers['state'].mean_, device=self.device, dtype=torch.float32)
        state_scale_t = torch.tensor(self.scalers['state'].scale_, device=self.device, dtype=torch.float32)
        current_state_norm = (current_state_raw - state_mean_t) / state_scale_t
        
        action_mean_t = torch.tensor(self.scalers['action'].mean_, device=self.device, dtype=torch.float32)
        action_scale_t = torch.tensor(self.scalers['action'].scale_, device=self.device, dtype=torch.float32)
        
        for i in range(self.iterations):
            # Sample action sequences (in normalized space)
            action_seqs = torch.normal(
                mean.expand(self.num_samples, -1, -1),
                std.expand(self.num_samples, -1, -1)
            )
            action_seqs = torch.clamp(action_seqs, -1.0, 1.0)
            
            # Denormalize actions to raw space
            action_seqs_raw = action_seqs * action_scale_t + action_mean_t
            action_seqs_raw = torch.clamp(action_seqs_raw, -1.0, 1.0)
            
            costs = torch.zeros(self.num_samples, device=self.device)
            state_norm = current_state_norm.expand(self.num_samples, -1)  # Keep normalized
            
            for t in range(self.horizon):
                action_raw = action_seqs_raw[:, t, :]
                action_norm = (action_raw - action_mean_t) / action_scale_t
                
                with torch.no_grad():
                    # PREDICT DELTA
                    delta_norm = self.world_model(state_norm, action_norm)
                    
                # APPLY DELTA: Next = Current + Delta
                next_state_norm = state_norm + delta_norm
                
                # Denormalize for cost evaluation
                next_state_raw = next_state_norm * state_scale_t + state_mean_t
                
                step_cost = self.cost_function.get_cost(next_state_raw, action_raw)
                costs += step_cost
                
                state_norm = next_state_norm
            
            # Select elites
            k = max(1, int(0.1 * self.num_samples))
            top_costs, top_indices = torch.topk(costs, k, largest=False)
            top_actions = action_seqs[top_indices]
            
            # Update distribution
            new_mean = top_actions.mean(dim=0)
            new_std = top_actions.std(dim=0)
            
            mean = 0.7 * new_mean + 0.3 * mean
            std = 0.7 * new_std + 0.3 * std
        
        best_action_raw = action_seqs_raw[top_indices[0], 0, :].cpu().numpy()
        return best_action_raw


