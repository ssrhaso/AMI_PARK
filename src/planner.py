""" PLANNER MODULE  - USING CEM WITH NORMALISED DATA TO PLAN ACTIONS """
import torch
import numpy as np


class CEMPlanner:
    def __init__(
        self,
        world_model,
        cost_fn,
        scalers,
        horizon : int = 12,
        num_samples : int = 1000,
        iterations : int = 5,
        device : str = "cpu",
    ):
        self.world_model = world_model
        self.cost_fn = cost_fn
        self.scalers = scalers
        self.horizon = horizon
        self.num_samples = num_samples
        self.iterations = iterations
        self.device = device
        self.action_dim = 2
    
    def normalize(
        self,
        state,
        action,
    ):
        """ CONVERT RAW STATE AND ACTION TO NORMALISED TENSORS FOR MODEL"""
        
        # MEAN FROM SKLEARN SCALER
        state_mean = torch.tensor(self.scalers['state'].mean_, device=self.device, dtype=torch.float32)
        state_scale = torch.tensor(self.scalers['state'].scale_, device=self.device, dtype=torch.float32)
        action_mean = torch.tensor(self.scalers['action'].mean_, device=self.device, dtype=torch.float32)
        action_scale = torch.tensor(self.scalers['action'].scale_, device=self.device, dtype=torch.float32)
        
        # NORMALISE
        state_norm = (state - state_mean) / state_scale
        action_norm = (action - action_mean) / action_scale
        return state_norm, action_norm
    
    def denormalize_state(
        self,
        state_norm,
    ):
        """ CONVERT NORMALISED PREDICTION TENSOR TO RAW STATE  """
        state_mean = torch.tensor(self.scalers['state'].mean_, device=self.device, dtype=torch.float32)
        state_scale = torch.tensor(self.scalers['state'].scale_, device=self.device, dtype=torch.float32)
        
        state = (state_norm * state_scale) + state_mean
        return state
        
    def plan(
        self,
        current_state,
    ):
        """ PLAN ACTIONS USING CEM AND WORLD MODEL """
        # INIT DISTRIBUTION (RANDOM GUESSES)
        mean = torch.zeros((self.horizon, self.action_dim), device=self.device)
        std = torch.ones((self.horizon, self.action_dim), device=self.device)
        
        # PREP CURRENT STATE TENSOR
        current_state_t = torch.tensor(current_state, device=self.device, dtype=torch.float32)
        
        # OPTIMIZATION LOOP (THINKING PROCESS)
        for i in range(self.iterations):
            
            # 1 - SAMPLE 1000 TRAJECTORIES FROM DISTRIBUTION
            
            action_seqs = torch.normal(
                mean.expand(self.num_samples, -1, -1),
                std.expand(self.num_samples, -1, -1)
            )
            action_seqs = torch.clamp(action_seqs, -1, 1)  # CLAMP ACTIONS TO VALID RANGE
            
            # 2 - SIMULATE FUTURE TRAJECTORIES USING WORLD MODEL
            costs = torch.zeros((self.num_samples,), device=self.device)
            state = current_state_t.expand(self.num_samples, -1) #DUPLICATE STATE 1000 TIMES
            
            for t in range(self.horizon):
                action = action_seqs[:, t, :]
                
                # NORMALISE > PREDICT > DENORMALISE
                state_norm, action_norm = self.normalize(state, action)
                
                # PREDICT NEXT STATE
                with torch.no_grad():
                    next_state_norm = self.world_model(state_norm, action_norm)
                
                # DENORMALISE NEXT STATE
                next_state = self.denormalize_state(next_state_norm)
                
                # COMPUTE COST (USING THE CORRECT METHOD NAME)
                # We only pass the state, because cost is purely state-dependent
                step_cost = self.cost_fn.get_cost(next_state)
                costs += step_cost
                
                # UPDATE STATE
                state = next_state

            # 3 - SELECT ELITES (BEST 10%)
            k = self.num_samples // 10
            top_costs, top_indices = torch.topk(costs, k, largest=False) # SMALLEST COST IS BEST
            top_actions = action_seqs[top_indices]

            # 4 - UPDATE DISTRIBUTION (LEARN FROM THE BEST)
            new_mean = top_actions.mean(dim=0)
            new_std = top_actions.std(dim=0)

            # SMOOTH UPDATE (MOMENTUM)
            mean = 0.8 * new_mean + 0.2 * mean
            std = 0.8 * new_std + 0.2 * std

        # RETURN THE FIRST ACTION OF THE BEST PLAN
        return mean[0].cpu().numpy()
