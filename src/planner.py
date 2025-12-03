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
        current_state = torch.tensor(current_state, device=self.device, dtype=torch.float32)
        
        # OPTIMIZATION LOOP (THINKING PROCESS)
        
        