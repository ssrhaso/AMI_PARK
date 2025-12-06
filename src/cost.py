""" COST FUNCTION DEFINITION , ACHIEVABLE GOAL """
import torch
import numpy as np

class CostFunction:
    def __init__(self, target_state):
        self.target = torch.tensor(target_state, dtype=torch.float32)

    def get_cost(self, states, actions):
        if isinstance(states, np.ndarray): 
            states = torch.tensor(states, dtype=torch.float32)
        
        pos = states[..., 0:2]
        target_pos = self.target[0:2]
        
        # 1. DISTANCE (Simple and effective)
        dist = torch.norm(pos - target_pos, dim=-1)
        
        # 2. WALL SAFETY (The invisible fence)
        limit = 0.35
        x_out = torch.relu(torch.abs(pos[..., 0]) - limit)
        y_out = torch.relu(torch.abs(pos[..., 1]) - limit)
        wall_cost = 1000.0 * (x_out + y_out)
        
        return dist + wall_cost
