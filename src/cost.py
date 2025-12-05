import torch
import numpy as np

class CostFunction:
    def __init__(self, target_state):
        self.target = torch.tensor(target_state, dtype=torch.float32)

    def get_cost(self, states, actions):
        if isinstance(states, np.ndarray): 
            states = torch.tensor(states, dtype=torch.float32)
        
        # Extract Position
        pos = states[..., 0:2]
        target_pos = self.target[0:2]
        
        # 1. DISTANCE (Primary Objective)
        # This forces movement.
        dist = torch.norm(pos - target_pos, dim=-1)
        
        # 2. WALL BARRIER (Hard Constraint)
        # Only penalize if we actually LEAVE the playable area.
        # Playable area is roughly [-0.4, 0.4].
        # ReLU is 0 if inside, positive if outside.
        x_out = torch.relu(torch.abs(pos[..., 0]) - 0.4)
        y_out = torch.relu(torch.abs(pos[..., 1]) - 0.4)
        
        # If x_out > 0, we are crashing. Add MASSIVE cost.
        wall_cost = 1000.0 * (x_out + y_out)
        
        return dist + wall_cost
