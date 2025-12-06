import torch
import numpy as np

class CostFunction:
    def __init__(self, target_state):
        self.target = torch.tensor(target_state, dtype=torch.float32)
        self.target_pos = self.target[0:2]
        self.target_heading = self.target[4]

    def get_cost(self, states, actions):
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32)
        
        pos = states[..., 0:2]
        heading = states[..., 4]
        
        #  1. PRIMARY OBJECTIVE: REACH TARGET 
        dist_to_target = torch.norm(pos - self.target_pos, dim=-1)
        
        #  2. SECONDARY OBJECTIVE: ALIGN HEADING (only final 10cm) 
        heading_error = torch.remainder(heading - self.target_heading + np.pi, 2 * np.pi) - np.pi
        
        # Smooth ramp: starts at dist=0.10, fully active by dist=0.00
        # Below 0.10m: heading matters. Above 0.10m: heading is ignored.
        heading_weight = torch.clamp(1.0 - (dist_to_target / 0.10), 0.0, 1.0)
        alignment_cost = (1.0 - torch.cos(heading_error)) * heading_weight * 0.2
        
        #  3. HARD CONSTRAINT: STAY AWAY FROM WALLS 
        # Safe zone: [-0.22, 0.22] (leaves 6cm buffer from 0.28 true limit)
        # The 6cm buffer absorbs ~3 steps of model drift
        
        safe_limit = 0.22
        x_violation = torch.relu(torch.abs(pos[..., 0]) - safe_limit)
        y_violation = torch.relu(torch.abs(pos[..., 1]) - safe_limit)
        
        # Penalty accelerates as we get closer to the wall
        # 1cm over = 10.0 cost. 3cm over = 30.0 cost.
        wall_cost = 1000.0 * (x_violation**2 + y_violation**2)
        
        #  COMBINE 
        total = dist_to_target + alignment_cost + wall_cost
        
        return total
