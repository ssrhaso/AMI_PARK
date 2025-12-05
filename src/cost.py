import torch
import numpy as np

class CostFunction:
    def __init__(self, target_state: np.ndarray):
        """TARGET STATE: [x, y, vx, vy, cos_h, sin_h]"""
        self.target = torch.tensor(target_state, dtype=torch.float32)

    def get_cost(self, states: torch.Tensor, actions: torch.Tensor):
        """Score trajectories - STRONGLY encourage reaching target"""
        
        # Position error
        pos_error = torch.norm(states[..., 0:2] - self.target[0:2], dim=-1)
        
        # Velocity (want zero)
        vel_error = torch.norm(states[..., 2:4], dim=-1)
        
        # Heading error
        heading_error = torch.norm(states[..., 4:6] - self.target[4:6], dim=-1)
        
        # Wall penalties
        x = states[..., 0]
        y = states[..., 1]
        wall_cost = torch.relu(torch.abs(x) - 12.0) + torch.relu(torch.abs(y) - 12.0)
        
        # Action magnitude penalty (small)
        action_mag = torch.norm(actions, dim=-1)
        
        # PRIMARY OBJECTIVE: Reach target
        cost = (
            50.0 * pos_error +           # Heavy position penalty
            5.0 * vel_error +            # Stop moving when close
            1.0 * heading_error +        # Correct heading
            1000.0 * wall_cost +         # Avoid walls
            0.01 * action_mag +          # Minimize actions
            -100.0 * torch.exp(-20.0 * pos_error)  # STRONG bonus for being close
        )
        
        return cost
