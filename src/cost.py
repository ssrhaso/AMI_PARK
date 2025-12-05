import torch
import numpy as np


class CostFunction:
    def __init__(
        self, 
        target_state : np.ndarray
    ):
        """ TARGET STATE : [ x, y, vx, vy, cos_h, sin_h ] """
        
        self.target = torch.tensor(target_state, dtype = torch.float32) # CONV TO TENSOR
                
    def get_cost(self, states: torch.Tensor, actions: torch.Tensor):
        """Score state-action pairs with robust penalties"""

        #  Standard Costs 
        pos_error = torch.norm(states[..., 0:2] - self.target[0:2], dim=-1)
        heading_error = torch.norm(states[..., 4:6] - self.target[4:6], dim=-1)
        action_mag = torch.norm(actions, dim=-1)
        
        #  Robust Penalties 

        # 1. HEAVY Velocity Penalty (Quadratic): Punish high speeds exponentially.
        # This is the most important change to prevent run-away acceleration.
        vel_error = torch.norm(states[..., 2:4], dim=-1)
        velocity_penalty = 4.0 * vel_error**2

        # 2. Wrong Direction Penalty: Directly penalize moving away from the goal.
        # This is a strong heuristic to counteract model bias.
        current_pos = states[..., 0:2]
        goal_pos = self.target[0:2].unsqueeze(0)
        vec_to_goal = goal_pos - current_pos
        vec_to_goal = vec_to_goal / (torch.norm(vec_to_goal, dim=-1, keepdim=True) + 1e-6)
        
        current_vel = states[..., 2:4]
        current_vel = current_vel / (torch.norm(current_vel, dim=-1, keepdim=True) + 1e-6)
        
        # Dot product is 1 if aligned, -1 if opposed. Penalize negative values.
        progress = (vec_to_goal * current_vel).sum(dim=-1)
        wrong_way_penalty = 5.0 * torch.relu(-progress)

        # 3. Stronger Goal Bonus
        goal_bonus = -20.0 * torch.exp(-10.0 * pos_error)

        #  Final Cost 
        # The weights are re-tuned for these new penalties
        cost = (
            10.0 * pos_error +          # Primary objective
            velocity_penalty +        # CRITICAL: Stop accelerating into walls
            0.5 * heading_error +       # Heavier weight on correct angle
            0.1 * action_mag +          # Discourage jerky movements
            wrong_way_penalty +       # CRITICAL: Stop going backwards
            goal_bonus
        )
        
        return cost

