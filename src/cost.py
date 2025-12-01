import torch
import numpy as np


class CostFunction:
    def __init__(
        self, 
        target_state : np.ndarray
    ):
        """ TARGET STATE : [ x, y, vx, vy, cos_h, sin_h ] """
        
        self.target = torch.tensor(target_state, dtype = torch.float32) # CONV TO TENSOR
        

    def get_cost(
        self,
        states: torch.Tensor
    ):
        """ COMPUTE COST AS EUCLIDEAN DISTANCE TO TARGET STATE """
        
        
        # 1 - DISTANCE COST (EUCLIDEAN NORM) (POSITION)
        
        pos_error = torch.norm(states[..., 0:2] - self.target[0:2], dim = -1)  # POSITION ERROR
        
        # 2 - VELOCITY COST (EUCLIDEAN NORM) (STOPPING)
        
        vel_error = torch.norm(states[..., 2:4], dim = -1)  # VELOCITY ERROR
        
        # 3 - HEADING COST (ANGLE DIFFERENCE) (ALIGNMENT)
        
        heading_error = torch.norm(states[..., 4:6] - self.target[4:6], dim = -1)  # HEADING ERROR
        
        # TOTAL COST , WEIGHTING ( POSITION MORE IMPORTANT, THEN VELOCITY, THEN HEADING )
        cost = 1.0 * pos_error + 0.1 * vel_error + 0.1 * heading_error
        
        return cost