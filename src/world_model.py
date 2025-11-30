""" WORLD MODEL : GIVEN CURRENT STATE AND ACTION, PREDICT NEXT STATE DELTA 
NEXT STATE = f(STATE, ACTION) """
import torch
import torch.nn as nn


class WorldModel(nn.Module): 
    def __init__(
        self,
        state_dim: int = 6, # OUTPUT VECTOR (X, Y, VX, VY, COS(THETA), SIN(THETA) ) - where car is and how its moving 
        action_dim: int = 2,# ACTION CONTROLS (ACCELERATION/BRAKING, STEERING)
        hidden_dim: int = 256 # SIZE OF HIDDEN LAYERS
    ):
        
        super().__init__() # HANDLE AUTOGRAD
        
        self.net = nn.Sequential(
            nn.Linear(in_features = state_dim + action_dim, out_features = hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features = hidden_dim, out_features = state_dim)
        )
        
        
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """ PREDICT NEXT STATE DELTA GIVEN CURRENT STATE AND ACTION """
        
        return self.net(torch.cat([state, action], dim = -1))   # CONCATENATE TENSORS 
    
    def save(
        self,
        path: str
    ):
        """ SAVE MODEL PARAMETERS TO DISK """
        torch.save(self.state_dict(), path)
        print(f"SAVED WORLD MODEL TO {path}")
        
    def load(
        self,
        path: str
    ):
        """ LOAD MODEL PARAMETERS FROM DISK """
        self.load_state_dict(torch.load(path))
        self.eval()
        print(f"LOADED WORLD MODEL FROM {path}")
        