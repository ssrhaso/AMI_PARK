""" PHYSICS MODEL OF THE ENVIRONMENT

LEARNS TO PREDICT FUTURE BASED ON ENVIRONMENT DYNAMICS """

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """Learns environment dynamics: s_{t+1} = f(s_t, a_t)"""
    
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=256):
        super(WorldModel, self).__init__()
        
        input_dim = state_dim + action_dim              # INPUT LAYER
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)     # HIDDEN LAYER 1 / ENCODER
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)    # HIDDEN LAYER 2 / PROCESSOR
        self.fc3 = nn.Linear(hidden_dim, state_dim)     # OUTPUT LAYER   / DECODER
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # CONCATENATE STATE AND ACTION
        x = torch.relu(self.fc1(x))             
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    # SAVE / LOAD MODEL FUNCTIONS
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
