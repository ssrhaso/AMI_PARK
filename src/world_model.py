import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """Learns environment dynamics: s_{t+1} = f(s_t, a_t)"""
    
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=256):
        super(WorldModel, self).__init__()
        input_dim = state_dim + action_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
