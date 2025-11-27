import torch

import torch

# Config
CONFIG = {
    "env_name": "parking-v0",
    "state_dim": 6,             # x, y, vx, vy, cos_h, sin_h
    "action_dim": 2,            # steering, acceleration
    "hidden_dim": 256,
    "horizon": 10,              # Planning horizon
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def get_device():
    return torch.device(CONFIG["device"])



