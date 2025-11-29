import torch

CONFIG = {
    "env_name": "parking-v0",           # environment name (parking simulation)
    "state_dim": 6,                     # x, y, vx, vy, cos_h, sin_h (dictating the state representation)
    "action_dim": 2,                    # steering, acceleration (actions for the vehicle)
    "hidden_dim": 256,                  # hidden layer size for neural networks (256 for better capacity)
    "horizon": 10,                      # planning horizon (number of steps to plan ahead)
    "device": "cuda" if torch.cuda.is_available() else "cpu" # device configuration
}

def get_device():
    return torch.device(CONFIG["device"])



