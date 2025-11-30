import sys, os, pickle, torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root /  'src'))

from world_model import WorldModel
from utils import cfg



def train():
    # LOAD DATASET
    data_path = os.path.join(cfg.data_dir, "test_env_data.pkl")

    print(f"LOADING DATASET FROM {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # PREPARE ARRAYS
    states = np.array([transition['state'] for transition in data], dtype = np.float32)
    actions = np.array([transition['action'] for transition in data], dtype = np.float32)
    next_states = np.array([transition['next_state'] for transition in data], dtype = np.float32)

    # NORMALIZE DATA
    scaler_state = StandardScaler()
    scaler_action = StandardScaler()
    states_norm = scaler_state.fit_transform(states)
    actions_norm = scaler_action.fit_transform(actions)
    
    next_states_norm = scaler_state.transform(next_states)
    
    # SAVE SCALERS (FOR INFERENCE LATER)
    with open(os.path.join(cfg.models_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump({'state': scaler_state, 'action': scaler_action}, f)
    print(f"SAVED SCALERS TO {os.path.join(cfg.models_dir, 'scalers.pkl')}")
    
    # TRAIN
    
    dataset = TensorDataset(
        torch.tensor(states_norm),
        torch.tensor(actions_norm),
        torch.tensor(next_states_norm)
    )
    
    loader = DataLoader(
        dataset,
        batch_size = 64,
        shuffle = True
    )
    
    model = WorldModel()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.MSELoss()
    
    print("STARTING TRAINING...")
    epochs = 30
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for s, a, target in loader:
            optimizer.zero_grad()
            pred = model(s, a)
            loss = loss_fn(pred, target)  # PREDICTING DELTA
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"EPOCH {epoch + 1}/{epochs} - LOSS: {total_loss / len(loader):.6f}")
    
    # SAVE MODEL
    model.save(os.path.join(cfg.models_dir, 'world_model.pth'))

if __name__ == "__main__":
    train()
            
        