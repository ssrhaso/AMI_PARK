import sys, os, pickle, torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from world_model import WorldModel
from utils import cfg

def train():
    # 1. LOAD DATASET
    data_path = os.path.join(cfg.data_dir, "test_env_data.pkl")
    print(f"LOADING DATASET FROM {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    # 2. PREPARE ARRAYS
    states = np.array([tr['state'] for tr in data], dtype=np.float32)
    actions = np.array([tr['action'] for tr in data], dtype=np.float32)
    next_states = np.array([tr['next_state'] for tr in data], dtype=np.float32)
    
    # --- CRITICAL CHANGE: CALCULATE RAW DELTAS ---
    deltas = next_states - states
    
    # 3. NORMALIZE
    scaler_state = StandardScaler()
    scaler_action = StandardScaler()
    scaler_delta = StandardScaler()  # NEW: Dedicated scaler for changes
    
    states_norm = scaler_state.fit_transform(states)
    actions_norm = scaler_action.fit_transform(actions)
    deltas_norm = scaler_delta.fit_transform(deltas)  # Normalize the deltas
    
    # 4. SAVE SCALERS (Now includes 'delta')
    scaler_path = os.path.join(cfg.models_dir, 'scalers.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'state': scaler_state, 
            'action': scaler_action,
            'delta': scaler_delta
        }, f)
    print(f"SAVED SCALERS (State, Action, Delta) TO {scaler_path}")
    
    # 5. PREPARE TRAINING
    dataset = TensorDataset(
        torch.tensor(states_norm, dtype=torch.float32),
        torch.tensor(actions_norm, dtype=torch.float32),
        torch.tensor(deltas_norm, dtype=torch.float32) # Target is DELTA
    )
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = WorldModel().to(torch.device("cpu"))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    print("STARTING TRAINING ON DELTA DYNAMICS...")
    epochs = 50 
    
    for epoch in range(epochs):
        total_loss = 0.0
        for s, a, target in loader:
            optimizer.zero_grad()
            pred = model(s, a) 
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"EPOCH {epoch+1}/{epochs} - LOSS: {total_loss / len(loader):.6f}")
            
    model_path = os.path.join(cfg.models_dir, 'world_model.pt')
    model.save(model_path)

if __name__ == "__main__":
    train()
