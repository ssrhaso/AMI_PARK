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
    print('='*70)
    print('TRAINING WORLD MODEL')
    print('='*70)
    
    data_path = os.path.join(cfg.data_dir, 'test_env_data.pkl')
    print(f'\n Loading data from {data_path}...')
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f'    Loaded {len(data)} transitions')
    
    states = np.array([tr['state'] for tr in data], dtype=np.float32)
    actions = np.array([tr['action'] for tr in data], dtype=np.float32)
    next_states = np.array([tr['next_state'] for tr in data], dtype=np.float32)
    
    print(f'    State range: [{states.min():.3f}, {states.max():.3f}]')
    
    scaler_state = StandardScaler()
    scaler_action = StandardScaler()
    
    states_norm = scaler_state.fit_transform(states)
    actions_norm = scaler_action.fit_transform(actions)
    next_states_norm = scaler_state.transform(next_states)
    
    dataset = TensorDataset(
        torch.tensor(states_norm, dtype=torch.float32),
        torch.tensor(actions_norm, dtype=torch.float32),
        torch.tensor(next_states_norm, dtype=torch.float32)
    )
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f'\n Building model...')
    device = torch.device('cpu')
    model = WorldModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    print(f'\n Training...\n')
    
    for epoch in range(50):
        total_loss = 0.0
        for s, a, target_ns in loader:
            s, a, target_ns = s.to(device), a.to(device), target_ns.to(device)
            optimizer.zero_grad()
            pred_ns = model(s, a)
            loss = loss_fn(pred_ns, target_ns)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1:2d}/50 - Loss: {total_loss / len(loader):.6f}')
    
    os.makedirs(cfg.models_dir, exist_ok=True)
    
    model_path = os.path.join(cfg.models_dir, 'world_model.pt')
    model.save(model_path)
    print(f'\n✓ Saved model to {model_path}')
    
    scaler_path = os.path.join(cfg.models_dir, 'scalers.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({'state': scaler_state, 'action': scaler_action}, f)
    print(f'✓ Saved scalers to {scaler_path}')
    
    print('='*70)


if __name__ == '__main__':
    train()
