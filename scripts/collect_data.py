import sys, os, pickle, numpy as np
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
from env_wrapper import ParkingWrapper
from utils import cfg

def collect():
    print("COLLECTING REAL DRIVING DATA...")
    env = ParkingWrapper()
    data = []
    
    for ep in range(100):  # 100 episodes
        obs, _ = env.reset()
        for t in range(50):
            # ACTION STRATEGY:
            # 80% of time: Drive forward/backward + steer (Meaningful motion)
            # 20% of time: Random noise (Exploration)
            
            if np.random.rand() < 0.8:
                throttle = np.random.choice([-1.0, 0.5, 1.0]) # Hard gas/brake
                steer = np.random.uniform(-1, 1)              # Random steer
                action = np.array([throttle, steer])
            else:
                action = np.random.uniform(-1, 1, size=2)
                
            next_obs, _, done, truncated, _ = env.step(action)
            
            data.append({'state': obs, 'action': action, 'next_state': next_obs})
            obs = next_obs
            if done or truncated: break
            
    os.makedirs(cfg.data_dir, exist_ok=True)
    with open(os.path.join(cfg.data_dir, "test_env_data.pkl"), 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Saved {len(data)} MEANINGFUL transitions.")

if __name__ == "__main__":
    collect()
