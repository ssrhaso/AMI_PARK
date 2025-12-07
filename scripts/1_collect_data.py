""" TEST ENVIRONMENT SETUP AND RUNNING SCRIPT """

import sys
import os
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

# IMPORTS FROM SRC
from env_wrapper import ParkingWrapper
from utils import save_dataset  


def main():
    print(f"STARTING TEST ENVIRONMENT SCRIPT...")
    
    # INIT ENV 
    env = ParkingWrapper(env_name = 'parking-v0', render_mode = 'human')
    obs, _ = env.reset()
    collected_data = []
    
    # RUN EPISODE
    for i in range(100):
        # RANDOM ACTION
        
        action = np.random.uniform(low = -1.0, high = 1.0, size = (2,)) # STEERING / ACCELERATION (RANGE [-1, 1])
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # STORE DATA (FOR TRAINING WORLD MODEL)
        transition = {
            'state' : obs,
            'action' : action,
            'next_state' : next_obs,
            'reward' : reward,
        }
        collected_data.append(transition)
        obs = next_obs
        
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    
    # SAVE COLLECTED DATA
    save_dataset(collected_data, filename = 'test_env_data.pkl')
    print(f"TEST ENVIRONMENT SCRIPT COMPLETED")

if __name__ == "__main__":
    main()
    
    
    