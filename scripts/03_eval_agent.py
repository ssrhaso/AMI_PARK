import sys
import os
from pathlib import Path
import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from env_wrapper import ParkingWrapper
from agent import DreamerAgent

def main():
    # Initialize Environment
    print("INITIALIZING ENVIRONMENT")
    env = ParkingWrapper(render_mode='human')
    obs, _ = env.reset()
    
    # Get Goal
    target_state = env.get_goal()
    print(f"TARGET STATE: {target_state}")
    print(f"TARGET STATE SHAPE: {target_state.shape}")
    
    # Initialize Agent
    agent = DreamerAgent(target_state)
    print("AGENT INITIALIZED. SIMULATION STARTED.")
    
    # Main Loop
    for step in range(200):
        env.env.render()
        
        # Get action from planner
        action = agent.act(obs)
        
        # Execute in environment (action already in [-1, 1] range)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print status
        x, y, vx, vy, c, s = obs
        # FIXED LINE BELOW: action[0] and action[1]
        print(f"step={step:3d} | pos=({x:6.2f},{y:6.2f}) | vel=({vx:5.2f},{vy:5.2f}) | action=({action[0]:5.2f},{action[1]:5.2f})")
        
        import time
        time.sleep(0.02)  # Keep rendering smooth
        
        if terminated or truncated:
            print("EPISODE COMPLETED")
            break
    
    env.close()
    print("COMPLETED")

if __name__ == "__main__":
    main()
