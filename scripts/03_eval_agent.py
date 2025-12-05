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
    # 1. Initialize Environment (Visual Mode)
    print("INITIALIZING ENVIRONMENT")
    # render_mode='human' will pop up a window so you can watch
    env = ParkingWrapper(render_mode='human') 
    obs, _ = env.reset()
    
    # 2. Define the Goal
    # For now, we tell the agent to go to [0,0,0,0,0,0] 
    # (Position 0,0, Velocity 0, Angle 0)
    target_state = np.zeros(6) 
    
    # 3. Initialize the Dreamer Agent
    print("LOADING DREAMER AGENT")
    agent = DreamerAgent(target_state)
    
    print("SIMULATION STARTED. CTRL+C TO STOP.")
    
    # 4. The Main Loop
    for step in range(200): # Run for 200 steps
        # A. PLAN: Agent thinks and returns best action
        action = agent.get_action(obs)
        
        # B. ACT: Execute action in environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Optional: Print status
        if step % 10 == 0:
            print(f"Step {step}: Action Taken {action}")

        if terminated or truncated:
            print("EPISODE COMPLETED")
            break
            
    env.close()
    print("COMPLETED")

if __name__ == "__main__":
    main()
