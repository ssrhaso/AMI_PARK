import sys, os
from pathlib import Path
import time

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from env_wrapper import ParkingWrapper
from agent import DreamerAgent


def main():
    print('='*70)
    print('DREAMER-PARK EVALUATION')
    print('='*70)
    
    print('\n Initializing environment...')
    env = ParkingWrapper(render_mode='human')
    obs, _ = env.reset()
    target_state = env.get_goal()
    print(f'    Target: {target_state}')
    
    print('\n Initializing agent...')
    agent = DreamerAgent(target_state, models_dir=str(project_root / 'models'))
    
    print('\n Running simulation...\n')
    
    for step in range(200):
        env.env.render()
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        x, y, vx, vy, c, s = obs
        print(f'step={step:3d} | pos=({x:7.4f},{y:7.4f}) | vel=({vx:6.4f},{vy:6.4f})')
        
        time.sleep(0.02)
        
        if terminated or truncated:
            print('\n✓ COMPLETED')
            break
    
    env.close()


if __name__ == '__main__':
    main()
