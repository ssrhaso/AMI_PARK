""" WRAPS HIGHWAY-ENV FUNCTIONS FOR EASE OF USE """

import gymnasium as gym
import highway_env
import numpy as np

class ParkingWrapper:
    def __init__(
        self,
        env_name : str = 'parking-v0',
        render_mode = None
    ):
        self.env = gym.make(id = env_name, render_mode = render_mode)
        
        
    def reset(
        self
    ):
        """ RESETS THE ENVIRONMENT AND RETURNS THE INITIAL STATE """
        
        obs_dict, info = self.env.reset()
        obs = obs_dict['observation']
        return obs, info
    
    def step(
        self,
        action : np.ndarray # STEERING / ACCELERATION ACTION, RANGE [-1, 1]
    ):
        """ TAKES A STEP IN THE ENVIRONMENT WITH THE GIVEN ACTION """

        obs_dict, reward, done, truncated, info = self.env.step(action)
        obs = obs_dict['observation']
        return obs, reward, done, truncated, info
    
    
    def close(
        self
    ):
        """ CLOSES THE ENVIRONMENT """
        
        self.env.close()
    

# ENTRY TEST
if __name__ == "__main__":
    
    print("TESTING PARKING WRAPPER...")
    env = ParkingWrapper(env_name = 'parking-v0', render_mode = 'rgb_array')
    obs, _ = env.reset()
    print(f"ENV CREATED WITH OBS SHAPE: {obs.shape}")
    env.close()