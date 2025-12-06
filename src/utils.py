""" HELPER SCRIPT

UTILITIES FOR DATASET HANDLING AND CONFIGURATION """

import os
import json
import pickle
import numpy as np
import torch
from dataclasses import dataclass, asdict

@dataclass
class Config:
    # ENV
    env_name : str = 'parking-v0'
    state_dim : int = 6                     # OUTPUT VECTOR (X, Y, VX, VY, COS(THETA), SIN(THETA) )
    action_dim : int = 2                    # ACTION CONTROLS (ACCELERATION/BRAKING, STEERING)   
    
    # DATA 
    num_data_steps : int = 50000              # TOTAL NUMBER OF STEPS TO COLLECT
    data_dir : str = "data"
    models_dir : str = "models"
    
    def save(
        self,
        path : str
    ):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent = 2)

cfg = Config()

def save_dataset(
    data,
    filename : str = "trajectories.pkl"
): 
    """ SAVES LIST OF DICT TO PICKLE FILE """

    path = os.path.join(cfg.data_dir, filename)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"SAVED {len(data)} TRANSITIONS TO {path}")
    
def load_dataset(
    filename : str = "trajectories.pkl"
):
    """ LOADS LIST OF DICT FROM PICKLE FILE """

    path = os.path.join(cfg.data_dir, filename)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"LOADED {len(data)} TRANSITIONS FROM {path}")
    return data