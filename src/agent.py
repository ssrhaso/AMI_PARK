""" AGENT MODULE - CONNECTS WORLD MODEL , COST FUNCTION AND PLANNER TOGETHER """
import os
import pickle
import numpy as np
import torch

from world_model import WorldModel
from cost import CostFunction
from planner import CEMPlanner
from utils import cfg


class DreamerAgent:
    def __init__(
        self,
        target_state : np.ndarray,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # LOAD TRAINED WORLD MODEL
        self.world_model = WorldModel(
            state_dim = cfg.state_dim,
            action_dim = cfg.action_dim,
        ).to(self.device)
        
        model_path = os.path.join(cfg.models_dir, "world_model.pt")
        state_dict = torch.load(model_path, map_location = self.device)
        self.world_model.load_state_dict(state_dict)
        self.world_model.eval()
        print(f"LOADED WORLD MODEL FROM {model_path}")
        
        # LOAD SCALERS
        scaler_path = os.path.join(cfg.models_dir, "scalers.pkl")
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f) # {'state': .. , 'action': ..}
            
        # COST FUNCTION
        self.cost_function = CostFunction(target_state = target_state)
        
        # PLANNER
        self.planner = CEMPlanner(
            world_model = self.world_model,
            cost_function = self.cost_function,
            scalers = scalers,
            horizon = 8, 
            num_samples = 500,
            iterations = 10,
            device = str(self.device)
        )
        
    def act(
        self,
        state : np.ndarray
    ) -> np.ndarray:
        """ GIVEN CURRENT STATE, RETURNS ACTION FROM PLANNER """
        
        action = self.planner.plan(state)
        return action
    