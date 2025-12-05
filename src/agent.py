import os, pickle, torch
from world_model import WorldModel
from cost import CostFunction
from planner import CEMPlanner

class DreamerAgent:
    def __init__(self, target_state, models_dir='models'):
        self.device = torch.device("cpu")
        
        # Load Model
        self.world_model = WorldModel().to(self.device)
        self.world_model.load_state_dict(torch.load(os.path.join(models_dir, 'world_model.pt'), map_location=self.device))
        self.world_model.eval()
        
        # Load Scalers
        with open(os.path.join(models_dir, 'scalers.pkl'), 'rb') as f:
            self.scalers = pickle.load(f)
            
        self.cost = CostFunction(target_state)
        
        # AGGRESSIVE PLANNER
        self.planner = CEMPlanner(
            world_model=self.world_model,
            cost_function=self.cost,
            scalers=self.scalers,
            horizon=5,           # Short horizon = Fast & Responsive
            num_samples=1000,    # Many samples = Find good moves
            num_iterations=5,    # Think hard
            device=str(self.device)
        )

    def act(self, state):
        return self.planner.plan(state)
