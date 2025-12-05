import sys, os, pickle, torch
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from env_wrapper import ParkingWrapper
from world_model import WorldModel
from planner import CEMPlanner
from cost import CostFunction
from utils import cfg

def main():
    print("\n" + "="*60)
    print("DREAMER-PARK DEBUG TEST")
    print("="*60)
    
    # 1. Test Environment
    print("\n Testing Environment...")
    env = ParkingWrapper(render_mode=None)
    obs, info = env.reset()
    goal = env.get_goal().astype(np.float32)
    
    print(f"  OBS shape: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"  GOAL shape: {goal.shape}, value: {goal}")
    assert obs.shape == (6,), "Observation should be 6D"
    assert goal.shape == (6,), "Goal should be 6D"
    print("  ✓ Environment OK")
    
    # 2. Test Scalers
    print("\n Testing Scalers...")
    scaler_path = os.path.join(cfg.models_dir, "scalers.pkl")
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    
    print(f"  State scaler mean shape: {scalers['state'].mean_.shape}")
    print(f"  State scaler scale shape: {scalers['state'].scale_.shape}")
    print(f"  Action scaler mean shape: {scalers['action'].mean_.shape}")
    print(f"  Action scaler scale shape: {scalers['action'].scale_.shape}")
    print("  ✓ Scalers OK")
    
    # 3. Test World Model
    print("\n Testing World Model...")
    wm = WorldModel(state_dim=cfg.state_dim, action_dim=cfg.action_dim)
    model_path = os.path.join(cfg.models_dir, "world_model.pt")
    wm.load_state_dict(torch.load(model_path, map_location="cpu"))
    wm.eval()
    
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action_t = torch.tensor([0.5, 0.5], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        pred = wm(obs_t, action_t)
    print(f"  Input obs: {obs}")
    print(f"  Input action: [0.5, 0.5]")
    print(f"  Predicted next state: {pred.numpy()}")
    print("  ✓ World Model OK")
    
    # 4. Test Cost Function
    print("\n Testing Cost Function...")
    cost_fn = CostFunction(goal)
    
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action_t = torch.tensor([0.5, 0.5], dtype=torch.float32).unsqueeze(0)
    
    cost = cost_fn.get_cost(obs_t, action_t)
    print(f"  Cost at initial state: {cost.item():.4f}")
    print("  ✓ Cost Function OK")
    
    # 5. Test Planner
    print("\n Testing Planner...")
    planner = CEMPlanner(
        world_model=wm,
        cost_function=cost_fn,
        scalers=scalers,
        horizon=5,           # Shorter horizon for debug
        num_samples=64,      # Fewer samples for speed
        iterations=2,        # Fewer iterations for debug
        device="cpu"
    )
    
    print("  Running plan()...")
    best_action = planner.plan(obs)
    
    print(f"  Best action: {best_action}")
    assert best_action.shape == (2,), "Action should be 2D"
    assert np.all(best_action >= -1.0) and np.all(best_action <= 1.0), "Action should be in [-1, 1]"
    print("  ✓ Planner OK")
    
    # 6. Test one environment step
    print("\n Testing Environment Step...")
    next_obs, reward, terminated, truncated, info = env.step(best_action)
    print(f"  Action executed: {best_action}")
    print(f"  Reward: {reward}")
    print(f"  Next obs: {next_obs}")
    print("  ✓ Step OK")
    
    env.close()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
