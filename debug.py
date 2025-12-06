"""
Multi-episode reliability test for Dreamer parking agent.
Runs N episodes and reports success / crash / timeout stats.
"""

import sys
from pathlib import Path
import numpy as np
import torch  # only for updating the cost target

# Make src/ importable
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / "src"))

from agent import DreamerAgent          # [file:2]
from env_wrapper import ParkingWrapper  # [file:5]


def run_episodes(num_episodes: int = 10, max_steps: int = 60):
    env = ParkingWrapper()
    # Dummy target for init; will overwrite per episode
    dummy_target = np.zeros(6, dtype=np.float32)
    agent = DreamerAgent(dummy_target)

    successes = 0
    crashes = 0
    timeouts = 0
    min_dists = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        target = env.get_goal()

        # Update cost function target for this episode
        agent.cost.target = torch.tensor(target, dtype=torch.float32)
        agent.cost.target_pos = agent.cost.target[0:2]
        agent.cost.target_heading = agent.cost.target[4]
        agent.planner.cost_function = agent.cost

        dist0 = np.linalg.norm(obs[0:2] - target[0:2])
        min_dist = dist0
        done = False
        truncated = False
        step = 0

        while not (done or truncated):
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            d = np.linalg.norm(obs[0:2] - target[0:2])
            min_dist = min(min_dist, d)
            step += 1
            if step >= max_steps:
                truncated = True

        min_dists.append(min_dist)

        # Simple outcome classification
        if min_dist < 0.1:
            print(f"Episode {ep+1}: SUCCESS (min dist={min_dist:.3f}, steps={step})")
            successes += 1
        elif step < max_steps and min_dist >= 0.1:
            print(f"Episode {ep+1}: CRASH  (min dist={min_dist:.3f}, steps={step})")
            crashes += 1
        else:
            print(f"Episode {ep+1}: TIMEOUT (min dist={min_dist:.3f}, steps={step})")
            timeouts += 1

    env.close()

    print("\n================ OVERALL STATS ================")
    print(f"Episodes    : {num_episodes}")
    print(f"Successes   : {successes} ({successes/num_episodes*100:.1f}%)")
    print(f"Crashes     : {crashes}")
    print(f"Timeouts    : {timeouts}")
    print(f"Avg min dist: {np.mean(min_dists):.3f}")


if __name__ == "__main__":
    run_episodes(num_episodes=10, max_steps=60)
