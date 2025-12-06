# A Model-Based Planning Agent for Autonomous Driving

## *Advanced Machine Intelligence (AMI) Implementation*

**Project Duration:** 4 weeks (December 20, 2025 – January 26, 2026)
**Deployment:** Google Colab + Local Laptop
**Theoretical Foundation:** Yann LeCun's JEPA + Model-Based RL + Test-Time Compute

---

## Executive Summary

This project implements a **cognitive planning architecture** for autonomous vehicle control, directly addressing the paradigm shift outlined by Yann LeCun in his 2024–2025 research and his recent launch of Advanced Machine Intelligence (AMI).

**Key Innovation:** Instead of training a monolithic policy network (standard RL), we decompose the problem into:

1. **World Model** – A learned physics simulator that predicts environment dynamics
2. **Cost Module** – An objective function defining task success
3. **Optimizer** – A test-time planner that imagines trajectories and selects optimal actions

**Why This Matters:**

- **System 2 Reasoning:** Moves AI from reactive (System 1) to deliberative (System 2)
- **Aligned with Industry:** This architecture powers OpenAI's o1/o3 reasoning models and DeepMind's MuZero
- **Data Efficient:** Model-based planning requires far fewer environment interactions than model-free RL
- **Robust to Distribution Shift:** Plans adapt to unseen scenarios via internal simulation

---

## 1. Project Overview & Motivation

### 1.1 The Problem Statement

**Core Challenge:** Autonomous parking in continuous control space with:

- Continuous state space (position, heading, velocity, acceleration)
- Continuous action space (steering angle, throttle)
- Hard constraints (cannot collide with other vehicles or boundaries)
- Requires multi-step reasoning (not a single-action task)

**Why Parking?**

- **Visually Intuitive:** Clear success/failure signal
- **Computationally Tractable:** Fast physics simulation
- **Generalizable:** Techniques transfer to lane-following, obstacle avoidance, trajectory optimization
- **Production-Relevant:** Automakers actively research this (Tesla, Waymo, Comma2k19 datasets)

### 1.2 Theoretical Foundation

**Yann LeCun's AMI Paradigm:**

```
Intelligence = Perception + World Model + Cost Function + Optimizer
```

Not: `Input -> Policy -> Output` (standard LLM/RL)
But: `Observation -> Predict Future -> Score Futures -> Execute Best Action`

**Connection to Cutting-Edge Research:**

- **V-JEPA (Meta, 2024):** World models in latent space (we implement this for simple states)
- **Test-Time Compute (MIT/OpenAI, 2024-2025):** Use compute at inference time, not training time
- **Model Predictive Control (Classical Control Theory):** Formalize planning as constrained optimization
- **MPPI (Informatica & ICML 2015, scaled by DeepMind):** Probabilistic path integral approach

---

## 2. Architecture & Technical Approach

### 2.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    DREAMER-PARK AGENT                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  PERCEPTION LAYER (t)                                  │
│  ├─ State from Environment: [x, y, θ, v, a]           │
│  └─ Encode to Latent: s_t = E(observation)            │
│                                                         │
│  ┌──────────────────────────────────────────┐          │
│  │     IMAGINATION LOOP (Test-Time)         │          │
│  │  (Runs 100+ times per decision)          │          │
│  │                                          │          │
│  │  1. Sample Actions: a ~ N(μ, Σ)        │          │
│  │  2. Predict Futures:                    │          │
│  │     s_{t+1} = WM(s_t, a_t)            │          │
│  │     s_{t+2} = WM(s_{t+1}, a_{t+1})    │          │
│  │     ... (H steps)                       │          │
│  │  3. Score Trajectory:                   │          │
│  │     J(τ) = Σ C(s_i) + λ||a||²         │          │
│  │  4. Select Elite Actions                │          │
│  │     (top-10% by score)                  │          │
│  │  5. Update Distribution                 │          │
│  │     μ_new = mean(elite_actions)        │          │
│  │                                          │          │
│  └──────────────────────────────────────────┘          │
│                                                         │
│  ACTION EXECUTION                                      │
│  └─ Execute Best First Action: a_t*                   │
│                                                         │
│  FEEDBACK                                              │
│  └─ Observe Real Transition: s_{t+1}^real            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Component Details

#### **Component A: World Model (Neural Network)**

**Purpose:** Learn the environment's dynamics without explicit physics equations.

**Architecture:**

```
Input:  [state (6D), action (2D)] = 8D
Hidden: 2 layers × 256 neurons, ReLU activations
Output: predicted_next_state (6D)

Total parameters: ~50K (tiny!)
```

**Training Data:**

- Collect via random exploration: 10,000–50,000 transitions
- Format: `(s_t, a_t, s_t+1)` tuples
- Loss Function: MSE + L2 regularization
  ```
  L = || W(s_t, a_t) - s_t+1 ||²_2 + λ||θ||²_2
  ```

**Evaluation:**

- One-step prediction error: || ŝ_t+1 - s_t+1 ||
- Multi-step rollout error: Rollout for H steps, compare to ground truth

**Implementation Notes:**

- Use PyTorch with mixed precision (float32 for stability)
- Data normalization: StandardScaler on input/output
- No fancy techniques needed: simple MLP works perfectly

---

#### **Component B: Cost Module (Scoring Function)**

**Purpose:** Define what "good parking" means

**Cost Function:**

```python
def cost_function(state, target_position, target_heading):
    x, y, heading, vx, vy, a = state
  
    # Position error (distance to target)
    pos_error = sqrt((x - target_x)² + (y - target_y)²)
  
    # Heading error (alignment)
    heading_error = min(|heading - target_heading|, 2π - |heading - target_heading|)
  
    # Velocity penalty (should be stopped)
    velocity_penalty = sqrt(vx² + vy²)
  
    # Collision penalty (binary, detected by environment)
    collision_penalty = 1000 if collided else 0
  
    # Total cost
    C = w1 * pos_error + w2 * heading_error + w3 * velocity_penalty + collision_penalty
  
    return C
```

**Weights (Tunable):**

- w1 = 1.0 (position most important)
- w2 = 0.5 (heading somewhat important)
- w3 = 0.1 (velocity penalty small)

**Special Cases:**

- If collision: return `np.inf` (invalid trajectory)
- If parked perfectly: return 0

---

#### **Component C: Planner / Optimizer (Test-Time Brain)**

**Algorithm: Cross-Entropy Method (CEM) + MPPI**

**Pseudocode:**

```
for each timestep:
  
    Initialize:
    μ = [0, 0, 0, ..., 0]  (mean action sequence)
    Σ = I                   (covariance, start at identity)
  
    for iteration in range(num_planning_iterations):
    
        # Sample action sequences
        action_sequences = sample(N=1000, mean=μ, cov=Σ)
    
        # Rollout each sequence through world model
        costs = []
        for seq in action_sequences:
            s_current = state_now
            total_cost = 0
            for action in seq:
                s_next = world_model(s_current, action)
                total_cost += cost_function(s_next)
                s_current = s_next
            costs.append(total_cost)
    
        # Select elite (top-10%)
        elite_idx = argsort(costs)[:int(0.1 * N)]
        elite_sequences = action_sequences[elite_idx]
    
        # Update distribution toward elites
        μ_new = mean(elite_sequences, axis=0)
        Σ_new = cov(elite_sequences)
    
        # Smooth update (momentum)
        μ = 0.9 * μ + 0.1 * μ_new
        Σ = 0.9 * Σ + 0.1 * Σ_new
  
    # Execute best first action
    best_sequence = elite_sequences[0]
    execute_action(best_sequence[0])
  
    # Observe real transition
    state_now = environment.step(best_sequence[0])
```

**Hyperparameters:**

- Planning Horizon (H): 10 steps
- Sample Budget (N): 1,000 trajectories per decision
- Planning Iterations: 3–5
- Elite Fraction: 10%
- Momentum: 0.9

**Computational Cost:**

- Per decision: 1,000 forward passes through World Model
- World Model forward pass: ~1 ms (tiny network)
- Total per decision: ~1 second → **1 Hz planning frequency**
- Feasible on Colab T4 GPU

---

### 2.3 Data Flow Summary

```
TRAINING PHASE (Week 1):
Random Exploration → Collect 50K Transitions → Train World Model
                                                ↓
INFERENCE PHASE (Weeks 2-4):
Environment State → Planner (1K Rollouts) → Best Action → Execute
                                                ↑
                                                └─ World Model
```

---

## 3. Project Timeline & Deliverables

### **Week 1: Data Collection & World Model Training**

#### **Goals:**

- Collect real environment trajectories
- Train World Model to 95%+ one-step accuracy
- Baseline Evaluation

#### **Detailed Tasks:**

**Task 1.1: Environment Setup (Day 1–2)**

- Install dependencies: `pip install highway-env gymnasium torch`
- Write `environment_wrapper.py`:
  - Initialize `ParkingEnv` from highway-env
  - Define observation/action spaces
  - Implement random agent for data collection
- Deliverable: Running environment with renders

**Task 1.2: Data Collection (Day 3–4)**

- Random action policy (uniform sampling):
  ```python
  action = np.random.uniform(-1, 1, size=2)  # steering, throttle
  ```
- Run 10,000–50,000 steps
- Save data: `data.pkl` with 50K transitions
- Visualize: Plot random trajectories (should look chaotic)
- Deliverable: `data.pkl` file

**Task 1.3: World Model Architecture (Day 5)**

- Define model in PyTorch:
  ```python
  class WorldModel(nn.Module):
      def __init__(self, state_dim=6, action_dim=2, hidden=256):
          super().__init__()
          self.fc1 = nn.Linear(state_dim + action_dim, hidden)
          self.fc2 = nn.Linear(hidden, hidden)
          self.fc3 = nn.Linear(hidden, state_dim)

      def forward(self, state, action):
          x = torch.cat([state, action], dim=-1)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          return self.fc3(x)
  ```
- Deliverable: Model architecture file

**Task 1.4: Training Loop (Day 6–7)**

- Data loader with batch size 64
- Adam optimizer, learning rate 1e-3
- Train for 50 epochs
- Validation set (20% of data)
- Early stopping if val loss plateaus
- Deliverable: Trained `world_model.pt`
- Metrics:
  - One-step MSE: target **< 0.01**
  - Multi-step trajectory RMSE: plot over 10 steps

#### **Week 1 Deliverables:**

- ✅ `data_collection.py` – Generates 50K transitions
- ✅ `world_model.py` – PyTorch model definition + training script
- ✅ `world_model.pt` – Trained weights
- ✅ `analysis/week1_prediction_accuracy.png` – Validation curves
- ✅ README documenting data statistics

#### **Success Criteria:**

- [ ] World Model one-step accuracy > 95%
- [ ] Multi-step accuracy (5 steps) > 90%
- [ ] No NaN values during training
- [ ] Model file size < 1 MB

---

### **Week 2: Planner Implementation & Basic Results**

#### **Goals:**

- Implement CEM-based planner
- First successful parking demonstration
- Establish baseline

#### **Detailed Tasks:**

**Task 2.1: Cost Function Design (Day 1–2)**

- Implement scoring:
  ```python
  def cost(state, target):
      x, y, theta, vx, vy, a = state
      tx, ty, t_theta = target

      pos_cost = ((x - tx)**2 + (y - ty)**2)**0.5
      ang_cost = min(abs(theta - t_theta), 2*pi - abs(theta - t_theta))
      vel_cost = (vx**2 + vy**2)**0.5

      return 1.0 * pos_cost + 0.5 * ang_cost + 0.1 * vel_cost
  ```
- Test with dummy states
- Deliverable: `cost_module.py`

**Task 2.2: Planner Core (Day 3–5)**

- Implement CEM in `planner.py`:
  ```python
  class CEMPlanner:
      def __init__(self, world_model, cost_fn, horizon=10):
          self.wm = world_model
          self.cost = cost_fn
          self.H = horizon

      def plan(self, state, num_samples=1000, iterations=3):
          mu = np.zeros(self.H * 2)  # actions
          sigma = np.ones(self.H * 2)

          for it in range(iterations):
              # Sample trajectories
              samples = np.random.normal(mu, sigma, (num_samples, self.H, 2))

              # Evaluate
              costs = []
              for sample in samples:
                  s = state.copy()
                  total_cost = 0
                  for action in sample:
                      s = self.wm(s, action)
                      total_cost += self.cost(s, self.target)
                  costs.append(total_cost)

              # Elite update
              elite_idx = np.argsort(costs)[:int(0.1 * num_samples)]
              elite_sequences = samples[elite_idx]

              # Update distribution
              mu = elite_sequences.mean(axis=0).flatten()
              sigma = elite_sequences.std(axis=0).flatten()

          return elite_sequences[0][0]  # Best first action
  ```
- Deliverable: `planner.py`

**Task 2.3: Integration (Day 6)**

- Write `agent.py`:
  ```python
  class DreamerAgent:
      def __init__(self, world_model_path, target_position, target_heading):
          self.planner = CEMPlanner(load_model(world_model_path), cost_fn)
          self.target = (target_position, target_heading)

      def act(self, state):
          return self.planner.plan(state, num_samples=1000, iterations=3)
  ```
- Deliverable: `agent.py`

**Task 2.4: Evaluation (Day 7)**

- Run agent on 10 different parking tasks
- Measure:
  - Success rate (%) [agent parked within 0.1m, ±5° heading]
  - Average steps to park
  - Collision rate
  - Min distance to obstacle
- Visualize: Plots of trajectories (planned vs. executed)
- Deliverable: `analysis/week2_parking_success.png`

#### **Week 2 Deliverables:**

- ✅ `cost_module.py` – Cost function
- ✅ `planner.py` – CEM planner
- ✅ `agent.py` – Integration
- ✅ `eval_week2.py` – Evaluation script
- ✅ `analysis/trajectories_week2/` – Plots of 10 parking attempts
- ✅ `results_week2.json` – Success metrics

#### **Success Criteria:**

- [ ] Agent successfully parks >= 5/10 scenarios
- [ ] No crashes in successful attempts
- [ ] Planning time < 2s per decision
- [ ] Trajectories smooth (not jittery)

---

### **Week 3: Advanced Techniques & Robustness**

#### **Goals:**

- Implement MPPI (smoother planning)
- Test on harder scenarios
- Improve success rate to 90%+

#### **Detailed Tasks:**

**Task 3.1: MPPI Upgrade (Day 1–3)**

- Replace CEM with MPPI:
  ```python
  class MPPIPlanner:
      def __init__(self, world_model, cost_fn, horizon=10, temperature=1.0):
          self.wm = world_model
          self.cost = cost_fn
          self.H = horizon
          self.T = temperature

      def plan(self, state, num_samples=1000):
          mu = np.zeros(self.H * 2)

          # Sample from prior
          samples = np.random.normal(mu, 1.0, (num_samples, self.H, 2))

          # Evaluate all trajectories
          costs = []
          for sample in samples:
              s = state.copy()
              total_cost = 0
              for action in sample:
                  s = self.wm(s, action)
                  total_cost += self.cost(s, self.target)
              costs.append(total_cost)

          costs = np.array(costs)

          # Importance weighting (MPPI core)
          weights = np.exp(-self.T * (costs - costs.min()))
          weights /= weights.sum()

          # Weighted mean (mode of distribution)
          mu_new = (samples * weights[:, None, None]).sum(axis=0)

          return mu_new[0]
  ```
- Deliverable: `planner_mppi.py`
- **Improvement:** Smoother trajectories, fewer oscillations

**Task 3.2: Harder Scenarios (Day 4–5)**

- Generate 20 diverse parking tasks:
  - Tight spots (narrow between cars)
  - Angled parking
  - Starting with bad heading
  - Starting far away
- Run agent on each
- Deliverable: `analysis/hard_scenarios/`

**Task 3.3: Failure Analysis (Day 6)**

- Categorize failures:
  - World Model prediction error (rollout diverges from reality)
  - Planner gets stuck (local minima)
  - Actual dynamics differ from learned
- Implement **Uncertainty Tracking**:
  - Track prediction error over time
  - Reduce planning horizon if error grows
- Deliverable: `analysis/failure_modes.md`

**Task 3.4: Hyperparameter Sweep (Day 7)**

- Test variations:
  - Horizon: [5, 10, 15, 20]
  - Samples: [500, 1000, 2000]
  - Temperature: [0.5, 1.0, 2.0]
- Measure: Success rate vs. compute time trade-off
- Deliverable: `analysis/hyperparameter_sweep.png`

#### **Week 3 Deliverables:**

- ✅ `planner_mppi.py` – MPPI implementation
- ✅ `eval_week3_hard.py` – Evaluation on 20 scenarios
- ✅ `analysis/hard_scenarios/` – Trajectory plots
- ✅ `analysis/failure_modes.md` – Root cause analysis
- ✅ `analysis/hyperparameter_sweep.png` – Trade-off curves
- ✅ `results_week3.json` – Updated metrics (target: 90%+ success)

#### **Success Criteria:**

- [ ] Success rate >= 90% on standard parking
- [ ] Success rate >= 70% on hard scenarios
- [ ] Planning time < 1s per decision (optimized)
- [ ] World Model remains accurate (multi-step RMSE < 5% of state range)

---

### **Week 4: Evaluation, Visualization & Final Report**

#### **Goals:**

- Comprehensive benchmarking
- Publication-quality visualizations
- Complete documentation for recruitment

#### **Detailed Tasks:**

**Task 4.1: Baseline Comparisons (Day 1–2)**

- Implement 2 baselines:

  **Baseline 1: Reactive Policy (Behavioral Cloning)**

  - Collect data from expert (highway-env's built-in solution)
  - Train a supervised learning policy: `state -> action`
  - Evaluate success rate
  - Expected: 60–70% success

  **Baseline 2: Random Shooting (Simpler Planner)**

  - Same MPC but just greedy: sample 100 actions, pick best first action (no optimization loop)
  - Expected: 40–50% success
- Deliverable: `baselines/` directory

**Task 4.2: Comprehensive Evaluation (Day 3–4)**

- Run all 3 approaches (Dreamer, BC, Random Shooting) on:
  - 50 random parking scenarios
  - Measure:
    - Success rate (%)
    - Average steps to success
    - Collisions (total, during attempt)
    - Compute time per decision
    - Max distance to nearest obstacle during trajectory
- Create comparison table
- Deliverable: `analysis/comparison_table.txt`

**Task 4.3: Visualization & Demo (Day 5–6)**

- Generate 4 publication-quality videos:

  1. **"Agent Dreaming"** – Show imagined trajectories (ghost cars) overlaid on real execution
  2. **"Success Cases"** – 3 diverse successful parkings
  3. **"Failure Analysis"** – 2 failure modes with explanation
  4. **"Comparison"** – Side-by-side: Dreamer vs. Baseline
- Render high-quality frames (2x resolution)
- Add annotations:

  - Cost values over time
  - Predicted vs. actual position
  - Planning iterations
- Deliverable: `videos/` directory with `.mp4` files

**Task 4.4: Academic-Style Report (Day 7)**

- Write final report `REPORT.md`:

  **Sections:**

  - **1. Introduction:** Problem statement, related work (LeCun, JEPA, MuZero)
  - **2. Method:** Architecture, equations, algorithms
  - **3. Experiments:** Data, baselines, metrics
  - **4. Results:** Tables, figures, ablations
  - **5. Analysis:** What worked? What didn't? Why?
  - **6. Future Work:** Multi-agent, sim-to-real, continuous learning
  - **7. Conclusion:** How this aligns with AMI philosophy
- Include:

  - All plots from weeks 1–4
  - Comparison table
  - Hyperparameter justifications
  - Code snippets explaining key parts
- Deliverable: `REPORT.md` (~2,000 words)

**Task 4.5: Code Organization & Documentation (Day 7)**

- Clean up all code:
  ```
  dreamer-park/
  ├── README.md                    # Quick start
  ├── REPORT.md                    # Full technical report
  ├── requirements.txt             # Dependencies
  ├── src/
  │   ├── environment_wrapper.py
  │   ├── world_model.py
  │   ├── cost_module.py
  │   ├── planner.py               # CEM
  │   ├── planner_mppi.py          # MPPI (advanced)
  │   ├── agent.py
  │   └── utils.py
  ├── scripts/
  │   ├── 1_collect_data.py
  │   ├── 2_train_world_model.py
  │   ├── 3_evaluate_agent.py
  │   ├── 4_compare_baselines.py
  │   └── 5_generate_visualizations.py
  ├── configs/
  │   ├── default.yaml             # Hyperparameters
  │   └── hard_scenarios.yaml
  ├── models/
  │   └── world_model_week1.pt
  ├── data/
  │   └── raw_trajectories.pkl
  ├── results/
  │   ├── week1/
  │   ├── week2/
  │   ├── week3/
  │   └── week4/
  ├── analysis/
  │   ├── plots/
  │   ├── videos/
  │   └── failure_modes.md
  └── .gitignore
  ```
- Add docstrings to every function
- Create Jupyter notebook: `demo.ipynb` (interactive walkthrough)
- Deliverable: Clean GitHub-ready repo

#### **Week 4 Deliverables:**

- ✅ `baselines/` – BC and Random Shooting implementations
- ✅ `analysis/comparison_table.txt` – Quantitative results
- ✅ `videos/` – 4 high-quality demo videos
- ✅ `REPORT.md` – Complete technical writeup
- ✅ `README.md` – Instructions to reproduce
- ✅ `demo.ipynb` – Interactive walkthrough
- ✅ Full clean GitHub repo structure

#### **Success Criteria:**

- [ ] Dreamer >> Baselines (ideally 2–3x better success rate)
- [ ] All videos render cleanly
- [ ] Report is publication-grade (can be submitted to arxiv.org)
- [ ] Code is documented and reproducible
- [ ] README has clear "Quick Start" section

---

## 4. Technical Stack & Dependencies

### **Core Dependencies:**

```
torch>=2.0.0                # Neural networks
gymnasium>=0.28.0           # RL environments
highway-env>=1.8            # Parking environment
numpy>=1.24.0
matplotlib>=3.7.0           # Visualization
scikit-learn>=1.3.0         # Preprocessing (StandardScaler)
jupyter>=1.0.0              # Notebooks
opencv-python>=4.8.0        # Video rendering
```

### **Installation (Colab Cell 1):**

```bash
!pip install torch gymnasium highway-env numpy matplotlib scikit-learn jupyter opencv-python
!git clone https://github.com/farama-foundation/highway-env.git
```

### **Hardware Requirements:**

- **Minimum:** Google Colab Free Tier (2 vCPU, 12 GB RAM)
- **Recommended:** Colab Pro (T4 GPU, 28 GB RAM) – Training 5x faster
- **Local:** Laptop CPU sufficient (slower, but works)

### **Estimated Runtime:**

| Phase                    | Hardware  | Time                |
| ------------------------ | --------- | ------------------- |
| Week 1 (Data + Training) | Colab T4  | 2 hours             |
| Week 2 (Planner)         | Colab CPU | 3 hours             |
| Week 3 (MPPI + Eval)     | Colab T4  | 4 hours             |
| Week 4 (Visualization)   | Colab CPU | 2 hours             |
| **Total**          | —        | **~11 hours** |

---

## 5. Key Innovation Highlights (For Recruiters)

### **Why This Stands Out:**

1. **Aligns with Industry Frontier:**

   - Implements Yann LeCun's "AMI" (Advanced Machine Intelligence) philosophy
   - Same architecture as OpenAI o1/o3 ("test-time compute")
   - Model-based RL is bleeding-edge (not the outdated DDPG/PPO)
2. **Technical Depth:**

   - Custom PyTorch implementation (not just API calls)
   - Probabilistic planning (CEM + MPPI)
   - Uncertainty quantification in world model predictions
   - Hyperparameter ablation studies
3. **Production-Ready Thinking:**

   - Modular architecture (easily swap components)
   - Quantified metrics (success rate, compute time trade-offs)
   - Failure analysis (not just cherry-picked successes)
   - Comparison to baselines (rigorous evaluation)
4. **Research-Grade Documentation:**

   - Technical report suitable for arxiv.org submission
   - Reproducible code with clean structure
   - Comprehensive ablation studies
5. **Scalable Insights:**

   - Techniques transfer to:
     - Multi-agent coordination
     - Sim-to-real robotics
     - Continuous learning / online adaptation
     - Hierarchical planning (short-term + long-term reasoning)

---

## 6. Recruitment Talking Points

### **For Machine Learning Roles (Google, DeepMind, OpenAI):**

> *"I implemented a model-based planning agent using learned world models and test-time compute—directly inspired by Yann LeCun's recent work on Advanced Machine Intelligence. The key insight is that instead of training a policy, we learn environment dynamics and use probabilistic search (CEM/MPPI) at inference time. This approach requires far fewer environment interactions and generalizes better to unseen scenarios. The project includes baselines, ablations, and achieves 90%+ success on autonomous parking tasks."*

### **For Robotics Roles (Boston Dynamics, Tesla, Waymo):**

> *"I built a planner that grounds itself in a learned simulator. Rather than hand-coded heuristics or reactive policies, the agent imagines trajectories and selects actions that minimize a cost function. This is directly applicable to real-world robotics: the approach is sample-efficient (needs few environment interactions), handles constraints gracefully (collision avoidance), and adapts to unseen scenarios through planning."*

### **For AI Safety/Research Roles:**

> *"The project explores model-based AI that is interpretable and grounded in learned world dynamics. Unlike end-to-end RL (black-box), each component is transparent: the world model makes explicit predictions, the cost function defines objectives, and the planner makes deliberative decisions. This alignment with the objective is crucial for safe AI systems."*

---

## 7. Optional Extensions (If Ahead of Schedule)

If you finish early or want to go deeper:

1. **Uncertainty Quantification:**

   - Train ensemble of World Models (5–10 models)
   - Use disagreement to detect uncertainty
   - Plan conservatively in high-uncertainty regions
2. **Hierarchical Planning:**

   - High-level planner (where to park)
   - Low-level planner (steering trajectory)
   - Multi-timescale reasoning
3. **Sim-to-Real Simulation:**

   - Add noise/dynamics randomization during training
   - Test robustness to model mismatch
4. **Multi-Agent Parking:**

   - 2–3 cars parking simultaneously
   - Incorporate other agents' actions into planning
5. **Paper Submission:**

   - Write arxiv preprint
   - Submit to top-tier venues (CoRL, ICLR)

---

## 8. Resources & Reading List

### **Core Papers:**

- Yann LeCun, "Self-Supervised Learning: The Dark Matter of Intelligence" (2022)
  - https://openreview.net/forum?id=B1xwMsEtvH
- "Dreamer: Scalable and Efficient RL with World Models" (Hafner et al., 2020)
  - https://arxiv.org/abs/1912.01603
- "Learning Latent Dynamics for Efficient Visual Exploration" (JEPA Foundation, Tian et al., 2023)
  - https://arxiv.org/abs/2306.02342

### **Algorithmic Reference:**

- "Model Predictive Path Integral" (Theodorou et al., 2010)
  - https://ieeexplore.ieee.org/abstract/document/5717754
- "Cross-Entropy Method" (de Boer et al., 2005)
  - https://en.wikipedia.org/wiki/Cross-entropy_method

### **Implementation Tutorials:**

- Deep RL Bootcamp (UC Berkeley): https://sites.google.com/berkeley.edu/deep-rl-bootcamp/
- Highway-env Documentation: https://highway-env.farama.org/

---

## 9. Success Metrics & Final Checklist

### **Technical Success:**

- [ ] World Model achieves < 0.05 MSE on validation set
- [ ] Agent successfully parks in >= 90% of standard scenarios
- [ ] Planning runs at > 0.5 Hz (< 2 seconds per decision)
- [ ] Dreamer outperforms baselines by >= 2x

### **Code Quality:**

- [ ] All functions documented with docstrings
- [ ] Modular design: easy to swap components
- [ ] No hardcoded constants (all in config file)
- [ ] Reproducible: anyone can run `python scripts/1_collect_data.py`

### **Documentation:**

- [ ] README with quick-start (5 min to first run)
- [ ] REPORT.md with full technical writeup (~2,000 words)
- [ ] Jupyter notebook with interactive walkthrough
- [ ] All plots and videos saved with captions

### **Recruitment Readiness:**

- [ ] GitHub repo is clean and well-organized
- [ ] README highlights key innovations
- [ ] A recruiter can understand the project in 3 minutes
- [ ] Code demonstrates PyTorch expertise
- [ ] Architecture aligns with cutting-edge research (AMI, test-time compute)

---

## 10. Conclusion

**Dreamer-Park** is not a toy project—it's a **working implementation of advanced AI concepts** at the frontier of machine learning research.

By the end of Week 4, you will have:

- ✅ A trained neural network that learned physics
- ✅ A probabilistic planner that reasons about the future
- ✅ Comprehensive benchmarking showing it outperforms simpler approaches
- ✅ Publication-quality documentation and visualizations
- ✅ A portfolio piece that demonstrates you understand:
  - Model-based RL (not just model-free)
  - Planning algorithms (CEM, MPPI)
  - PyTorch implementation depth
  - Rigorous experimental methodology

**This is the kind of project that gets you interviews at FAANG + DeepMind + OpenAI + Robotics companies.**

Good luck. Let's build the future of AI. 🚀

---

**Questions? Start with Week 1. Good luck!**
