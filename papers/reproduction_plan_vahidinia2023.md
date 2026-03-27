# Reproduction Plan: Vahidinia et al. 2023

**Paper:** "Mitigating Cold Start Problem in Serverless Computing: A Reinforcement Learning Approach"
**Target:** IEEE Internet of Things Journal, DOI 10.1109/JIOT.2022.3165127

---

## 1. Overview

The paper proposes a two-layer approach to reduce cold start delay in serverless computing:
- **Layer 1 (RL):** TD Advantage Actor-Critic to dynamically set the idle-container window
- **Layer 2 (LSTM):** Predict concurrent invocations to determine pre-warmed container count

We reproduce this using the existing `sl-simulation` SimPy-based simulator, which already provides container lifecycle management, autoscaling with idle timeout control, Gymnasium RL integration, and Poisson workload generation.

---

## 2. Gap Analysis: Existing Infrastructure vs. Paper Requirements

| Component | Paper Requires | Codebase Has | Gap |
|-----------|---------------|-------------|-----|
| Simulation engine | Discrete-event serverless sim | SimPy-based engine with full lifecycle | None |
| Container lifecycle | null -> prewarm -> warm -> running -> evicted | OpenWhiskExtendedStateMachine | None |
| Idle-container window | Dynamic, continuous (3-15 min) | `idle_timeout` via AutoscalingAPI | Need continuous action space |
| Workload generation | Poisson process (5, 10, 20/hr) | PoissonFixedSizeGenerator | None |
| RL algorithm | TD Advantage Actor-Critic (A2C) | PPO (Stable-Baselines3) | Add A2C agent variant |
| State space | Inter-arrival times + cold/warm flag | Observation builder (configurable metrics) | Extend observation builder |
| Action space | Continuous idle-container window value | Discrete (inc/dec by 5s) | Add continuous action mapper |
| Reward function | -(Cold/N) - P | Weighted penalty sum | Add paper-specific reward |
| LSTM predictor | Predict max concurrent invocations | Not implemented | Build LSTM predictor module |
| Pre-warm scaling | Adjust prewarm pool from LSTM output | pool_target via AutoscalingAPI | Wire LSTM output to API |
| Baselines | OpenWhisk default (10-min fixed window) | ThresholdPolicy, min/max baselines | Add fixed-window baseline |
| Metrics | Cold starts, memory consumption, CDF plots | RequestStore counters, MetricStore timeseries | Add CDF export/plotting |

---

## 3. Implementation Plan

### Phase 1: Workload & Baseline Setup

#### Task 1.1: Create Paper-Specific Simulation Configs

Create 3 configs for arrival rates 5, 10, 20 invocations/hour:
- **File:** `configs/paper_vahidinia/rate_{5,10,20}.json`
- Single service, single node (matching OpenWhisk setup)
- Poisson generator with `arrival_rate` = 5/10/20 per hour (convert to per-second: rate/3600)
- `job_size` calibrated so response time ~ 6 seconds (I/O-bound function per paper)
- Container lifecycle: `null -> prewarm -> warm -> running -> evicted`
- Default OpenWhisk idle-container window: 600s (10 minutes)
- 10,000 invocations per experiment (duration = 10000 / (rate/3600) seconds)
- Seed: fixed for reproducibility

#### Task 1.2: Implement Fixed-Window Baseline (OpenWhisk Default)

- Already supported: set `idle_timeout: 600` in config with no controller
- Verify 2 default pre-warmed containers (OpenWhisk default) via `min_instances: 2`
- Run baseline simulations for rates 5, 10, 20
- Collect: cold_start count, memory consumption (resource-seconds), invocations on pre-warmed containers

---

### Phase 2: Layer 1 - Actor-Critic for Idle-Container Window

#### Task 2.1: Extend Observation Builder for Paper State Space

**File:** `src/gym_env/observation_builder.py`

Add a new observation mode or config option that produces the paper's state:
- **Inter-arrival time** (time since last invocation for the service)
- **Cold/warm flag** (1 if last invocation was cold start, 0 if warm)

Implementation:
- Add `inter_arrival_time` metric to MonitorManager or compute in observation builder
- Track last invocation timestamp per service in RequestStore or a new collector
- Config: `"observation_metrics": ["inter_arrival_time", "last_cold_start_flag"]`

#### Task 2.2: Implement Continuous Action Mapper

**File:** `src/gym_env/action_mapper.py` (extend or new class)

The paper uses a continuous action space where the action is the idle-container window value directly:
- `gymnasium.spaces.Box(low=min_idle, high=max_idle, shape=(1,))`
- Action maps directly to `autoscaling_api.set_idle_timeout(service_id, action_value)`
- Bounds: `[180, 900]` seconds (3 to 15 minutes, from paper's results)
- Create `ContinuousActionMapper` class

#### Task 2.3: Implement Paper Reward Function

**File:** `src/gym_env/reward_calculator.py` (extend or new class)

Paper reward: `Reward = -(Cold / N) - P`
- `Cold`: number of cold starts in the episode/step
- `N`: total invocations in the episode/step
- `P`: penalty for memory wastage (proportional to idle-container time beyond needed)

Implementation:
- `PaperRewardCalculator` class
- `cold_ratio = new_cold_starts / max(total_invocations, 1)`
- `memory_penalty = alpha * (idle_container_time - actual_inter_arrival_time)` when container sits idle
- Config: `"reward_type": "paper_vahidinia"`

#### Task 2.4: Add A2C Training Script

**File:** `src/rl_agent/train_a2c.py`

Paper uses TD Advantage Actor-Critic:
- Use `stable_baselines3.A2C` (already available if SB3 is installed)
- Actor network: 3 hidden layers, 32 neurons each, ReLU activation
- Critic network: 2 layers (32, 16 neurons), ReLU
- Actor output: Gaussian policy (mu, sigma) - SB3 handles this natively for continuous actions
- Config:
  ```json
  {
    "algorithm": "A2C",
    "policy": "MlpPolicy",
    "policy_kwargs": {
      "net_arch": {
        "pi": [32, 32, 32],
        "vf": [32, 16]
      },
      "activation_fn": "ReLU"
    },
    "learning_rate": 0.0007,
    "gamma": 0.99,
    "total_timesteps": 100000
  }
  ```
- Training: episodes = sequences of 200 invocations (matching paper's test dataset)
- Save model to `models/a2c_idle_window/`

#### Task 2.5: CLI Integration

Extend `runtime/cli.py` and `runtime/app.py`:
- Support `--algorithm a2c` flag in train/infer commands
- Load appropriate action mapper (continuous vs discrete) based on config

---

### Phase 3: Layer 2 - LSTM for Pre-Warmed Container Prediction

#### Task 3.1: Build LSTM Predictor Module

**File:** `src/lstm_predictor/model.py`

Architecture (from paper):
- Input: sequence of past invocation counts (windowed time series)
- 5 hidden layers, 32 neurons each, ReLU activation
- Dropout layer (0.5) to prevent overfitting
- Output: 1 neuron (predicted max concurrent invocations), linear activation
- Loss: MSE

```python
class InvocationPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=5, dropout=0.5):
        # LSTM layers + linear output
        ...
    def forward(self, x):
        # Return predicted max concurrent invocations
        ...
```

#### Task 3.2: LSTM Data Collection & Training Pipeline

**File:** `src/lstm_predictor/train.py`

- **Data collection:** Run baseline simulations, export request traces with timestamps
- **Feature engineering:**
  - Window invocations into time buckets (e.g., 1-minute intervals)
  - Compute max concurrent invocations per bucket
  - Create sliding window sequences (e.g., past 10 buckets -> predict next bucket)
- **Training:**
  - Train/val/test split (70/15/15)
  - PyTorch DataLoader
  - Adam optimizer, MSE loss
  - Early stopping on validation loss
  - Save model to `models/lstm_predictor/`

#### Task 3.3: LSTM-Based Pre-Warm Controller

**File:** `src/controller/lstm_controller.py`

Integrate LSTM predictions into the simulation loop:
- Periodically (e.g., every control interval):
  1. Collect recent invocation history from MonitorAPI
  2. Feed into trained LSTM model
  3. Get predicted max concurrent invocations
  4. Set `pool_target` for prewarm state = predicted value
- Use `autoscaling_api.set_pool_target(service_id, "prewarm", predicted_count)`
- Register as a BaseController subclass

---

### Phase 4: Two-Layer Integration

#### Task 4.1: Combined Controller

**File:** `src/controller/two_layer_controller.py`

Combine Layer 1 (A2C) and Layer 2 (LSTM) into a single controller:
1. A2C agent sets `idle_timeout` (continuous action from trained policy)
2. LSTM predictor sets `pool_target` for pre-warmed containers
3. Both operate on the same control interval
4. Config references both trained models

```json
{
  "controller": {
    "type": "two_layer",
    "a2c_model_path": "models/a2c_idle_window/best_model.zip",
    "lstm_model_path": "models/lstm_predictor/best_model.pt",
    "interval": 5
  }
}
```

#### Task 4.2: Evaluation Config

Create evaluation configs that wire everything together:
- `configs/paper_vahidinia/eval_rate_{5,10,20}.json`
- Two-layer controller enabled
- Export mode 2 (full request traces + system metrics)
- Same Poisson workload as baselines

---

### Phase 5: Experiments & Evaluation

#### Task 5.1: Experiment Definitions

**File:** `experiments/vahidinia_reproduction.json`

Use `tools/run_experiments.py` to define experiment matrix:

| Experiment | Controller | Rate | Metric Focus |
|-----------|-----------|------|-------------|
| baseline_r5 | None (fixed 10-min window) | 5/hr | All |
| baseline_r10 | None (fixed 10-min window) | 10/hr | All |
| baseline_r20 | None (fixed 10-min window) | 20/hr | All |
| proposed_r5 | Two-layer (A2C + LSTM) | 5/hr | All |
| proposed_r10 | Two-layer (A2C + LSTM) | 10/hr | All |
| proposed_r20 | Two-layer (A2C + LSTM) | 20/hr | All |

#### Task 5.2: Metrics Collection

For each experiment, collect:
1. **Cold start count** per time interval (from request traces)
2. **Memory consumption** (idle-container time * memory per container)
3. **Idle-container window** over time (from autoscaling metrics)
4. **Number of invocations on pre-warmed containers** (cold_start=False on first use)
5. **CDF of idle-container window** values
6. **CDF of cold start occurrences** per hour

#### Task 5.3: Plotting & Comparison Scripts

**File:** `tools/plot_vahidinia.py`

Reproduce paper figures:
- **Fig. 6:** Idle-container window over time (proposed vs OpenWhisk) for 3 rates
- **Fig. 7:** CDF of idle-container window for 3 rates
- **Fig. 8:** CDF of cold start occurrences for 3 rates
- **Fig. 9:** Idle-container time comparison for 3 rates
- **Fig. 10:** Pre-warmed container invocations over time
- **Table II:** Cold start count comparison (proposed vs OpenWhisk)
- **Memory improvement %:** `(baseline_memory - proposed_memory) / baseline_memory * 100`

Target numbers from paper:

| Rate | Paper Memory Improvement | Paper Pre-warm Improvement |
|------|------------------------|--------------------------|
| 5/hr | 11.11% | - |
| 10/hr | 12.73% | - |
| 20/hr | 4.05% | - |
| Combined | - | 22.65% |

---

## 4. Implementation Order & Dependencies

```
Phase 1 (Baseline)          Phase 2 (RL Layer 1)         Phase 3 (LSTM Layer 2)
  1.1 Configs ──────────────► 2.1 Observation Builder       3.1 LSTM Model
  1.2 Run baselines           2.2 Continuous Actions        3.2 Training Pipeline
         │                    2.3 Paper Reward               3.3 LSTM Controller
         │                    2.4 A2C Training                      │
         │                    2.5 CLI Integration                   │
         │                           │                              │
         │                           ▼                              ▼
         │                    Phase 4 (Integration)
         │                      4.1 Combined Controller ◄───────────┘
         │                      4.2 Eval Configs
         │                           │
         ▼                           ▼
                              Phase 5 (Evaluation)
                                5.1 Run Experiments
                                5.2 Collect Metrics
                                5.3 Plot & Compare
```

Phase 2 and Phase 3 can be developed in **parallel** as they are independent.

---

## 5. Key Design Decisions

### 5.1 Simulation Duration
- Paper: 200 requests per test, 10,000 for training
- At 5/hr: 200 requests = 40 hours simulated time
- At 10/hr: 200 requests = 20 hours
- At 20/hr: 200 requests = 10 hours
- Training: scale accordingly

### 5.2 Time Granularity
- Paper's idle-container window ranges 3-15 minutes
- Monitor interval: 1s (sufficient resolution)
- Control interval: match paper's per-invocation decision or use 5s periodic

### 5.3 Memory Metric
- Paper measures "memory consumption" as idle-container time (how long warm containers sit unused)
- Use `LifecycleManager` resource-seconds tracking (already implemented)
- Or compute from system metrics: sum of (idle_timeout - actual_wait_before_next_invocation) per container

### 5.4 A2C vs PPO
- Paper specifies TD Advantage Actor-Critic
- SB3 provides both `A2C` and `PPO`; use `A2C` for fidelity
- Can also run PPO as an ablation to compare

### 5.5 Continuous vs Discrete Action Space
- Paper: continuous (Gaussian policy over idle window value)
- Current codebase: discrete (increment/decrement by 5s)
- Must implement continuous mapper for paper fidelity
- Keep discrete as ablation option

---

## 6. Validation Criteria

The reproduction is successful if:
1. Idle-container window dynamically adapts to invocation patterns (Fig. 6 shape)
2. Memory improvement is within reasonable range of paper values (~5-15%)
3. Pre-warmed container utilization improves over baseline
4. CDF distributions show similar trends to paper figures
5. The two-layer approach outperforms either layer alone

**Note:** Exact numerical reproduction is unlikely due to:
- Different simulation engine (SimPy vs actual OpenWhisk)
- Simplified I/O model (no real HTTP calls)
- Potentially different random seeds
- Paper uses actual OpenWhisk platform; we use simulation

---

## 7. Estimated File Changes

| File | Action | Description |
|------|--------|-------------|
| `configs/paper_vahidinia/*.json` | Create | 6 config files (3 baseline + 3 eval) |
| `src/gym_env/observation_builder.py` | Modify | Add inter-arrival time, cold flag metrics |
| `src/gym_env/action_mapper.py` | Modify | Add ContinuousActionMapper class |
| `src/gym_env/reward_calculator.py` | Modify | Add PaperRewardCalculator class |
| `src/gym_env/serverless_gym_env.py` | Modify | Support continuous action space selection |
| `src/rl_agent/train_a2c.py` | Create | A2C training script |
| `src/lstm_predictor/model.py` | Create | LSTM model definition |
| `src/lstm_predictor/train.py` | Create | LSTM training pipeline |
| `src/controller/lstm_controller.py` | Create | LSTM-based pre-warm controller |
| `src/controller/two_layer_controller.py` | Create | Combined A2C + LSTM controller |
| `src/runtime/cli.py` | Modify | Add algorithm flag, LSTM commands |
| `src/runtime/app.py` | Modify | Wire new controllers |
| `tools/plot_vahidinia.py` | Create | Reproduction plots |
| `experiments/vahidinia_reproduction.json` | Create | Experiment definitions |
| **Total** | | ~10 new files, ~5 modified files |
