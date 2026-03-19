# serverless-sim

SimPy-based serverless system simulator with Gymnasium wrapper and Stable-Baselines3 PPO training.

## Overview

`serverless-sim` models a serverless computing platform inspired by Apache OpenWhisk. It simulates the full lifecycle of function invocations — from Poisson-driven request arrival, through consistent-hash load balancing, container cold starts, concurrent execution, to autoscaling and eviction — all built on the [SimPy](https://simpy.readthedocs.io/) discrete-event simulation framework.

The simulator exposes a [Gymnasium](https://gymnasium.farama.org/) environment, allowing reinforcement learning agents (e.g., PPO via Stable-Baselines3) to learn autoscaling policies by controlling prewarm counts and idle timeouts.

## Architecture

```
                     +-----------+
                     |  CLI/App  |
                     +-----+-----+
                           |
                  +--------v--------+
                  | SimulationBuilder|
                  +--------+--------+
                           |
           +---------------+---------------+
           |               |               |
    +------v------+ +------v------+ +------v------+
    |  Workload   | |   Cluster   | |  Lifecycle  |
    |  Manager    | |   Manager   | |  Manager    |
    +------+------+ +------+------+ +------+------+
           |               |               |
           v               v               v
    PoissonGenerator    Node(s)      StateMachine
           |               |         ContainerInstance
           v               |
    +------+------+        |
    | LoadBalancer +------->+
    +-------------+        |
                           v
                    +------+------+
                    | Autoscaler  |
                    +------+------+
                           |
                    +------v------+
                    | Controller  |<--- ThresholdPolicy
                    +------+------+
                           |
                    +------v------+
                    |  Monitoring |
                    +------+------+
                           |
                    +------v------+
                    |   Export    |
                    +-------------+
```

## Quick Start

### Prerequisites

- Python >= 3.10
- conda environment (recommended)

### Installation

```bash
conda create -n sl-venv python=3.11
conda activate sl-venv
pip install -e .
```

### Run a Simulation

```bash
python -m serverless_sim simulate \
    --sim-config configs/simulation/sample_minimal.json \
    --export-mode 2 \
    --log-mode both
```

Output goes to `logs/run_<timestamp>/` containing:
- `simulation.log` — runtime log
- `summary.txt` — request statistics and latency percentiles
- `system_metrics.csv` — time-series cluster/request metrics (mode >= 1)
- `request_trace.csv` — per-request lifecycle trace (mode 2)

### Train a PPO Agent

```bash
python -m serverless_sim train \
    --sim-config configs/simulation/sample_minimal.json \
    --gym-config configs/gym/sample_gym_discrete.json \
    --rl-config configs/rl/sample_ppo_train.json
```

### Run Inference with Trained Model

```bash
python -m serverless_sim infer \
    --sim-config configs/simulation/sample_minimal.json \
    --gym-config configs/gym/sample_gym_discrete.json \
    --rl-config configs/rl/sample_ppo_infer.json
```

## Project Structure

```
serverless_sim/
  core/           Config loader, SimContext, SimulationBuilder, SimulationEngine
  cluster/        Node, ResourceProfile, ClusterManager, ServingModel
  workload/       ServiceClass, Invocation, PoissonGenerator, WorkloadManager
  scheduling/     ShardingContainerPoolBalancer (consistent-hash load balancer)
  lifecycle/      StateMachine, ContainerInstance, LifecycleManager
  autoscaling/    OpenWhiskPoolAutoscaler, AutoscalingAPI
  controller/     BaseController, ThresholdPolicy
  monitoring/     MetricStore, Collectors, MonitorManager, MonitorAPI
  export/         SummaryWriter, SystemMetricsExporter, RequestTraceExporter
  gym_env/        ServerlessGymEnv, ObservationBuilder, ActionMapper, RewardCalculator
  rl_agent/       PPO training (train.py) and inference (infer.py)
  runtime/        CLI (cli.py) and app entry point (app.py)
configs/
  simulation/     Simulation configs (minimal, extended states, multi-service)
  gym/            Gymnasium environment configs
  rl/             RL training and inference configs
tests/            Pytest test suite
docs/             Detailed design documentation
```

## Configuration

See [docs/config_reference.md](docs/config_reference.md) for the full configuration reference.

## Documentation

- [Design Overview](docs/design_overview.md) — system architecture and module responsibilities
- [Request Flow](docs/request_flow.md) — detailed request lifecycle from arrival to completion
- [Edge Cases](docs/edge_cases.md) — timeout, resource exhaustion, drain period, and other edge cases
- [Configuration Reference](docs/config_reference.md) — all configuration keys and their meaning
- [Gym Wrapper](docs/gym_wrapper.md) — Gymnasium environment, observations, actions, and rewards
- [RL Training](docs/rl_training.md) — PPO training and inference guide

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
