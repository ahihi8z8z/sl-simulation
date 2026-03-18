# Implementation Plan

Mỗi step tạo ra code chạy được, test được trước khi sang step tiếp theo.

---

## Step 0: Project skeleton

**Mục tiêu:** Tạo toàn bộ cấu trúc thư mục, file `__init__.py`, `pyproject.toml` (hoặc `requirements.txt`), và stub cho tất cả module. Chạy `python -m serverless_sim` không lỗi import.

**File tạo:**
```
serverless_sim/
├── __init__.py
├── __main__.py
├── runtime/
│   ├── __init__.py
│   ├── cli.py              # argparse stub: simulate/train/infer
│   └── app.py              # placeholder
├── core/
│   ├── __init__.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── sim_context.py   # SimContext dataclass stub
│   │   ├── sim_builder.py   # SimulationBuilder stub
│   │   └── sim_engine.py    # SimulationEngine stub
│   ├── config/
│   │   ├── __init__.py
│   │   └── loader.py        # load_config() stub
│   ├── logging/
│   │   ├── __init__.py
│   │   └── logger_factory.py
│   └── utils/
│       └── __init__.py
├── workload/
│   ├── __init__.py
│   ├── service_class.py
│   ├── invocation.py
│   ├── generators.py
│   └── workload_manager.py
├── scheduling/
│   ├── __init__.py
│   └── load_balancer.py     # ShardingContainerPoolBalancer stub
├── cluster/
│   ├── __init__.py
│   ├── resource_profile.py
│   ├── compute_class.py
│   ├── node.py
│   ├── serving_model.py
│   └── cluster_manager.py
├── lifecycle/
│   ├── __init__.py
│   ├── state_definition.py
│   ├── transition_definition.py
│   ├── container_instance.py
│   ├── state_machine.py
│   └── lifecycle_manager.py
├── autoscaling/
│   ├── __init__.py
│   ├── autoscaler.py
│   └── autoscaling_api.py
├── monitoring/
│   ├── __init__.py
│   ├── metric_store.py
│   ├── collectors.py
│   ├── monitor_manager.py
│   └── monitor_api.py
├── controller/
│   ├── __init__.py
│   ├── base_controller.py
│   └── policies/
│       ├── __init__.py
│       ├── base_policy.py
│       └── threshold_policy.py
├── export/
│   ├── __init__.py
│   ├── batch_csv_writer.py
│   ├── summary_writer.py
│   ├── system_metrics_exporter.py
│   ├── request_trace_exporter.py
│   └── export_manager.py
├── gym_env/
│   ├── __init__.py
│   ├── serverless_gym_env.py
│   ├── observation_builder.py
│   ├── action_mapper.py
│   └── reward_calculator.py
└── rl_agent/
    ├── __init__.py
    └── train.py
configs/
├── simulation/
├── gym/
└── rl/
logs/
```

**Test:** `python -m serverless_sim --help` hiện usage message.

---

## Step 1: Config loader + logging + run directory

**Mục tiêu:** Load JSON config, validate, tạo run directory, tạo logger.

**File code:**
- `core/config/loader.py` — load JSON, validate required keys, trả về dict/dataclass
- `core/logging/logger_factory.py` — `create_logger(module_name, run_dir, mode, level)`
- `core/simulation/sim_context.py` — SimContext với `env`, `config`, `rng`, `logger`, `run_dir`

**Config tạo:**
- `configs/simulation/sample_minimal.json` — 1 service, 1 compute class, minimal settings

**Test:** Chạy `python -m serverless_sim simulate --sim-config configs/simulation/sample_minimal.json` → tạo được run directory trong `logs/`, ghi file log, in ra "Config loaded successfully".

---

## Step 2: Resource model + Cluster + Node (no lifecycle yet)

**Mục tiêu:** Node có thể nhận request vào queue, có resource accounting.

**File code:**
- `cluster/resource_profile.py` — ResourceProfile(cpu, memory) với add/subtract/can_fit
- `cluster/compute_class.py` — ComputeClass dataclass
- `cluster/node.py` — Node với `simpy.Store` queue, resource tracking, pull loop (chỉ log "received request", chưa xử lý)
- `cluster/serving_model.py` — FixedRateModel: `service_time = job_size * processing_factor`
- `cluster/cluster_manager.py` — tạo nodes từ config, `get_enabled_nodes()`, `get_node()`

**Test:** Unit test — tạo ClusterManager từ config, verify nodes được tạo đúng, put request vào Store, node pull ra được.

---

## Step 3: Workload generation

**Mục tiêu:** Tạo request theo Poisson, đẩy vào hệ thống.

**File code:**
- `workload/service_class.py` — ServiceClass dataclass
- `workload/invocation.py` — Invocation dataclass với tất cả fields + drop_reason
- `workload/generators.py` — PoissonFixedSizeGenerator: SimPy process, exponential inter-arrival
- `workload/workload_manager.py` — quản lý services, start generators

**Test:** Chạy standalone simulation 10s, workload tạo ra N requests (verify N ~ arrival_rate * 10), tất cả request nằm trong request table với arrival_time hợp lệ.

---

## Step 4: Load balancer (ShardingContainerPoolBalancer)

**Mục tiêu:** Request từ workload → load balancer → node queue.

**File code:**
- `scheduling/load_balancer.py` — consistent hash ring trên service_id, memory check realtime, fallback walk, drop nếu no_capacity

**Test:** Chạy simulation — request được hash đến đúng node, verify affinity (cùng service → cùng node). Test fallback khi 1 node hết memory. Test drop khi tất cả node hết memory.

---

## Step 5: Minimal lifecycle + execution

**Mục tiêu:** Request được xử lý end-to-end: arrive → queue → instance → execute → complete.

**File code:**
- `lifecycle/state_definition.py` — StateDefinition, stable/transient classification
- `lifecycle/transition_definition.py` — TransitionDefinition
- `lifecycle/container_instance.py` — ContainerInstance với `simpy.Resource(capacity=max_concurrency)`, state tracking
- `lifecycle/state_machine.py` — minimal: null → prewarm → warm → running (transient) → warm → evicted
- `lifecycle/lifecycle_manager.py` — find_reusable_instance, prepare_instance, start/finish execution, warm↔running transitions, per-request CPU allocate/release

**Tích hợp:** Node pull loop gọi lifecycle_manager để xử lý request.

**Test:** Chạy simulation end-to-end — request arrive, dispatch, execute, complete. Verify:
- Invocation có đầy đủ timestamps (arrival, dispatch, queue_enter, exec_start, exec_end, completion)
- Cold start đúng (request đầu tiên = cold, sau đó warm hit)
- Concurrency hoạt động (max_concurrency > 1 → nhiều request chạy song song trên 1 instance)
- Per-request CPU allocated/released đúng

---

## Step 6: Timeout handling

**Mục tiêu:** Request bị timeout khi quá thời gian.

**File code:**
- Cập nhật lifecycle_manager / node pull loop: SimPy event race giữa execution và timeout
- Timeout tính từ arrival_time
- Abort execution nếu timeout, release resources

**Test:** Config service có timeout ngắn (vd 0.5s) + job_size lớn (service_time > timeout) → verify request bị timeout, resources released, `timed_out=True`, `drop_reason="timeout"`.

---

## Step 7: Basic monitoring

**Mục tiêu:** Monitor thu thập metrics định kỳ.

**File code:**
- `monitoring/metric_store.py` — ring buffer, put/get_latest/query_range
- `monitoring/collectors.py` — RequestCollector, ClusterCollector (basic versions)
- `monitoring/monitor_manager.py` — SimPy periodic process
- `monitoring/monitor_api.py` — MonitorAPI facade

**Test:** Chạy simulation → verify metric_store có data, get_latest() trả về giá trị hợp lệ, ring buffer không vượt max_history_length.

---

## Step 8: Export (3 modes)

**Mục tiêu:** Xuất kết quả simulation ra file.

**File code:**
- `export/batch_csv_writer.py` — BatchCSVWriter với buffer + flush
- `export/summary_writer.py` — SummaryWriter ghi summary.txt
- `export/system_metrics_exporter.py` — hook vào monitor, ghi system_metrics.csv
- `export/request_trace_exporter.py` — ghi request_trace.csv
- `export/export_manager.py` — chọn mode 0/1/2, coordinate exporters

**Test:** Chạy simulation với mỗi export mode → verify file output đúng:
- Mode 0: chỉ summary.txt
- Mode 1: summary.txt + system_metrics.csv (có header, có data rows)
- Mode 2: + request_trace.csv (số rows = số completed requests)

---

## Step 9: SimulationEngine + CLI hoàn chỉnh

**Mục tiêu:** Chạy standalone simulation end-to-end từ CLI.

**File code:**
- `core/simulation/sim_builder.py` — build toàn bộ system từ config
- `core/simulation/sim_engine.py` — setup(), run(), shutdown()
- `runtime/cli.py` — `simulate` command hoàn chỉnh
- `runtime/app.py` — wire CLI → builder → engine

**Test:** `python -m serverless_sim simulate --sim-config configs/simulation/sample_minimal.json --log-mode both` → chạy full simulation, output trong `logs/`, có summary.txt, log file, optional CSVs.

---

## Step 10: Extended lifecycle states + transition resources

**Mục tiêu:** Hỗ trợ config-driven states (partial loading stages).

**File code:**
- Cập nhật state_machine.py — load states/transitions từ config, validate graph
- Cập nhật lifecycle_manager.py — transition resource allocate/release, transition time delay
- Cập nhật container_instance.py — track target_state, mid-transition = transient

**Config tạo:**
- `configs/simulation/sample_extended_states.json` — thêm states: code_loaded, deps_loaded giữa prewarm và warm

**Test:** Chạy simulation với extended states → verify:
- Instance đi qua đúng transition path (null → prewarm → code_loaded → warm → running)
- Transition resource allocated rồi released đúng
- Transition time reflected trong timestamps

---

## Step 11: Autoscaling (memory-bounded + LRU + idle timeout)

**Mục tiêu:** Autoscaler quản lý container pool theo OpenWhisk model.

**File code:**
- `autoscaling/autoscaler.py` — OpenWhiskPoolAutoscaler: periodic reconcile, prewarm top-up, LRU eviction, idle timeout
- `autoscaling/autoscaling_api.py` — get/set idle_timeout, get/set prewarm_count, trigger_reconcile

**Test:** Verify:
- Prewarm containers tự động tạo khi dưới target
- Idle container bị evict sau idle_timeout
- Khi node hết memory → evict LRU idle container để nhường chỗ
- Chỉ evict container ở stable state

---

## Step 12: Controller + ThresholdPolicy

**Mục tiêu:** Controller định kỳ đọc monitor và điều chỉnh autoscaling.

**File code:**
- `controller/base_controller.py` — BaseController với SimPy periodic loop
- `controller/policies/base_policy.py` — BaseControlPolicy interface
- `controller/policies/threshold_policy.py` — rule-based: CPU > 80% → tăng prewarm, etc.

**Test:** Chạy simulation với controller enabled → verify:
- Controller step chạy đúng interval
- Prewarm count / idle timeout thay đổi theo rules
- System phản ứng đúng với thay đổi (thêm prewarm containers, thay đổi eviction timing)

---

## Step 13: Richer monitoring

**Mục tiêu:** Bổ sung đầy đủ metrics theo design.

**File code:**
- Cập nhật collectors.py — LifecycleCollector, AutoscalingCollector
- Thêm metrics: instance counts per state, per service, transition counts, pool targets, latency percentiles (p50/p95/p99)

**Config tạo:**
- `configs/simulation/sample_multi_service.json` — 2-3 services, monitor full metrics

**Test:** Chạy simulation → verify tất cả metric families có data trong MetricStore. Export mode 1 CSV có đầy đủ columns.

---

## Step 14: Gymnasium wrapper

**Mục tiêu:** Gym env chạy được với random actions.

**File code:**
- `gym_env/observation_builder.py` — monitor snapshot → numpy vector
- `gym_env/action_mapper.py` — action index → autoscaling API calls
- `gym_env/reward_calculator.py` — weighted penalty sum
- `gym_env/serverless_gym_env.py` — ServerlessGymEnv: reset(), step(), close()

**Config tạo:**
- `configs/gym/sample_gym_discrete.json`

**Test:**
```python
import gymnasium as gym
env = ServerlessGymEnv("configs/simulation/sample_minimal.json", "configs/gym/sample_gym_discrete.json")
obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
env.close()
```
Verify: observation shape đúng, reward là float, không crash.

---

## Step 15: VecEnv compatibility

**Mục tiêu:** Gym env chạy được với DummyVecEnv và SubprocVecEnv.

**Test:**
```python
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# DummyVecEnv
env = DummyVecEnv([make_env(i) for i in range(4)])
env.reset()
env.step([0, 0, 0, 0])
# SubprocVecEnv
env = SubprocVecEnv([make_env(i) for i in range(4)])
env.reset()
env.step([0, 0, 0, 0])
```
Verify: không crash, observation shapes consistent, mỗi env có seed khác nhau.

---

## Step 16: PPO training

**Mục tiêu:** Train PPO model, save checkpoint.

**File code:**
- `rl_agent/train.py` — load configs, create VecEnv, PPO.learn(), save model
- `runtime/cli.py` — `train` command

**Config tạo:**
- `configs/rl/sample_ppo_train.json`

**Test:** `python -m serverless_sim train --sim-config ... --gym-config ... --rl-config ...` → model saved, training log output, no crash.

---

## Step 17: PPO inference

**Mục tiêu:** Load trained model, chạy inference qua Gym env.

**File code:**
- Cập nhật `rl_agent/train.py` hoặc thêm inference logic trong CLI
- `runtime/cli.py` — `infer` command

**Config tạo:**
- `configs/rl/sample_ppo_infer.json`

**Test:** `python -m serverless_sim infer --sim-config ... --gym-config ... --rl-config ...` → chạy episodes, output metrics/exports.

---

## Step 18: Sample configs + docs + polish

**Mục tiêu:** Hoàn thiện documentation và sample configs.

**File tạo:**
- `README.md`
- `docs/config_reference.md`
- `docs/simpy_standalone.md`
- `docs/gym_wrapper.md`
- `docs/rl_training.md`
- Đảm bảo tất cả sample configs trong `configs/` hợp lệ và chạy được

**Test:** Chạy tất cả sample configs qua CLI — không lỗi.
