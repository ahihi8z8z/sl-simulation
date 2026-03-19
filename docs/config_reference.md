# Tham chiếu cấu hình

## Simulation Config

File JSON chính điều khiển toàn bộ simulation. 3 section bắt buộc: `simulation`, `services`, `cluster`. Các section khác tùy chọn.

### simulation (bắt buộc)

```json
{
  "simulation": {
    "duration": 60.0,
    "seed": 42,
    "export_mode": 0,
    "drain_timeout": 30.0
  }
}
```

| Key | Type | Bắt buộc | Mặc định | Mô tả |
|-----|------|----------|----------|-------|
| `duration` | float | có | — | Thời gian sinh workload (giây). Simulation chạy thêm drain period sau duration |
| `seed` | int | có | — | Seed cho numpy RNG. Cùng seed → cùng kết quả |
| `export_mode` | int | không | 0 | 0: summary, 1: +metrics CSV, 2: +request trace CSV |
| `drain_timeout` | float | không | max(service.timeout) | Thời gian chờ in-flight requests hoàn thành sau duration |

### services (bắt buộc, non-empty list)

```json
{
  "services": [
    {
      "service_id": "svc-hello",
      "arrival_rate": 5.0,
      "job_size": 0.1,
      "memory": 256,
      "cpu": 1.0,
      "max_concurrency": 4,
      "prewarm_count": 1,
      "idle_timeout": 30.0,
      "pool_targets": {"prewarm": 2, "code_loaded": 1}
    }
  ]
}
```

| Key | Type | Bắt buộc | Mặc định | Mô tả |
|-----|------|----------|----------|-------|
| `service_id` | str | có | — | Định danh duy nhất của service |
| `arrival_rate` | float | có | — | Poisson arrival rate (requests/giây) |
| `job_size` | float | có | — | Kích thước công việc. `service_time = job_size * processing_factor` |
| `memory` | float | có | — | Memory per container (MB) |
| `cpu` | float | có | — | CPU per request (cores). Allocate khi execution, release khi done |
| `max_concurrency` | int | có | — | Số request đồng thời tối đa trên 1 container |
| `display_name` | str | không | service_id | Tên hiển thị |
| `arrival_mode` | str | không | "poisson" | Kiểu phân phối arrival (hiện chỉ có poisson) |
| `prewarm_count` | int | không | 0 | Alias — đặt pool target cho state đầu tiên trong chain |
| `idle_timeout` | float | không | 60.0 | Thời gian idle trước khi bị evict (giây) |
| `pool_targets` | dict | không | {} | Per-state pool targets, ví dụ `{"prewarm": 2, "code_loaded": 1}`. Chỉ áp dụng cho states trung gian (không bao gồm `warm`) |

### cluster (bắt buộc)

```json
{
  "cluster": {
    "nodes": [
      {
        "node_id": "node-0",
        "cpu_capacity": 8.0,
        "memory_capacity": 8192,
        "max_queue_depth": 100
      }
    ]
  }
}
```

| Key | Type | Bắt buộc | Mặc định | Mô tả |
|-----|------|----------|----------|-------|
| `nodes` | list | có | — | Danh sách nodes (phải có ít nhất 1) |

**Mỗi node:**

| Key | Type | Bắt buộc | Mặc định | Mô tả |
|-----|------|----------|----------|-------|
| `node_id` | str | có | — | Định danh duy nhất của node |
| `cpu_capacity` | float | có | — | Tổng CPU (cores) |
| `memory_capacity` | float | có | — | Tổng memory (MB) |
| `max_queue_depth` | int | không | 0 | Giới hạn request trong queue (0 = unlimited). Khi đầy, LoadBalancer walk sang node khác |

### autoscaling (tùy chọn)

```json
{
  "autoscaling": {
    "enabled": true,
    "reconcile_interval": 5.0
  }
}
```

| Key | Type | Mặc định | Mô tả |
|-----|------|----------|-------|
| `enabled` | bool | false | Bật/tắt autoscaler |
| `reconcile_interval` | float | 5.0 | Chu kỳ reconcile (giây) |

Khi `enabled: true`, autoscaler thực hiện 3 việc mỗi reconcile:
1. Evict containers idle quá `idle_timeout` của service
2. LRU evict khi node overcommit memory
3. Top-up pool targets — tạo containers đến target count cho mỗi state trung gian (không bao gồm `warm`). Warm containers tạo tự nhiên bởi request

### monitoring (tùy chọn)

```json
{
  "monitoring": {
    "interval": 1.0,
    "max_history_length": 1000
  }
}
```

| Key | Type | Mặc định | Mô tả |
|-----|------|----------|-------|
| `interval` | float | 1.0 | Chu kỳ thu thập metrics (giây) |
| `max_history_length` | int | 1000 | Số lượng data points tối đa mỗi metric (ring buffer) |

### controller (tùy chọn, yêu cầu autoscaling enabled)

```json
{
  "controller": {
    "enabled": true,
    "interval": 5.0,
    "cpu_high": 0.8,
    "cpu_low": 0.3,
    "prewarm_max": 10,
    "prewarm_min": 0,
    "prewarm_step": 1,
    "idle_timeout_min": 5.0,
    "idle_timeout_max": 120.0,
    "idle_timeout_step": 5.0
  }
}
```

| Key | Type | Mặc định | Mô tả |
|-----|------|----------|-------|
| `enabled` | bool | false | Bật/tắt controller. Chỉ hoạt động khi autoscaling cũng enabled |
| `interval` | float | 5.0 | Chu kỳ quyết định (giây) |
| `cpu_high` | float | 0.8 | Ngưỡng CPU cao → tăng prewarm, giảm idle_timeout |
| `cpu_low` | float | 0.3 | Ngưỡng CPU thấp → giảm prewarm, tăng idle_timeout |
| `prewarm_max` | int | 10 | Giới hạn trên prewarm_count |
| `prewarm_min` | int | 0 | Giới hạn dưới prewarm_count |
| `prewarm_step` | int | 1 | Bước tăng/giảm prewarm |
| `idle_timeout_min` | float | 5.0 | Giới hạn dưới idle_timeout (giây) |
| `idle_timeout_max` | float | 120.0 | Giới hạn trên idle_timeout (giây) |
| `idle_timeout_step` | float | 5.0 | Bước tăng/giảm idle_timeout |

### lifecycle (tùy chọn)

```json
{
  "lifecycle": {
    "cold_start_chain": ["null", "prewarm", "code_loaded", "warm"],
    "states": [
      {"name": "null", "category": "stable"},
      {"name": "prewarm", "category": "stable", "steady_memory": 0.0},
      {"name": "code_loaded", "category": "stable", "service_bound": true},
      {"name": "warm", "category": "stable", "service_bound": true, "reusable": true},
      {"name": "running", "category": "transient", "service_bound": true, "reusable": false},
      {"name": "evicted", "category": "stable", "reusable": false}
    ],
    "transitions": [
      {"from": "null", "to": "prewarm", "time": 0.3, "cpu": 0.1},
      {"from": "prewarm", "to": "code_loaded", "time": 0.4, "cpu": 0.2, "memory": 64},
      {"from": "code_loaded", "to": "warm", "time": 0.2, "cpu": 0.1},
      {"from": "warm", "to": "running", "time": 0.0},
      {"from": "running", "to": "warm", "time": 0.0},
      {"from": "warm", "to": "evicted", "time": 0.0},
      {"from": "prewarm", "to": "evicted", "time": 0.0},
      {"from": "code_loaded", "to": "evicted", "time": 0.0}
    ]
  }
}
```

**cold_start_chain:**

| Key | Type | Mặc định | Mô tả |
|-----|------|----------|-------|
| `cold_start_chain` | list[str] | tự suy | Danh sách tuyến tính các state từ `"null"` đến `"warm"`. Phải bắt đầu bằng `"null"` và kết thúc bằng `"warm"`. Transitions phải tồn tại giữa mọi cặp liền kề |

Nếu không khai báo `cold_start_chain`, hệ thống tự suy bằng cách đi theo forward transitions từ `null`. Nếu một state có nhiều hơn 1 forward transition (ngoài `running` và `evicted`) → báo lỗi yêu cầu khai báo tường minh.

**State:**

| Key | Type | Mặc định | Mô tả |
|-----|------|----------|-------|
| `name` | str | bắt buộc | Tên state. Phải có: null, warm, running, evicted |
| `category` | str | "stable" | "stable" hoặc "transient" |
| `steady_cpu` | float | 0.0 | CPU tiêu thụ khi ở state này |
| `steady_memory` | float | 0.0 | Memory tiêu thụ khi ở state này |
| `service_bound` | bool | false | True nếu state gắn với 1 service cụ thể |
| `reusable` | bool | true | True nếu instance ở state này có thể nhận request |

**Transition:**

| Key | Type | Mặc định | Mô tả |
|-----|------|----------|-------|
| `from` | str | bắt buộc | State nguồn |
| `to` | str | bắt buộc | State đích |
| `time` | float | 0.0 | Thời gian chuyển đổi (giây) |
| `cpu` | float | 0.0 | CPU tạm thời trong quá trình chuyển đổi |
| `memory` | float | 0.0 | Memory tạm thời trong quá trình chuyển đổi |

Nếu không có section `lifecycle`, hệ thống dùng default chain:
```
null --0.5s--> prewarm --0.3s--> warm
```

---

## Gym Config

```json
{
  "max_steps": 50,
  "prewarm_max": 10,
  "idle_timeout_max": 120.0,
  "observation_metrics": null,
  "reward": {
    "drop_penalty": -1.0,
    "cold_start_penalty": -0.1,
    "latency_penalty": -0.5,
    "resource_penalty": -0.1,
    "throughput_reward": 0.1
  }
}
```

| Key | Type | Mặc định | Mô tả |
|-----|------|----------|-------|
| `max_steps` | int | 100 | Số step tối đa mỗi episode. `step_duration` tự lấy từ `controller.interval` |
| `prewarm_max` | int | 10 | Giới hạn trên action tăng pool target |
| `idle_timeout_max` | float | 120.0 | Giới hạn trên action tăng idle_timeout |
| `observation_metrics` | list/null | null | Danh sách tên metric cho observation. null = dùng default 11 metrics |
| `reward.*` | float | xem trên | Hệ số reward cho từng thành phần |

---

## RL Config

### Training

```json
{
  "n_envs": 2,
  "total_timesteps": 10000,
  "use_subproc": false,
  "learning_rate": 0.0003,
  "n_steps": 128,
  "batch_size": 64,
  "n_epochs": 10,
  "gamma": 0.99,
  "model_name": "ppo_serverless"
}
```

| Key | Type | Mặc định | Mô tả |
|-----|------|----------|-------|
| `n_envs` | int | 4 | Số parallel environments (DummyVecEnv hoặc SubprocVecEnv) |
| `total_timesteps` | int | 10000 | Tổng số timesteps training |
| `use_subproc` | bool | false | Dùng SubprocVecEnv thay vì DummyVecEnv |
| `learning_rate` | float | 3e-4 | PPO learning rate |
| `n_steps` | int | 128 | Steps per PPO update |
| `batch_size` | int | 64 | Minibatch size |
| `n_epochs` | int | 10 | PPO epochs per update |
| `gamma` | float | 0.99 | Discount factor |
| `model_name` | str | "ppo_serverless" | Tên file model lưu |

### Inference

```json
{
  "model_path": "logs/run_xxx/ppo_serverless",
  "n_episodes": 5,
  "seed": 100
}
```

| Key | Type | Mặc định | Mô tả |
|-----|------|----------|-------|
| `model_path` | str | bắt buộc | Đường dẫn đến model đã train (không cần .zip) |
| `n_episodes` | int | 1 | Số episodes chạy inference |
| `seed` | int | 42 | Seed cho môi trường inference |

---

## Sample Configs

| File | Mô tả |
|------|-------|
| `configs/simulation/sample_minimal.json` | 1 service, 1 node, autoscaling on, 60s |
| `configs/simulation/sample_extended_states.json` | Custom lifecycle (code_loaded state), 30s |
| `configs/simulation/sample_multi_service.json` | 3 services, 2 nodes, controller on, 30s |
| `configs/gym/sample_gym_discrete.json` | Gym env với 5s step, max 50 steps |
| `configs/rl/sample_ppo_train.json` | PPO training 1000 timesteps, 2 envs |
| `configs/rl/sample_ppo_infer.json` | Inference 2 episodes |
