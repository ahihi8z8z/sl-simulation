# Gymnasium Wrapper

## Tổng quan

`ServerlessGymEnv` wrap toàn bộ simulation thành một Gymnasium environment chuẩn, cho phép RL agent tương tác qua `reset()` và `step()`.

Mỗi `step()` gọi:
1. Áp dụng action (điều chỉnh autoscaling parameters)
2. Chạy simulation thêm `step_duration` giây
3. Thu thập metrics
4. Tính reward
5. Trả về observation, reward, terminated, truncated, info

## Observation Space

**Mặc định:** `Box(shape=(10,), dtype=float32)` với 10 metrics:

| Index | Metric | Mô tả |
|-------|--------|-------|
| 0 | `cluster.cpu_utilization` | Tỷ lệ CPU đang dùng / tổng CPU |
| 1 | `cluster.memory_utilization` | Tỷ lệ memory đang dùng / tổng memory |
| 2 | `request.completed` | Tổng số request hoàn thành |
| 3 | `request.dropped` | Tổng số request bị drop |
| 4 | `request.in_flight` | Số request đang xử lý |
| 5 | `request.cold_starts` | Tổng số cold starts |
| 6 | `lifecycle.instances_total` | Tổng container (không kể evicted) |
| 7 | `lifecycle.instances_warm` | Container đang warm (idle, sẵn sàng) |
| 8 | `lifecycle.instances_running` | Container đang chạy request |
| 10 | `lifecycle.instances_prewarm` | Container đang prewarm |

Có thể tùy chỉnh bằng `observation_metrics` trong gym config:
```json
{
  "observation_metrics": [
    "cluster.cpu_utilization",
    "request.completed",
    "request.dropped"
  ]
}
```

## Action Space

**`Discrete(n_services * 5)`** — mỗi service có 5 actions:

| Local Action | Mô tả |
|-------------|-------|
| 0 | No-op (không làm gì) |
| 1 | Tăng pool target (state đầu tiên trong chain) thêm 1 (tối đa `prewarm_max`) |
| 2 | Giảm pool target (state đầu tiên trong chain) đi 1 (tối thiểu 0) |
| 3 | Tăng `idle_timeout` thêm `idle_timeout_step` (tối đa `idle_timeout_max`) |
| 4 | Giảm `idle_timeout` đi `idle_timeout_step` (tối thiểu `idle_timeout_min`) |

**Ví dụ:** Với 2 services → 10 actions:
- Action 0-4: điều khiển service đầu tiên
- Action 5-9: điều khiển service thứ hai

## Reward

Reward tính từ **delta** giữa 2 steps (không phải giá trị tuyệt đối):

```
reward = throughput_reward * d_completed
       + drop_penalty * d_dropped
       + cold_start_penalty * d_cold_starts
       + latency_penalty * current_latency
       + resource_penalty * current_cpu_util
```

| Thành phần | Mặc định | Mô tả |
|-----------|----------|-------|
| `throughput_reward` | +0.1 | Thưởng cho mỗi request hoàn thành trong step |
| `drop_penalty` | -1.0 | Phạt cho mỗi request bị drop trong step |
| `cold_start_penalty` | -0.1 | Phạt cho mỗi cold start trong step |
| `latency_penalty` | -0.5 | Nhân với mean latency hiện tại |
| `resource_penalty` | -0.1 | Nhân với CPU utilization hiện tại |

**Lưu ý:** `d_completed` là số request mới hoàn thành trong step này (không phải tổng). Tương tự cho d_dropped, d_cold_starts.

## Episode Flow

```python
env = ServerlessGymEnv("sim_config.json", "gym_config.json")
obs, info = env.reset()

for step in range(max_steps):
    action = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### reset()

1. Tạo lại toàn bộ simulation (mới env, rng, components)
2. Gọi `SimulationEngine.setup()` — bắt đầu tất cả processes
3. Reset reward calculator (xóa delta counters)
4. Thu thập initial snapshot
5. Trả về observation vector và info dict

### step(action)

1. Decode action → xác định service và hành động
2. Gọi `AutoscalingAPI` để thay đổi pool target hoặc idle_timeout
3. `env.run(until=env.now + step_duration)` — chạy thêm N giây simulation (`step_duration` = `controller.interval` từ config)
4. `monitor_manager.collect_once()` — thu thập metrics
5. Build observation từ snapshot
6. Tính reward từ delta metrics
7. `truncated = (current_step >= max_steps)`
8. `terminated = False` (không có điều kiện dừng sớm)

### close()

Hiện tại no-op. Simulation tự động giải phóng khi GC.

## VecEnv Compatibility

`ServerlessGymEnv` tương thích với Stable-Baselines3 VecEnv:

```python
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from serverless_sim.rl_agent.train import make_env

# DummyVecEnv — chạy trong cùng process
env = DummyVecEnv([make_env(sim_cfg, gym_cfg, seed=i) for i in range(4)])

# SubprocVecEnv — chạy trong subprocess riêng
env = SubprocVecEnv([make_env(sim_cfg, gym_cfg, seed=i) for i in range(4)])
```

`make_env()` trả về callable (factory function), mỗi callable tạo 1 env với seed riêng. Đảm bảo:
- Mỗi env có RNG riêng → workload patterns khác nhau
- Observation shapes nhất quán giữa các envs
- Không có shared state giữa envs

## Tùy chỉnh reward

Để thay đổi chiến lược reward, chỉnh trong gym config:

```json
{
  "reward": {
    "drop_penalty": -2.0,
    "cold_start_penalty": 0.0,
    "latency_penalty": -1.0,
    "resource_penalty": 0.0,
    "throughput_reward": 0.5
  }
}
```

- Set penalty = 0 để bỏ qua thành phần đó
- Tăng |penalty| để agent chú ý hơn đến yếu tố đó
- Điều chỉnh cân bằng giữa throughput và resource usage
