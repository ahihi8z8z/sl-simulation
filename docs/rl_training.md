# RL Training & Inference

## Tổng quan

Hệ thống hỗ trợ train PPO agent (Stable-Baselines3) để học chính sách autoscaling. Agent quan sát trạng thái cluster và điều chỉnh `prewarm_count` + `idle_timeout` của từng service.

## Training Pipeline

```
                +------------+
                | RL Config  |
                +-----+------+
                      |
                      v
+----------+   +------+------+   +-----------+
| Sim Cfg  +-->| make_env()  +-->| VecEnv    |
| Gym Cfg  |   | x n_envs    |   | (Dummy/   |
+----------+   +-------------+   |  Subproc) |
                                  +-----+-----+
                                        |
                                  +-----v-----+
                                  | PPO.learn()|
                                  +-----+-----+
                                        |
                                  +-----v-----+
                                  | model.save|
                                  +-----------+
```

### Chạy training

```bash
python -m serverless_sim train \
    --sim-config configs/simulation/sample_minimal.json \
    --gym-config configs/gym/sample_gym_discrete.json \
    --rl-config configs/rl/sample_ppo_train.json \
    --run-name my_experiment
```

Output:
- `logs/run_<timestamp>_train/ppo_serverless.zip` — model đã train
- Console log từ Stable-Baselines3 (reward, timesteps, ...)

### RL Config giải thích

```json
{
  "n_envs": 4,
  "total_timesteps": 100000,
  "use_subproc": false,
  "learning_rate": 0.0003,
  "n_steps": 128,
  "batch_size": 64,
  "n_epochs": 10,
  "gamma": 0.99,
  "model_name": "ppo_serverless"
}
```

**Các tham số quan trọng:**

| Tham số | Gợi ý | Mô tả |
|---------|-------|-------|
| `n_envs` | 4-8 | Nhiều env → dữ liệu đa dạng hơn, nhưng chậm hơn. DummyVecEnv sequential, SubprocVecEnv parallel |
| `total_timesteps` | 50000-500000 | Tổng bước training. Tăng để agent học tốt hơn, nhưng tốn thời gian |
| `n_steps` | 64-256 | Số bước mỗi update. Lớn hơn → estimate variance thấp nhưng chậm update |
| `batch_size` | 32-128 | Phải chia hết n_envs * n_steps |
| `gamma` | 0.95-0.999 | Cao → agent quan tâm đến tương lai xa |
| `use_subproc` | false | True cho parallel envs (nhanh hơn nhưng dùng nhiều memory) |

### DummyVecEnv vs SubprocVecEnv

| | DummyVecEnv | SubprocVecEnv |
|---|------------|---------------|
| Thực thi | Sequential | Parallel (multiprocessing) |
| Tốc độ | Chậm hơn với nhiều envs | Nhanh hơn với nhiều envs |
| Memory | Ít | Nhiều (mỗi env 1 process) |
| Debug | Dễ | Khó (subprocess) |
| Khi nào dùng | n_envs <= 4, debug | n_envs > 4, production |

## Inference Pipeline

### Chạy inference

```bash
python -m serverless_sim infer \
    --sim-config configs/simulation/sample_minimal.json \
    --gym-config configs/gym/sample_gym_discrete.json \
    --rl-config configs/rl/sample_ppo_infer.json
```

### RL Inference Config

```json
{
  "model_path": "logs/run_20260319_train/ppo_serverless",
  "n_episodes": 10,
  "seed": 100
}
```

| Key | Mô tả |
|-----|-------|
| `model_path` | Đường dẫn model (không cần .zip — SB3 tự thêm) |
| `n_episodes` | Số episodes chạy. Mỗi episode = 1 simulation hoàn chỉnh |
| `seed` | Seed base. Episode i dùng seed = seed + i |

### Output

Console in reward và steps mỗi episode:
```
Episode 1/10: reward=-12.34, steps=50
Episode 2/10: reward=-8.56, steps=50
...
Mean reward: -10.45
```

`run_inference()` trả về dict:
```python
{
    "n_episodes": 10,
    "mean_reward": -10.45,
    "total_steps": 500,
    "rewards": [-12.34, -8.56, ...]
}
```

## Tips

### Chọn simulation config cho training

- **Duration ngắn** (10-30s): Episode nhanh, nhiều update/phút, nhưng agent thấy ít patterns
- **Duration dài** (60-300s): Phong phú hơn nhưng chậm
- **Arrival rate vừa phải**: Quá thấp → ít quyết định. Quá cao → node liên tục quá tải
- **Nhiều services**: Tăng action space (5 actions/service) → cần nhiều timesteps hơn

### Điều chỉnh reward

- Bắt đầu với default penalties
- Nếu agent học được nhưng không giảm drops → tăng |drop_penalty|
- Nếu agent không bao giờ tăng prewarm → throughput_reward quá thấp
- Nếu agent giữ quá nhiều resources → tăng |resource_penalty|
- Theo dõi reward curve trong training log

### So sánh với ThresholdPolicy

Chạy inference với trained model và so sánh kết quả (summary.txt) với simulation dùng ThresholdPolicy (controller enabled). Các chỉ số để so sánh:
- Drop rate
- Cold start rate
- Mean latency
- CPU utilization

### Reproducibility

- Cùng `seed` trong sim config → cùng workload pattern
- Cùng `seed` trong rl infer config → cùng episodes
- Tuy nhiên, PPO training có randomness riêng (weight initialization, sampling). Để reproduce training, set seed trong PyTorch/numpy trước khi train
