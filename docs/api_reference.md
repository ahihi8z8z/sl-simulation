# API Reference

Tài liệu các API nội bộ mà Controller, RL agent, và Gym env sử dụng để đọc metrics và điều khiển autoscaling.

---

## MonitorAPI — Đọc metrics

**File:** `monitoring/monitor_api.py`

Facade read-only trên `MetricStore`. Được dùng bởi:
- `BaseController._step()` — đọc snapshot để ra quyết định
- `ServerlessGymEnv.step()` — build observation vector
- `SimulationEngine.get_snapshot()` — lấy metrics tại 1 thời điểm

### Methods

| Method | Signature | Mô tả |
|--------|-----------|-------|
| `get_latest` | `(metric_name: str) → (float, Any) \| None` | Trả về `(timestamp, value)` mới nhất của metric. `None` nếu chưa có data |
| `get_latest_value` | `(metric_name: str, default=0.0) → Any` | Chỉ trả value (không timestamp). Trả `default` nếu chưa có |
| `query_range` | `(metric_name: str, start: float, end: float) → list[(float, Any)]` | Tất cả entries có `start ≤ timestamp ≤ end` |
| `get_snapshot` | `(metric_names: list[str] \| None = None) → dict[str, Any]` | Dict `{metric: latest_value}`. Nếu `None` → lấy tất cả metrics |
| `get_all_metric_names` | `() → list[str]` | Danh sách tên tất cả metrics đã ghi |

### Ví dụ

```python
api = MonitorAPI(ctx.monitor_manager)

# Đọc 1 metric
cpu = api.get_latest_value("cluster.cpu_utilization", default=0.0)

# Snapshot toàn bộ
snap = api.get_snapshot()
print(snap["request.completed"], snap["cluster.memory_utilization"])

# Query theo thời gian
entries = api.query_range("request.completed", start=10.0, end=20.0)
for timestamp, value in entries:
    print(f"t={timestamp}: {value}")
```

---

## AutoscalingAPI — Điều khiển autoscaling

**File:** `autoscaling/autoscaling_api.py`

Facade để controller/RL điều chỉnh tham số autoscaler per-service. Được dùng bởi:
- `BaseController._apply_action()` — áp dụng quyết định từ policy
- `ActionMapper` (Gym) — map discrete action → API call

### Methods

| Method | Signature | Mô tả |
|--------|-----------|-------|
| `get_idle_timeout` | `(service_id: str) → float` | Thời gian idle trước khi bị evict (giây). Mặc định 60.0 |
| `set_idle_timeout` | `(service_id: str, value: float) → None` | Đặt idle timeout. Có hiệu lực từ reconcile tiếp theo |
| `get_prewarm_count` | `(service_id: str) → int` | Số container prewarm target. Mặc định 0 |
| `set_prewarm_count` | `(service_id: str, count: int) → None` | Đặt prewarm target. Có hiệu lực từ reconcile tiếp theo |
| `trigger_reconcile` | `() → None` | Trigger reconcile ngay lập tức (không đợi interval) |

### Ví dụ

```python
autoscaler = ctx.autoscaling_manager

# Đọc tham số hiện tại
print(autoscaler.get_prewarm_count("svc-api"))   # 2
print(autoscaler.get_idle_timeout("svc-api"))     # 30.0

# Điều chỉnh
autoscaler.set_prewarm_count("svc-api", 5)
autoscaler.set_idle_timeout("svc-api", 15.0)

# Áp dụng ngay (thay vì đợi reconcile_interval)
autoscaler.trigger_reconcile()
```

### Tác động của mỗi tham số

| Tham số | Tăng | Giảm |
|---------|------|------|
| `prewarm_count` | Nhiều container sẵn sàng hơn → ít cold start, nhiều memory | Ít container → nhiều cold start, tiết kiệm memory |
| `idle_timeout` | Giữ container lâu → ít cold start, tốn memory | Evict nhanh → nhiều cold start, tiết kiệm memory |

---

## Danh sách Metrics

Các metrics được thu thập bởi 4 collectors, mỗi `monitoring.interval` giây (mặc định 1s).

### RequestCollector

| Metric | Type | Mô tả |
|--------|------|-------|
| `request.total` | int | Tổng requests từng tạo |
| `request.completed` | int | Requests hoàn thành |
| `request.dropped` | int | Requests bị drop (no capacity) |
| `request.timed_out` | int | Requests bị timeout |
| `request.truncated` | int | Requests bị truncate (simulation end) |
| `request.cold_starts` | int | Requests completed có cold start |
| `request.in_flight` | int | Requests đang xử lý (trong `_active` dict) |
| `request.latency_mean` | float | Latency trung bình (giây) |
| `request.latency_p50` | float | Latency percentile 50 |
| `request.latency_p95` | float | Latency percentile 95 |
| `request.latency_p99` | float | Latency percentile 99 |

**Lưu ý:** Latency = `completion_time - arrival_time` (bao gồm queue + cold start + execution). Percentiles chỉ xuất hiện khi có ≥1 completed request.

### ClusterCollector

| Metric | Type | Mô tả |
|--------|------|-------|
| `cluster.nodes_enabled` | int | Số nodes đang enabled |
| `cluster.cpu_total` | float | Tổng CPU capacity (cores) |
| `cluster.cpu_used` | float | CPU đã allocate |
| `cluster.cpu_utilization` | float | `cpu_used / cpu_total` (0.0–1.0+) |
| `cluster.memory_total` | float | Tổng memory capacity (MB) |
| `cluster.memory_used` | float | Memory đã allocate |
| `cluster.memory_utilization` | float | `memory_used / memory_total` (0.0–1.0+) |

**Lưu ý:** Utilization có thể > 1.0 khi overcommit (xem Edge Cases).

### LifecycleCollector

| Metric | Type | Mô tả |
|--------|------|-------|
| `lifecycle.instances_total` | int | Tổng container không kể evicted |
| `lifecycle.instances_warm` | int | Container ở trạng thái warm |
| `lifecycle.instances_running` | int | Container đang chạy request |
| `lifecycle.instances_prewarm` | int | Container ở trạng thái prewarm |
| `lifecycle.instances_evicted` | int | Container đã bị evict |
| `lifecycle.{svc_id}.instances_total` | int | Container của service cụ thể |
| `lifecycle.{svc_id}.instances_running` | int | Container running của service cụ thể |

### AutoscalingCollector

| Metric | Type | Mô tả |
|--------|------|-------|
| `autoscaling.{svc_id}.prewarm_target` | int | Prewarm target hiện tại của service |
| `autoscaling.{svc_id}.idle_timeout` | float | Idle timeout hiện tại của service (giây) |

---

## Luồng điều khiển

```
MonitorManager (mỗi 1s)
  │
  └─► MetricStore
        │
        ├─► MonitorAPI.get_snapshot()
        │     │
        │     └─► BaseController._step() (mỗi 5s)
        │           │
        │           └─► ThresholdPolicy.decide(snapshot)
        │                 │
        │                 └─► actions: [{action, service_id, value}, ...]
        │                       │
        │                       └─► AutoscalingAPI
        │                             ├─ set_prewarm_count()
        │                             └─ set_idle_timeout()
        │                                   │
        │                                   └─► Autoscaler._reconcile_loop() (mỗi 5s)
        │                                         ├─ evict idle
        │                                         ├─ LRU evict
        │                                         └─ prewarm top-up
        │
        └─► ObservationBuilder.build(snapshot)  (Gym, mỗi step)
              │
              └─► RL Agent
                    │
                    └─► ActionMapper → AutoscalingAPI
```

### Timing

| Component | Interval | Mặc định | Cấu hình |
|-----------|----------|----------|----------|
| MonitorManager | `monitoring.interval` | 1.0s | `monitoring.interval` |
| Controller | `controller.interval` | 5.0s | `controller.interval` |
| Autoscaler | `autoscaling.reconcile_interval` | 5.0s | `autoscaling.reconcile_interval` |
| Gym step | `gym.step_duration` | 5.0s | `step_duration` |

**Lưu ý:** Controller đọc `get_snapshot()` — lấy **latest value** từ MetricStore, không trigger collect mới. Nên `controller.interval` nên ≥ `monitoring.interval` để có data mới.
