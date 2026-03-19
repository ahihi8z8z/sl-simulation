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

**Per-state pool targets:**

| Method | Signature | Mô tả |
|--------|-----------|-------|
| `get_pool_target` | `(service_id: str, state: str) → int` | Target số container chính xác tại state này |
| `set_pool_target` | `(service_id: str, state: str, count: int) → None` | Đặt target (chỉ cho state trung gian, không phải warm). Có hiệu lực từ reconcile tiếp theo |
| `get_all_pool_targets` | `(service_id: str) → dict[str, int]` | Tất cả pool targets của service |

**Idle timeout và backward-compatible:**

| Method | Signature | Mô tả |
|--------|-----------|-------|
| `get_idle_timeout` | `(service_id: str) → float` | Thời gian idle trước khi bị evict (giây). Mặc định 60.0 |
| `set_idle_timeout` | `(service_id: str, value: float) → None` | Đặt idle timeout |
| `get_prewarm_count` | `(service_id: str) → int` | Alias cho `get_pool_target(svc, first_chain_state)` |
| `set_prewarm_count` | `(service_id: str, count: int) → None` | Alias cho `set_pool_target(svc, first_chain_state, count)` |
| `trigger_reconcile` | `() → None` | Trigger reconcile ngay lập tức (không đợi interval) |

### Per-state pool targets

Pool targets kiểm soát số container stem-cell tại các state **trung gian** trong cold-start chain (giữa `null` và `warm`, exclusive). Mỗi pool đếm chính xác containers tại state đó — **không** đếm containers ở state sâu hơn.

Warm containers **không có pool target** — chúng được tạo tự nhiên bởi request processing và giữ alive bởi `idle_timeout`. Giống OpenWhisk, số warm containers chỉ bị giới hạn bởi memory node.

Ví dụ với chain `[null, prewarm, code_loaded, warm]`:

```
pool_target("svc-a", "prewarm")     = 3  → giữ chính xác 3 container tại prewarm
pool_target("svc-a", "code_loaded") = 2  → giữ chính xác 2 container tại code_loaded
# warm: KHÔNG có target — tạo bởi request, evict bởi idle_timeout
```

Nếu hiện có 1 warm + 1 code_loaded + 0 prewarm:
- warm target (1): ✓ đã có 1
- code_loaded target (2): có 1 warm + 1 code_loaded = 2 ✓
- prewarm target (3): có 1 warm + 1 code_loaded + 0 prewarm = 2 → tạo thêm 1 prewarm

### Ví dụ

```python
autoscaler = ctx.autoscaling_manager

# Đọc tham số hiện tại
print(autoscaler.get_pool_target("svc-api", "prewarm"))      # 2
print(autoscaler.get_pool_target("svc-api", "warm"))          # 0
print(autoscaler.get_all_pool_targets("svc-api"))             # {"prewarm": 2}
print(autoscaler.get_idle_timeout("svc-api"))                 # 30.0

# Điều chỉnh per-state targets
autoscaler.set_pool_target("svc-api", "prewarm", 5)
autoscaler.set_pool_target("svc-api", "code_loaded", 3)
autoscaler.set_pool_target("svc-api", "warm", 1)
autoscaler.set_idle_timeout("svc-api", 15.0)

# Backward-compatible alias (tác động lên state đầu tiên trong chain)
autoscaler.set_prewarm_count("svc-api", 5)  # == set_pool_target("svc-api", "prewarm", 5)

# Áp dụng ngay
autoscaler.trigger_reconcile()
```

### Tác động của mỗi tham số

| Tham số | Tăng | Giảm |
|---------|------|------|
| `pool_target(state)` | Nhiều container sẵn sàng tại state → ít cold start từ state trở lên, tốn memory | Ít container → nhiều cold start, tiết kiệm memory |
| `idle_timeout` | Giữ container lâu → ít cold start, tốn memory | Evict nhanh → nhiều cold start, tiết kiệm memory |

**Trade-off giữa các state:**
- Target `prewarm` cao: container nhanh hơn khi cold start (bỏ qua null→prewarm), rẻ memory
- Target `warm` cao: container sẵn sàng ngay, không cold start, tốn memory nhất
- Target trung gian (`code_loaded`): cân bằng — bỏ qua giai đoạn tốn thời gian nhất, tiết kiệm hơn warm

---

## Danh sách Metrics

Các metrics được thu thập bởi 4 collectors, mỗi `monitoring.interval` giây (mặc định 1s).

### RequestCollector

| Metric | Type | Mô tả |
|--------|------|-------|
| `request.total` | int | Tổng requests từng tạo |
| `request.completed` | int | Requests hoàn thành |
| `request.dropped` | int | Requests bị drop (no capacity) |
| `request.truncated` | int | Requests bị truncated khi simulation kết thúc |
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
| `autoscaling.{svc_id}.idle_timeout` | float | Idle timeout hiện tại của service (giây) |
| `autoscaling.{svc_id}.pool_target.{state}` | int | Pool target cho state cụ thể (chỉ xuất hiện nếu target > 0) |

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
