# Tổng quan thiết kế

## Tổng quan

`serverless-sim` mô phỏng một nền tảng serverless dựa trên mô hình Apache OpenWhisk. Toàn bộ simulation chạy trên SimPy — một framework discrete-event simulation bằng Python — cho phép mô phỏng hàng ngàn request đồng thời mà không cần multi-threading.

## Kiến trúc module

```
+------------------+     +------------------+     +------------------+
|    Workload      |     |    Scheduling    |     |    Cluster       |
|  - ServiceClass  |     |  - LoadBalancer  |     |  - Node          |
|  - Invocation    |     |    (consistent   |     |  - ResourceProfile|
|  - Generator     |     |     hash ring)   |     |  - ClusterManager|
|  - WorkloadMgr   |     +--------+---------+     +--------+---------+
+--------+---------+              |                         |
         |                        v                         |
         +----------> dispatch(inv) ---------> node.queue --+
                                                            |
                                                            v
+------------------+     +------------------+     +--------+---------+
|   Autoscaling    |     |   Controller     |     |    Lifecycle     |
|  - Autoscaler    |     |  - BaseController|     |  - StateMachine  |
|  - AutoscalingAPI|     |  - ThresholdPolicy     |  - ContainerInst |
+--------+---------+     +--------+---------+     |  - LifecycleMgr  |
         ^                        |               +------------------+
         |                        v
         +--- set_pool_target/idle/min/max ---+
                                  |
+------------------+              |
|   Monitoring     |<--- read ----+
|  - MetricStore   |
|  - Collectors    |
|  - MonitorManager|
+--------+---------+
         |
         v
+------------------+
|     Export       |
|  - Summary       |
|  - SystemMetrics |
|  - RequestTrace  |
+------------------+
```

## Mô tả từng module

### core/ — Hạ tầng lõi

| File | Chức năng |
|------|-----------|
| `config/loader.py` | Load và validate JSON config. Kiểm tra required keys cho simulation, services, cluster |
| `logging/logger_factory.py` | Tạo logger với 3 mode: console, file, both. File log ghi vào `run_dir/simulation.log` |
| `simulation/sim_context.py` | `SimContext` dataclass — trung tâm lưu trữ tất cả reference: env, config, rng, logger, và mọi module manager |
| `simulation/sim_builder.py` | `SimulationBuilder.build()` — tạo và wire tất cả component từ config. Trả về SimContext đã sẵn sàng |
| `simulation/sim_engine.py` | `SimulationEngine` — điều khiển `setup()` → `run()` → `shutdown()`. Xử lý drain period và mark truncated requests |

### cluster/ — Mô hình cụm máy

| File | Chức năng |
|------|-----------|
| `resource_profile.py` | `ResourceProfile(cpu, memory)` — arithmetic: `add()`, `subtract()`, `can_fit()` |
| `compute_class.py` | `ComputeClass` dataclass — định danh loại máy (hiện tại chỉ có `class_id`) |
| `serving_model.py` | `FixedRateModel` — tính `service_time = job_size * processing_factor` |
| `node.py` | `Node` — có SimPy Store queue, resource accounting (allocate/release), và pull loop xử lý request |
| `cluster_manager.py` | `ClusterManager` — tạo nodes từ config, cung cấp `get_node()`, `get_enabled_nodes()`, `start_all()` |

**Cách resource accounting hoạt động:**

Node duy trì 2 ResourceProfile: `capacity` (cố định) và `available` (động). Khi allocate:
- `allocated += request`, `available -= request`
- Trả về `True` nếu đủ, `False` nếu không đủ (state không thay đổi khi False)
- Khi release: ngược lại

### workload/ — Sinh tải

| File | Chức năng |
|------|-----------|
| `service_class.py` | `ServiceClass` dataclass — định nghĩa service: arrival_rate, job_size, memory, cpu, max_concurrency, min_instances, max_instances |
| `invocation.py` | `Invocation` dataclass — đại diện 1 request với toàn bộ timestamps và trạng thái |
| `request_store.py` | `RequestStore` — lưu trữ request hiệu quả: chỉ giữ in-flight trong memory, duy trì running counters, sorted latency list (bisect.insort), và streaming trace CSV |
| `generators.py` | `PoissonFixedSizeGenerator` — SimPy process sinh request theo Poisson. Hỗ trợ `stop_time` để ngừng sinh khi hết duration |
| `workload_manager.py` | `WorkloadManager` — quản lý services, khởi động generators |

**Invocation status flow:**
```
"created" → "arrived" → "queued" → "completed"
                                 → "dropped"
                                 → "truncated" (khi simulation kết thúc)
```

### scheduling/ — Cân bằng tải

| File | Chức năng |
|------|-----------|
| `load_balancer.py` | `ShardingContainerPoolBalancer` — consistent hash ring. Hash `service_id` (built-in `hash()`, cached) để chọn primary node, walk ring tìm node có đủ memory |

**Thuật toán dispatch:**
1. Hash `hash(service_id)` → primary node index (cached per service_id)
2. Thử primary node: kiểm tra `node.queue_is_full` và `node.available.memory >= service.memory`
3. Nếu queue đầy hoặc không đủ memory: walk vòng ring (offset 1, 2, ..., n-1)
4. Nếu không node nào phù hợp: DROP request với `drop_reason="no_capacity"`

### lifecycle/ — Vòng đời container

| File | Chức năng |
|------|-----------|
| `state_definition.py` | `StateDefinition` — tên state, category (stable/transient), steady resource |
| `transition_definition.py` | `TransitionDefinition` — chuyển trạng thái: time, cpu, memory tạm thời |
| `container_instance.py` | `ContainerInstance` — simpy.Resource cho concurrency slots, theo dõi state và active requests |
| `state_machine.py` | `OpenWhiskExtendedStateMachine` — cold-start chain tuyến tính. Hỗ trợ default và config-driven |
| `lifecycle_manager.py` | `LifecycleManager` — find/reuse instance, promote từ pool, cold start, start/finish execution, eviction |

**Cold-start chain tuyến tính:**

State machine đảm bảo chỉ có **một đường duy nhất** từ `null` → `warm`. Đường này gọi là cold-start chain — một danh sách tuyến tính các trạng thái, mỗi trạng thái chỉ có một transition tiến về phía trước.

Default chain:
```
null --0.5s--> prewarm --0.3s--> warm
```

Extended chain (config-driven):
```
null --0.3s--> prewarm --0.4s--> code_loaded --0.2s--> warm
```

Ngoài chain còn có:
- Chu trình thực thi: `warm ↔ running`
- Eviction: bất kỳ stable state nào → `evicted`

Config khai báo chain tường minh qua `cold_start_chain`, hoặc hệ thống tự suy nếu mỗi state chỉ có 1 forward transition. Nếu đồ thị có nhánh mà không khai báo chain → báo lỗi rõ ràng.

### autoscaling/ — Tự động scale

| File | Chức năng |
|------|-----------|
| `autoscaler.py` | `OpenWhiskPoolAutoscaler` — reconcile chỉ xử lý eviction (idle timeout + LRU, tôn trọng min_instances). Pool top-up là **reactive**: trigger ngay khi instance bị evict, bị promote, warm→running, hoặc khi target thay đổi. `initial_fill()` fill warm đến min_instances và pool targets khi startup. Budget logic: fill warm đến min_instances trước (ưu tiên), sau đó fill pool_targets với budget còn lại (max_instances - total) |
| `autoscaling_api.py` | `AutoscalingAPI` — facade để controller/RL điều chỉnh pool_targets (per-state), idle_timeout, min_instances, max_instances. Warm containers không có pool target — tạo tự nhiên bởi request, đảm bảo tối thiểu bởi min_instances, giới hạn tổng bởi max_instances |

### controller/ — Vòng điều khiển

| File | Chức năng |
|------|-----------|
| `base_controller.py` | `BaseController` — SimPy periodic loop, đọc metrics và gọi policy.decide() |
| `policies/threshold_policy.py` | `ThresholdPolicy` — rule-based: CPU cao → tăng pool_target (state đầu tiên trong chain), giảm idle_timeout; CPU thấp → ngược lại. Controller/policy quyết định pool_targets và idle_timeout tại runtime dựa trên metrics |

### monitoring/ — Quan sát

| File | Chức năng |
|------|-----------|
| `metric_store.py` | `MetricStore` — ring buffer cho mỗi metric, hỗ trợ `put()`, `get_latest()`, `query_range()` |
| `collectors.py` | 4 collector: `RequestCollector`, `ClusterCollector`, `LifecycleCollector`, `AutoscalingCollector` |
| `monitor_manager.py` | `MonitorManager` — chạy collectors định kỳ, ghi vào MetricStore |
| `monitor_api.py` | `MonitorAPI` — facade để lấy snapshot, latest value |

### export/ — Xuất kết quả

| File | Chức năng |
|------|-----------|
| `batch_csv_writer.py` | `BatchCSVWriter` — buffer + flush CSV |
| `summary_writer.py` | `SummaryWriter` — ghi `summary.txt` với thời gian mô phỏng, wall-clock time, thống kê request và latency |
| `system_metrics_exporter.py` | `SystemMetricsExporter` — ghi `system_metrics.csv` từ MetricStore |
| `request_trace_exporter.py` | `RequestTraceExporter` — ghi `request_trace.csv` với mỗi request 1 dòng |
| `export_manager.py` | `ExportManager` — chọn mode 0/1/2, điều phối exporters |

**Export modes:**
- **Mode 0:** Chỉ `summary.txt`
- **Mode 1:** `summary.txt` + `system_metrics.csv`
- **Mode 2:** Tất cả + `request_trace.csv`

### gym_env/ — Gymnasium wrapper

| File | Chức năng |
|------|-----------|
| `serverless_gym_env.py` | `ServerlessGymEnv(gym.Env)` — wrap simulation thành Gym env với `reset()`/`step()` |
| `observation_builder.py` | `ObservationBuilder` — snapshot metrics → numpy vector |
| `action_mapper.py` | `ActionMapper` — action index → autoscaling API calls (5 actions/service) |
| `reward_calculator.py` | `RewardCalculator` — tính reward từ delta metrics (throughput bonus, drop/timeout/cold_start/latency/resource penalties) |

### rl_agent/ — RL training

| File | Chức năng |
|------|-----------|
| `train.py` | `run_training()` — tạo VecEnv, train PPO, save model |
| `infer.py` | `run_inference()` — load model, chạy episodes, trả về summary statistics |

## Nguyên tắc thiết kế

1. **SimPy-driven:** Mọi hoạt động bất đồng bộ (request arrival, cold start, execution, timeout, autoscaler reconcile, monitoring) đều là SimPy process. Không có real-time waiting.

2. **RequestStore:** `ctx.request_table` là `RequestStore` — chỉ giữ in-flight requests trong memory. Khi request hoàn thành, `finalize()` cập nhật running counters, ghi latency (sorted via `bisect.insort`), stream row ra CSV (nếu mode 2), rồi xóa khỏi dict. Collectors đọc counters trực tiếp thay vì scan toàn bộ table.

3. **Config-driven:** State machine, autoscaling parameters, controller thresholds đều đọc từ JSON config. Có thể thay đổi behavior mà không sửa code.

4. **Pluggable policy:** Controller nhận bất kỳ `BaseControlPolicy` nào. Hiện có `ThresholdPolicy`, có thể thay bằng RL policy hoặc custom logic.

5. **Drain period:** Khi hết duration, workload ngừng sinh nhưng simulation chạy thêm để in-flight requests hoàn thành. Requests còn lại sau drain bị mark "truncated".
