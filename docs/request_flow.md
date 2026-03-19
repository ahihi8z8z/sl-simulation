# Luồng đi của Request

Chi tiết luồng đi của một request từ khi sinh ra đến khi kết thúc.

## Tổng quan

```
Generator  →  LoadBalancer  →  Node Queue  →  Promote / Cold Start / Reuse  →  Execution  →  Completion
   |               |               |                    |                          |              |
arrival_time  dispatch_time  queue_enter_time     (transition time)          exec_start_time  completion_time
```

## Phase 1: Arrival (PoissonFixedSizeGenerator)

**File:** `workload/generators.py` — `_arrival_loop()`

1. Sample `interval` từ phân phối exponential với `mean = 1/arrival_rate`
2. `yield env.timeout(interval)` — chờ đến thời điểm arrival
3. Kiểm tra `stop_time`: nếu `env.now >= stop_time` thì ngừng sinh
4. Tạo `Invocation`:
   - `status = "arrived"`
   - `arrival_time = env.now`
   - `service_id`, `job_size` từ ServiceClass
5. Đăng ký vào `ctx.request_table.register(inv)` — thêm vào `_active` dict, tăng `counters.total`
6. Gọi `ctx.dispatcher.dispatch(invocation)`

## Phase 2: Dispatch (ShardingContainerPoolBalancer)

**File:** `scheduling/load_balancer.py` — `dispatch()`

1. Tìm primary node: `hash(service_id) % n_nodes` (cached per service_id)
2. Walk hash ring (primary → primary+1 → ... → primary+n-1):
   - Bỏ qua node nếu `node.queue_is_full` (queue depth >= `max_queue_depth`)
   - Kiểm tra `node.available.memory >= service.memory`
   - Nếu đủ:
     - Set `invocation.assigned_node_id = node.node_id`
     - Set `invocation.dispatch_time = env.now`
     - Set `invocation.queue_enter_time = env.now`
     - Set `invocation.status = "queued"`
     - `node.queue.put(invocation)` — đưa vào SimPy Store
     - Return True
3. Nếu không node nào phù hợp (hết memory hoặc queue đầy):
   - Set `invocation.dropped = True`
   - Set `invocation.status = "dropped"`
   - Set `invocation.drop_reason = "no_capacity"`
   - Set `invocation.completion_time = env.now`

**Lưu ý:** Memory check ở đây là check thô — chỉ xem node còn bao nhiêu memory available, không reserve trước. Nhiều request có thể pass check cùng lúc và gây overcommit. Autoscaler xử lý overcommit qua LRU eviction.

## Phase 3: Queue (Node._pull_loop)

**File:** `cluster/node.py` — `_pull_loop()`

1. `yield node.queue.get()` — block cho đến khi có request
2. Nhận `invocation` từ queue
3. Spawn SimPy process `_process_request(invocation)`
4. Quay lại bước 1 (không đợi request hiện tại hoàn thành)

**Lưu ý:** Queue là `simpy.Store` (unbounded nếu `max_queue_depth=0`). Nhiều request có thể nằm trong queue đồng thời.

## Phase 4: Chọn Instance

**File:** `cluster/node.py` — `_process_request()`

### 4a. Tìm warm instance

```python
instance = lifecycle_manager.find_reusable_instance(node, service_id)
```

`find_reusable_instance()` duyệt tất cả instances trên node, tìm instance thỏa mãn:
- `service_id` khớp
- `state == "warm"`
- `available_slots > 0` (còn slot concurrency)

Nếu tìm thấy → nhảy sang Phase 5 (không cold start).

### 4b. Promote instance từ pool

Nếu không có warm instance, thử tìm instance ở state trung gian:

```python
promotable = lifecycle_manager.find_promotable_instance(node, service_id)
if promotable is not None:
    instance = yield lifecycle_manager.promote_instance(node, promotable)
```

`find_promotable_instance()` tìm instance **sâu nhất** (gần warm nhất) trong cold-start chain:
- `service_id` khớp
- State là trung gian (prewarm, code_loaded, ...)
- `target_state is None` (đã hoàn thành transition, không đang mid-transition)
- `is_idle` (không có active requests)

**Promote** chạy phần còn lại của chain. Ví dụ: `code_loaded → warm` chỉ mất 0.2s thay vì full cold start 0.9s.

Sau khi promote, `notify_pool_change()` được gọi để **fill lại pool ngay lập tức**.

### 4c. Full cold start

Nếu không có warm instance và không có instance trung gian → tạo mới từ null:

```python
cold_start_proc = lifecycle_manager.prepare_instance_for_service(node, service_id)
instance = yield cold_start_proc
```

**Cold start process** (`lifecycle_manager._cold_start()`):

1. Tạo `ContainerInstance(max_concurrency, memory, cpu)`
2. Allocate memory trên node: `node.allocate(ResourceProfile(cpu=0, memory=service.memory))`
3. Lấy cold-start chain từ state machine (tuyến tính: `null → ... → warm`)
4. Đi qua từng transition:
   - Allocate transition resources (cpu, memory tạm thời)
   - `yield env.timeout(transition_time)`
   - Release transition resources
5. Set `instance.state = "warm"`

## Phase 5: Concurrency Slot

```python
req = instance.slots.request()
yield req
```

`instance.slots` là `simpy.Resource(capacity=max_concurrency)`. Nếu tất cả slots đang bận, request đợi ở đây.

## Phase 6: Execution

### 6a. Bắt đầu

```python
lifecycle_manager.start_execution(instance, invocation)
```

- Set `instance.state = "running"`
- `instance.active_requests += 1`
- Allocate per-request CPU: `node.allocate(ResourceProfile(cpu=service.cpu, memory=0))`
- Set `invocation.execution_start_time = env.now`
- Set `invocation.assigned_instance_id`

### 6b. Thời gian phục vụ

```python
service_time = serving_model.estimate_service_time(job_size, node)
# FixedRateModel: service_time = job_size * processing_factor (default factor=1.0)
```

### 6c. Chờ execution hoàn thành

```python
yield env.timeout(service_time)
```

Execution time cố định = `job_size * processing_factor`. Không có timeout — request luôn chạy đến khi xong.

## Phase 7: Hoàn thành

### 7a. Hoàn thành bình thường

```python
lifecycle_manager.finish_execution(instance, invocation)
instance.slots.release(req)
invocation.status = "completed"
ctx.request_table.finalize(invocation)
```

`finish_execution()`:
- `instance.active_requests -= 1`
- Release per-request CPU: `node.release(ResourceProfile(cpu=service.cpu, memory=0))`
- Set `invocation.execution_end_time = env.now`
- Set `invocation.completion_time = env.now`
- Mark `cold_start` flag trên invocation nếu là request đầu tiên của instance
- Nếu instance hết active requests → set `state = "warm"` (sẵn sàng nhận request tiếp)

`finalize()`:
- Cập nhật counters, ghi latency (bisect.insort), stream CSV row, xóa khỏi `_active`

### 7b. Dropped (no capacity / queue full)

Xảy ra ở Phase 2:
- Set `dropped = True`
- Set `status = "dropped"`, `drop_reason = "no_capacity"`
- Gọi `finalize(inv)`

### 7c. Truncated (simulation kết thúc)

Xảy ra sau drain period trong `SimulationEngine.shutdown()`:
- Set `status = "truncated"`, `drop_reason = "simulation_end"`
- Set `completion_time = env.now`
- Gọi `finalize(inv)`

Mọi trường hợp terminal đều kết thúc bằng `finalize()` — đảm bảo request được flush khỏi memory sau khi counters và trace đã ghi.

## Timeline minh họa

```
t=0.000  ARRIVE    req-1 service=svc-hello
t=0.000  DISPATCH  req-1 → node-0 (hash=0)
t=0.000  QUEUE     req-1 enters node-0 queue
t=0.000  COLD_START req-1 → null→prewarm (0.5s)
t=0.500  COLD_START req-1 → prewarm→warm (0.3s)
t=0.800  SLOT      req-1 acquires slot on inst-1
t=0.800  EXEC_START req-1 on inst-1 (service_time=0.1s)
t=0.900  EXEC_END  req-1 completed (latency=0.900s, cold_start=True)
t=0.900  inst-1 state: running → warm (idle, có thể tái sử dụng)

t=1.200  ARRIVE    req-2 service=svc-hello
t=1.200  DISPATCH  req-2 → node-0
t=1.200  REUSE     req-2 → inst-1 (warm, còn slots)
t=1.200  EXEC_START req-2 on inst-1 (service_time=0.1s)
t=1.300  EXEC_END  req-2 completed (latency=0.100s, cold_start=False)
```

**Timeline với promote (extended states):**

```
-- Pool có 1 code_loaded instance (inst-2) sẵn sàng --

t=5.000  ARRIVE    req-10 service=svc-hello
t=5.000  DISPATCH  req-10 → node-0
t=5.000  QUEUE     req-10 enters node-0 queue
t=5.000  PROMOTE   inst-2 code_loaded→warm (0.2s thay vì cold start 0.9s)
t=5.000  POOL_FILL tạo inst-3 mới từ null→code_loaded (reactive)
t=5.200  SLOT      req-10 acquires slot on inst-2
t=5.200  EXEC_START req-10 on inst-2 (service_time=0.1s)
t=5.300  EXEC_END  req-10 completed (latency=0.300s, cold_start=True)
```

## Sơ đồ trạng thái Invocation

```
                    +----------+
                    | created  |
                    +----+-----+
                         |
                    +----v-----+
                    | arrived  |
                    +----+-----+
                         |
              +----------+----------+
              |                     |
         +----v-----+         +----v-----+
         |  queued  |         | dropped  |  (no_capacity / queue_full)
         +----+-----+         +----------+
              |
         +----v------+
         | completed |
         +-----------+

                  +----------+
                  | truncated|  (simulation_end, sau drain)
                  +----------+
```
