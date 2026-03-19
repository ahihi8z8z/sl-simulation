# Edge Cases

Tài liệu này mô tả các tình huống đặc biệt và cách hệ thống xử lý chúng.

## 1. Queue Depth Limit

### Cơ chế

Mỗi node có `max_queue_depth` (cấu hình trong ComputeClass, mặc định 0 = unlimited). Khi queue đầy, LoadBalancer bỏ qua node đó và walk sang node tiếp trong hash ring.

### Khi nào request bị drop?

Request bị drop khi **tất cả** nodes đều không phù hợp:
- Queue đầy (`queue_depth >= max_queue_depth`)
- Hoặc không đủ memory (`available.memory < service.memory`)

Drop xảy ra **tại dispatch time**, request không bao giờ vào queue.

### Không có execution timeout

Execution time cố định (`job_size * processing_factor`), không có timeout. Mọi request đã vào queue đều được xử lý đến hoàn thành — chỉ bị truncated nếu simulation kết thúc.

## 2. Cạn kiệt tài nguyên

### Hết memory khi dispatch

Khi tất cả nodes hết memory, `ShardingContainerPoolBalancer.dispatch()` walk hết hash ring mà không tìm được node phù hợp:

```python
for offset in range(n):
    idx = (primary_idx + offset) % n
    node = enabled[idx]
    if node.can_fit(ResourceProfile(cpu=0, memory=service.memory)):
        # dispatch thành công
        ...
# Hết ring → drop
self._drop(invocation, "no_capacity")
```

Request bị drop ngay, không vào queue.

### Memory overcommit

Có thể xảy ra khi nhiều request đồng thời pass memory check và cùng tạo container trên cùng node. `Node.allocate()` không có lock — trong SimPy điều này an toàn vì mỗi step là atomic, nhưng 2 request có thể được dispatch cùng thời điểm và cả 2 đều thấy "còn đủ memory".

**Xử lý:** Autoscaler trong giai đoạn LRU eviction kiểm tra `node.available.memory < 0` và evict idle containers cho đến khi memory cân bằng.

### CPU overcommit

CPU được allocate per-request khi `start_execution()` và release khi `finish_execution()`. Không có pre-check trước khi allocate CPU — node cho phép negative available CPU.

**Ảnh hưởng:** `cluster.cpu_utilization` có thể > 1.0. ThresholdPolicy sẽ phát hiện và tăng prewarm/giảm idle_timeout để giảm tải.

### Không thể evict — tất cả instances đang active

Khi node hết memory và tất cả instances đều đang chạy (active_requests > 0):
- LRU eviction loop thoát vì không có candidate
- Memory tạm thời overcommit
- Khi một số request hoàn thành → instance trở về "warm" → có thể bị evict trong reconcile tiếp theo

## 3. Drain Period

### Vấn đề

Khi `env.run(until=duration)` kết thúc, SimPy dừng mọi process ngay lập tức. Requests đang in-flight (trong queue, đang cold start, đang execution) bị "đóng băng" — không có trạng thái terminal, không release resources.

### Giải pháp: 2-phase run

```
t=0              t=duration              t=duration+drain
|--- sinh request ---|--- chỉ xử lý nốt ---|--- mark truncated ---|
```

1. **Phase 1 (0 → duration):** Generator sinh request bình thường
2. **Generator stop:** `PoissonFixedSizeGenerator` kiểm tra `env.now >= stop_time` và ngừng sinh
3. **Phase 2 (duration → duration+drain):** SimPy tiếp tục chạy, in-flight requests hoàn thành
4. **Truncation sweep:** Sau drain, `_mark_truncated()` đánh dấu requests còn lại là "truncated"

### Cấu hình drain_timeout

```json
{
  "simulation": {
    "duration": 60.0,
    "drain_timeout": 30.0
  }
}
```

- Nếu không set: mặc định = 30s
- Set `drain_timeout: 0` để tắt drain (truncate ngay)

### Requests bị truncated

Sau drain, `SimulationEngine._mark_truncated()` duyệt request_table:

```python
terminal_statuses = {"completed"}
for inv in request_table.values():
    if inv.status not in terminal_statuses and not inv.dropped:
        inv.status = "truncated"
        inv.drop_reason = "simulation_end"
        inv.completion_time = env.now
```

Truncated requests:
- Xuất hiện trong `summary.txt` (dòng "Truncated: N")
- Xuất hiện trong `request_trace.csv` với status="truncated"
- Được đếm bởi `RequestCollector` metric `request.truncated`

## 4. Consistent Hash Ring

### Service affinity

Cùng một `service_id` luôn hash về cùng primary node (dùng `hash()` built-in, cached per service_id). Điều này đảm bảo:
- Các request của cùng service có xu hướng đến cùng node
- Tăng khả năng reuse warm instance (giảm cold start)
- Khi node hết memory → fallback walk ring, mất affinity

### Node failure / disable

Khi `node.enabled = False`:
- Node không nằm trong `get_enabled_nodes()` kết quả
- Hash ring chỉ chứa enabled nodes
- Nếu tất cả nodes disabled → mọi request bị drop

### Không có re-balancing

Khi số lượng nodes thay đổi (enable/disable), hash ring thay đổi kích thước. Requests đang trong queue của node cũ vẫn được xử lý — chỉ requests mới bị ảnh hưởng bởi ring mới.

## 5. Concurrency

### max_concurrency > 1

Mỗi `ContainerInstance` có `simpy.Resource(capacity=max_concurrency)`. Nhiều request có thể chạy đồng thời trên 1 instance:

```python
req = instance.slots.request()  # đợi slot
yield req                       # có slot → chạy
# ... execution ...
instance.slots.release(req)     # trả slot
```

### Trạng thái instance khi nhiều request đồng thời

- Instance ở trạng thái "running" khi `active_requests > 0`
- Chỉ về "warm" khi `active_requests == 0` (tất cả requests hoàn thành)
- `available_slots = max_concurrency - slots.count`

### Per-request CPU

Mỗi request allocate `service.cpu` CPU riêng trên node khi bắt đầu execution. Với max_concurrency=4 và cpu=1.0, 1 instance có thể dùng đến 4.0 CPU của node.

## 6. Cold Start

### Khi nào xảy ra cold start?

Cold start xảy ra khi `find_reusable_instance()` không tìm thấy warm instance phù hợp. Các trường hợp:
- Request đầu tiên của 1 service (chưa có instance nào)
- Tất cả instances đang busy (running, hết slots)
- Tất cả instances đã bị evict

### Thời gian cold start

Là tổng transition time trên cold-start chain (tuyến tính, chỉ có 1 đường duy nhất từ "null" → "warm"):
- Default chain `["null", "prewarm", "warm"]`: 0.5s + 0.3s = **0.8s**
- Extended chain `["null", "prewarm", "code_loaded", "warm"]`: 0.3 + 0.4 + 0.2 = **0.9s**

### Tài nguyên cold start

Ngoài memory (allocate 1 lần khi tạo instance), mỗi transition có thể dùng thêm CPU/memory tạm thời:
- Allocate trước khi wait transition_time
- Release sau khi wait xong
- Ví dụ: transition prewarm→code_loaded dùng 0.2 CPU và 64 MB tạm thời

## 7. Autoscaler: Reconcile + Reactive Top-up

### Reconcile chỉ xử lý eviction (2 giai đoạn)

1. **Evict idle:** Chỉ áp dụng cho containers ở state `warm`. Containers ở intermediate states (prewarm, code_loaded, ...) **không bị idle evict** — giống OpenWhisk stem cells, chúng tồn tại cho đến khi bị consume hoặc LRU evict
2. **LRU evict:** Nếu vẫn overcommit → evict LRU idle → đảm bảo memory dương. Áp dụng cho **tất cả** idle containers kể cả pool (vì đây là tình huống khẩn cấp)

### Pool top-up là reactive

Pool **không** được fill trong reconcile. Thay vào đó, `notify_pool_change()` được gọi ngay khi:
- Instance bị **evict** (từ `evict_instance()`)
- Instance bị **promote** lên warm (từ `promote_instance()`)
- **Pool target thay đổi** (từ `set_pool_target()`)
- **Startup** (từ `initial_fill()` khi gọi `autoscaler.start()`)

Điều này loại bỏ delay tối đa = reconcile_interval khi pool cạn — fill xảy ra ngay lập tức.

### Promote edge case

Khi request promote instance từ `code_loaded` → `warm`:
1. Instance chuyển sang warm, phục vụ request
2. `notify_pool_change()` tạo instance mới từ null → code_loaded
3. Instance mới mất 0.7s để sẵn sàng — trong thời gian đó pool tạm thiếu 1
4. Nếu burst request đến trong khoảng này → full cold start cho các request tiếp theo

### reconcile_interval

Mặc định 5.0s. Chỉ ảnh hưởng đến tốc độ phát hiện idle containers để evict. Không ảnh hưởng đến pool fill (đã reactive).

## 8. RequestStore và Memory

### Tại sao không dùng plain dict?

Với arrival_rate cao và duration dài (ví dụ 100 req/s × 3600s = 360,000 requests), plain dict giữ toàn bộ Invocation trong memory suốt simulation. `RequestStore` giải quyết bằng cách:

- **Chỉ giữ in-flight:** Completed/dropped requests bị xóa khỏi `_active` dict sau `finalize()`
- **Running counters:** `counters.total`, `counters.completed`, ... cập nhật O(1) trên mỗi finalize, không cần scan
- **Sorted latencies:** `bisect.insort()` giữ list luôn sorted — collectors đọc trực tiếp không cần sort lại
- **Streaming trace:** Mode 2 ghi CSV row ngay khi finalize, không cần giữ data để dump cuối

### Latency list vẫn tăng

`store.latencies` là list tất cả latencies của completed requests — vẫn tăng theo số completed. Nhưng mỗi entry chỉ là 1 float (8 bytes) thay vì toàn bộ Invocation (~500 bytes). Với 360,000 requests → ~2.8 MB thay vì ~180 MB.

### `len(store)` vs `store.active_count`

- `len(store)` trả về `counters.total` — tổng requests từng tạo (bao gồm đã flush)
- `store.active_count` trả về `len(_active)` — chỉ in-flight requests hiện tại
- `store.values()` chỉ iterate in-flight requests

## 9. Monitoring Edge Cases

### MetricStore ring buffer

`MetricStore` dùng `collections.deque(maxlen=max_history_length)` cho mỗi metric. Khi vượt quá max:
- Dữ liệu cũ nhất tự động bị xóa
- `get_latest()` vẫn đúng
- `query_range()` có thể mất dữ liệu cũ

### Metric chưa có data

- `get_latest("nonexistent")` trả về `None`
- `query_range("nonexistent", 0, 100)` trả về `[]`
- `ObservationBuilder` dùng `snapshot.get(name, 0.0)` — metric chưa có → 0.0

### Latency percentiles với ít requests

Khi chỉ có 1-2 completed requests:
- p50, p95, p99 đều bằng chính giá trị đó (index truncation)
- Không phải vấn đề — chỉ là do số lượng nhỏ

### Latency performance

`RequestCollector` trước đây sort toàn bộ latency list mỗi lần collect (O(n log n) mỗi giây, n tăng dần). Giờ `RequestStore` dùng `bisect.insort()` để giữ list luôn sorted — collector chỉ cần đọc trực tiếp O(1). Mean latency dùng `_latency_sum` tích lũy thay vì `sum()` mỗi lần.

## 10. Gym Environment Edge Cases

### reset() tạo simulation mới hoàn toàn

Mỗi lần gọi `reset()`, `ServerlessGymEnv` tạo lại toàn bộ simulation từ đầu — mới SimPy env, mới RNG, mới components. Không có state leakage giữa episodes.

### step() với action không hợp lệ

`ActionMapper` tính `svc_idx = action // 5`. Nếu `svc_idx >= len(service_ids)` thì action là no-op (không làm gì). Không raise exception.

### Observation chứa NaN

Nếu metric có giá trị infinity hoặc NaN (ví dụ chia cho 0), observation vector có thể chứa NaN → PPO có thể gặp vấn đề. Hiện tại `ClusterCollector` trả về `0.0` khi `total_cpu == 0` để tránh trường hợp này.

## 11. Configuration Edge Cases

### Services list rỗng

`load_config()` raise `ValueError` nếu `services` là list rỗng hoặc không phải list.

### Node list rỗng

Tương tự — `ValueError` nếu `cluster.nodes` rỗng.

### drain_timeout = 0

Hợp lệ — simulation kết thúc ngay tại `duration`, tất cả in-flight requests bị mark truncated. Hữu ích khi muốn biết chính xác bao nhiêu request bị ảnh hưởng.

### Không có autoscaling section

Mặc định autoscaling tắt (`ctx.autoscaling_manager = None`). Khi đó:
- Không có prewarm top-up
- Không có idle eviction
- Containers tồn tại vĩnh viễn (chỉ bị evict khi code gọi trực tiếp)

### Không có lifecycle section

Mặc định dùng `OpenWhiskExtendedStateMachine.default()` — 5 states cơ bản (null, prewarm, warm, running, evicted).
