# TODO

## Bugs / Logic issues

- [ ] `find_reusable_instance()` chỉ tìm instance ở state `warm`, bỏ qua instance `running` còn slot. Khi `max_concurrency > 1`, request mới sẽ cold start thay vì dùng instance running còn slot trống. Fix: tìm cả `running` instances có `available_slots > 0`.

## Features

- [ ] Eviction Policy pluggable (LRU, LFU, FIFO, TTL-per-state, Predictive) — config: `autoscaling.eviction_policy`
- [ ] PlacementStrategy cho autoscaler pool fill (HashAffinity, BinPack, Spread) — config: `autoscaling.placement`
- [ ] Control Policy chọn từ config thay vì hardcode trong sim_builder — config: `controller.policy`
- [ ] Reward Calculator pluggable (MultiObjective, Pareto) — config: `gym.reward_type`
- [ ] FindThenRoute load balancer strategy (warm-aware routing)
