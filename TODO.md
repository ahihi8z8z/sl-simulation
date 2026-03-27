# TODO

## Bugs / Logic issues

- [x] `find_reusable_instance()` now also returns running instances with free concurrency slots (warm still preferred).

## Features

- [ ] Eviction Policy pluggable (LRU, LFU, FIFO, TTL-per-state, Predictive) — config: `autoscaling.eviction_policy`
- [ ] PlacementStrategy cho autoscaler pool fill (HashAffinity, BinPack, Spread) — config: `autoscaling.placement`
- [ ] Control Policy chọn từ config thay vì hardcode trong sim_builder — config: `controller.policy`
- [ ] Reward Calculator pluggable (MultiObjective, Pareto) — config: `gym.reward_type`
- [ ] FindThenRoute load balancer strategy (warm-aware routing)
