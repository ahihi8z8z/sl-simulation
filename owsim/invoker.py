import random
from collections import deque

import simpy

from .config import Action, Config


def draw(rng: random.Random, samples: tuple, const: float):
    return rng.choice(samples) if samples else const


class Container:
    def __init__(self, cid: int, action: Action, rank: float):
        self.id, self.action, self.rank = cid, action, rank
        self.created_at = None  # prewarm: when the stem cell finished starting
        self.ready_at = None
        self.last_used = None  # WarmedData.lastUsed: set when a run starts
        self.epoch = 0  # bumped when taken, so an older keep-alive timer is ignored


class Invoker:
    def __init__(self, env: simpy.Environment, cfg: Config, rng: random.Random):
        self.env, self.cfg, self.rng = env, cfg, rng
        # one stream per action for request service times, drawn at arrival: the same request gets
        # the same times whatever the knobs
        self.svc_rng = {a: random.Random(rng.random()) for a in cfg.actions}
        self.pause_grace = cfg.pause_grace_s
        self.prewarm_ttl = cfg.prewarm_ttl_s
        self.prewarm_min, self.prewarm_max = cfg.prewarm_min_count, cfg.prewarm_max_count
        self.free = []  # Ready containers, in the order they became free
        self.busy = []
        self.prewarm_starting = []
        self.prewarmed = []
        self.buffer = deque()  # runBuffer: (action, arrival, times); only the head is retried
        self.hash_order = False
        self.records = []  # (arrival, action, class, latency, queue wait)
        self.mem_s = {"ready": 0.0, "prewarmed": 0.0, "prewarm_starting": 0.0, "busy": 0.0}  # MB*s per state
        self._mem_t = 0.0
        self.removed = 0
        self.evicted = 0
        self.prewarm_skipped = 0
        self.max_buffer = 0
        self.log = []  # policy decisions, like the invoker's lifecycle log lines
        self.prewarm_expired = 0
        self.prewarm_removed = 0  # stem cells removed by a lowered min/max count
        self.cold_starts = {}  # per action since the last check (coldStartCount)
        self._ids = 0
        for a in cfg.actions:
            for _ in range(a.stem_cells):
                self._start_prewarm(a)
        # OpenWhisk draws the variance once, in whole seconds: nextInt(variance.toSeconds)
        variance = rng.randrange(int(cfg.prewarm_check_variance_s)) if cfg.prewarm_check_variance_s >= 1 else 0
        self.check_interval = cfg.prewarm_check_s + variance
        env.process(self._prewarm_check())

    def request(self, action: Action):
        r = self.svc_rng[action]
        init_cold = action.init_s if action.init_cold_s is None else action.init_cold_s
        times = (draw(r, action.pod_start_samples, action.pod_start_s), draw(r, action.init_samples, action.init_s),
                 draw(r, action.run_samples, action.run_s), draw(r, action.init_cold_samples, init_cold))
        if self.buffer or not self._dispatch(action, self.env.now, times):
            self.buffer.append((action, self.env.now, times))
            self.max_buffer = max(self.max_buffer, len(self.buffer))

    def _dispatch(self, a: Action, t0: float, times: tuple):
        placed = self._place(a)
        if placed:
            self.env.process(self._run(a, t0, times, *placed))
        return placed is not None

    def _place(self, a: Action):
        if c := self._take_warm(a):
            return c, "warm"
        if c := self._take_prewarm(a):
            return c, "prewarm"
        if self._has_space(a.memory_mb):
            return self._new_container(a), "cold"
        if self._evict(a.memory_mb):
            return self._new_container(a), "recreated"
        return None

    def _retry_buffer(self):
        while self.buffer and self._dispatch(*self.buffer[0]):
            self.buffer.popleft()

    def _run(self, a: Action, t0: float, times: tuple, c: Container, cls: str):
        pod_start, init, run, init_cold = times
        wait = self.env.now - t0
        c.last_used = self.env.now
        if cls == "prewarm":
            yield self.env.timeout(init)
        elif cls != "warm":
            yield self.env.timeout(pod_start + init_cold)
        self.records.append((t0, a.name, cls, self.env.now - t0, wait))  # latency ends when the run starts (waitTime + initTime)
        yield self.env.timeout(run + self.cfg.log_collect_s)
        self._set_ready(c)

    def _take_warm(self, a: Action):
        # OpenWhisk's freePool is a Scala immutable Map: insertion order while it has never held more
        # than 4 entries, then hash order of the ActorRef (a random uid per container) for good.
        match = [c for c in self.free if c.action is a]
        if not match:
            return None
        c = min(match, key=lambda c: c.rank) if self.hash_order else match[0]
        self._account()
        self.free.remove(c)
        c.epoch += 1
        self.busy.append(c)
        return c

    def _take_prewarm(self, a: Action):
        match = [p for p in self.prewarmed if p.action is a]
        if not match:
            return None
        p = min(match, key=lambda p: p.created_at)  # earliest stem-cell expiry first
        self._account()
        self.prewarmed.remove(p)
        self._note_free_size(len(self.free) + 1)
        self.busy.append(p)
        self._start_prewarm(a)  # a taken stem cell is always replaced
        return p

    def _new_container(self, a: Action):
        self._account()
        c = self._container(a)
        self._note_free_size(len(self.free) + 1)  # createContainer puts it in freePool first
        self.busy.append(c)
        self.cold_starts[a] = self.cold_starts.get(a, 0) + 1  # "recreated" counts too
        return c

    def _has_space(self, memory_mb: int):
        used = sum(c.action.memory_mb for c in self.busy + self.free + self.prewarmed + self.prewarm_starting)
        return used + memory_mb <= self.cfg.user_memory_mb

    def _evict(self, memory_mb: int):
        # ContainerPool.remove: free Ready containers, oldest lastUsed first, until the need is covered;
        # the new container is then created without another space check
        idle = list(self.free)
        need = min(memory_mb, sum(c.action.memory_mb for c in idle))
        evicted = []
        while need > 0 and idle and sum(c.action.memory_mb for c in idle) >= need:
            c = min(idle, key=lambda c: c.last_used)
            idle.remove(c)
            evicted.append(c)
            need -= c.action.memory_mb
        self._account()
        for c in evicted:
            self.free.remove(c)
            c.epoch += 1
            self.evicted += 1
        return bool(evicted)

    def _container(self, a: Action):
        self._ids += 1
        return Container(self._ids, a, self.rng.random())

    def _start_prewarm(self, a: Action):
        if not self._has_space(a.memory_mb):
            self.prewarm_skipped += 1
            return
        self._account()
        p = self._container(a)
        self.prewarm_starting.append(p)
        self.env.process(self._prewarm_up(p))

    def _prewarm_up(self, p: Container):
        yield self.env.timeout(draw(self.rng, p.action.pod_start_samples, p.action.pod_start_s))
        self._account()
        self.prewarm_starting.remove(p)
        p.created_at = self.env.now
        self.prewarmed.append(p)

    def _ttl(self, p: Container):
        return self.prewarm_ttl if self.prewarm_ttl is not None else p.action.reactive.ttl_s

    def _counts(self, a: Action, min_count=None, max_count=None):
        """Effective (minCount, maxCount) of a stem cell under the given (default: current) overrides."""
        lo = min_count if min_count is not None else self.prewarm_min
        hi = max_count if max_count is not None else self.prewarm_max
        return (lo if lo is not None else a.reactive.min_count, hi if hi is not None else a.reactive.max_count)

    def _stems(self, a: Action):
        return sum(p.action is a for p in self.prewarmed + self.prewarm_starting)

    def _backfill(self):
        """Non-scheduled adjust: start stem cells up to minCount."""
        for a in self.cfg.actions:
            for _ in range(self._counts(a)[0] - self._stems(a)):
                self._start_prewarm(a)

    def _prewarm_check(self):
        yield self.env.timeout(self.cfg.prewarm_check_init_s)
        while True:
            expired = self._expire_prewarms()
            for a in self.cfg.actions:
                r = a.reactive
                lo, hi = self._counts(a)
                want = lo
                if a in self.cold_starts:
                    want = min(max(lo, self.cold_starts[a] // r.threshold * r.increment), hi)
                for _ in range(want - self._stems(a)):
                    self._start_prewarm(a)
            self.cold_starts = {}
            if expired:
                self._retry_buffer()
            yield self.env.timeout(self.check_interval)

    def _set_ready(self, c: Container):
        self._account()
        self.busy.remove(c)
        self.free.append(c)
        self._note_free_size(len(self.free))
        c.ready_at = self.env.now
        self.env.process(self._keepalive(c, c.epoch, self.pause_grace))
        self._retry_buffer()

    def _keepalive(self, c: Container, epoch: int, delay: float):
        yield self.env.timeout(delay)
        if c.epoch != epoch:
            return
        self._account()
        self.free.remove(c)
        self.removed += 1
        self._retry_buffer()

    def _expire_prewarms(self):
        expired = [p for p in self.prewarmed if p.created_at + self._ttl(p) < self.env.now]
        self._account()
        for p in expired:
            self.prewarmed.remove(p)
            self.prewarm_expired += 1
        return expired

    def _account(self):
        """Add memory * time held in each state since the last change."""
        dt = self.env.now - self._mem_t
        self._mem_t = self.env.now
        for state, pool in (("ready", self.free), ("prewarmed", self.prewarmed),
                            ("prewarm_starting", self.prewarm_starting), ("busy", self.busy)):
            self.mem_s[state] += dt * sum(c.action.memory_mb for c in pool)

    def memory_seconds(self):
        self._account()
        return dict(self.mem_s, total=sum(self.mem_s.values()))

    def set_policy(self, pause_grace: float | None = None, prewarm_ttl: float | None = None,
                   min_count: int | None = None, max_count: int | None = None):
        """PATCH /lifecycle: None leaves a knob unchanged; all are validated before any is applied, then the
        counts, the ttl and the pause grace are applied in that order."""
        if pause_grace is not None and pause_grace < 0:
            raise ValueError("pause_grace must be >= 0")
        if prewarm_ttl is not None and prewarm_ttl <= 0:
            raise ValueError("prewarm_ttl must be > 0")
        if min_count is not None and min_count < 0 or max_count is not None and max_count < 1:
            raise ValueError("min_count must be >= 0 and max_count >= 1")
        for a in self.cfg.actions:
            lo, hi = self._counts(a, min_count, max_count)
            if lo > hi or a.reactive.increment > hi:
                raise ValueError(f"counts {lo}/{hi} do not fit the stem cell of {a.name}")
        now = self.env.now
        if min_count is not None or max_count is not None:
            self.log.append(f"t={now:.1f} lifecycle_policy_updated field=prewarmCounts old_min={self.prewarm_min} "
                            f"new_min={min_count if min_count is not None else self.prewarm_min} "
                            f"old_max={self.prewarm_max} new_max={max_count if max_count is not None else self.prewarm_max}")
            old = {a: self._counts(a) for a in self.cfg.actions}
            if min_count is not None:
                self.prewarm_min = min_count
            if max_count is not None:
                self.prewarm_max = max_count
            removed = False
            for a in self.cfg.actions:
                (lo0, hi0), (lo, hi) = old[a], self._counts(a)
                lowered = [c for c, c0 in ((lo, lo0), (hi, hi0)) if c < c0]
                if not lowered:
                    continue
                # oldest started stem cells above the lowered count go; starting ones count but stay
                started = sorted((p for p in self.prewarmed if p.action is a), key=lambda p: p.created_at)
                starting = sum(p.action is a for p in self.prewarm_starting)
                excess = started[:max(0, len(started) + starting - min(lowered))]
                self._account()
                for p in excess:
                    self.prewarmed.remove(p)
                    self.prewarm_removed += 1
                    self.log.append(f"t={now:.1f} prewarm_retroactive_remove container={p.id} action={a.name} "
                                    f"age={now - p.created_at:.1f} limit={min(lowered)}")
                removed |= bool(excess)
            self._backfill()
            if removed:
                self._retry_buffer()
        if prewarm_ttl is not None:
            self.log.append(f"t={now:.1f} lifecycle_policy_updated field=prewarmTtl old={self.prewarm_ttl} "
                            f"new={prewarm_ttl} prewarms={len(self.prewarmed)}")
            self.prewarm_ttl = prewarm_ttl
            expired = self._expire_prewarms()
            for p in expired:
                self.log.append(f"t={now:.1f} prewarm_retroactive_expire container={p.id} action={p.action.name} "
                                f"age={now - p.created_at:.1f} ttl={prewarm_ttl}")
            if expired:
                # non-scheduled backfill: only up to minCount, cold-start counts untouched
                self._backfill()
                self._retry_buffer()
        if pause_grace is not None:
            self.log.append(f"t={now:.1f} lifecycle_policy_updated field=pauseGrace old={self.pause_grace} "
                            f"new={pause_grace} free={len(self.free)} busy={len(self.busy)}")
            old, self.pause_grace = self.pause_grace, pause_grace
            # Ready containers keep the time already waited; busy ones use the new value at their next Ready
            for c in list(self.free):
                waited = now - c.ready_at
                remaining = max(0.0, pause_grace - waited)
                event = "warm_idle_retroactive_expire" if remaining == 0 else "warm_idle_timeout_rescheduled"
                self.log.append(f"t={now:.1f} {event} container={c.id} action={c.action.name} waited={waited:.1f} "
                                f"old={old} new={pause_grace} remaining={remaining:.1f}")
                c.epoch += 1
                self.env.process(self._keepalive(c, c.epoch, remaining))

    def _note_free_size(self, n: int):
        if n > 4:
            self.hash_order = True
