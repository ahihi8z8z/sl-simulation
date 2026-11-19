import random
from collections import deque
from dataclasses import dataclass

import simpy

from .config import Action, Config
from .platform import Container, Platform


@dataclass(frozen=True)
class Reactive:  # ansible/files/runtimes.json nodejs:14 stem cell, used for gemma and yolo too
    min_count: int = 1
    max_count: int = 4
    ttl_s: float = 120.0
    threshold: int = 1
    increment: int = 1


@dataclass(frozen=True)
class OpenWhiskConfig(Config):
    pause_grace_s: float = 10.0  # whisk.container-proxy.timeouts.pause-grace
    log_collect_s: float = 2.0  # after a run, before Ready: KubernetesContainer waits waitForLogs for the sentinel
    stem_cells: int = 2  # initialCount
    reactive: Reactive = Reactive()
    prewarm_ttl_s: float | None = None  # None = the reactive value, also for the two counts
    prewarm_min_count: int | None = None
    prewarm_max_count: int | None = None
    prewarm_check_init_s: float = 600.0  # whisk.container-pool.prewarm-expiration-check-* defaults
    prewarm_check_s: float = 600.0
    prewarm_check_variance_s: float = 10.0


def fixed_stems(p: int, ttl_s: float = 3600.0):
    """Config fields that pin the stem cell count of every action to p: no reactive growth; max stays >= 1, as the
    API demands, so p = 0 is min 0 / max 1. Change p later with set_policy(min_count=p, max_count=max(p, 1))."""
    return dict(stem_cells=p, reactive=Reactive(threshold=10**6), prewarm_ttl_s=ttl_s,
                prewarm_min_count=p, prewarm_max_count=max(p, 1))


class OpenWhisk(Platform):
    """The OpenWhisk invoker with the lifecycle-control API."""

    def __init__(self, env: simpy.Environment, cfg: OpenWhiskConfig, rng: random.Random):
        super().__init__(env, cfg, rng)
        self.post_run_s = cfg.log_collect_s
        self.pause_grace = cfg.pause_grace_s
        self.buffer = deque()
        self.max_buffer = 0
        self.removed = 0
        self.evicted = 0
        self.prewarm_ttl = cfg.prewarm_ttl_s
        self.prewarm_min, self.prewarm_max = cfg.prewarm_min_count, cfg.prewarm_max_count
        self.hash_order = False
        self.prewarm_skipped = 0
        self.prewarm_expired = 0
        self.prewarm_removed = 0
        self.cold_starts = {}
        for a in cfg.actions:
            for _ in range(cfg.stem_cells):
                self._start_prewarm(a)
        # OpenWhisk draws the variance once, in whole seconds: nextInt(variance.toSeconds)
        variance = rng.randrange(int(cfg.prewarm_check_variance_s)) if cfg.prewarm_check_variance_s >= 1 else 0
        self.check_interval = cfg.prewarm_check_s + variance
        env.process(self._prewarm_check())

    def request(self, action: Action):
        if self.buffer:
            self.unplaced(self.job(action))
        else:
            super().request(action)

    def unplaced(self, job: tuple):
        self.buffer.append(job)
        self.max_buffer = max(self.max_buffer, len(self.buffer))

    def _retry_buffer(self):
        while self.buffer and self.dispatch(self.buffer[0]):
            self.buffer.popleft()

    # freePool is a Scala immutable Map: insertion order while it has never held more than 4 entries, then hash
    # order of the ActorRef (a random uid per container) for good. A container passes through freePool when taken.

    def new(self, a: Action):
        c = super().new(a)
        c.rank = self.rng.random()
        c.recreated = False
        return c

    def _note_free_size(self, n: int):
        if n > 4:
            self.hash_order = True

    def place(self, a: Action):
        if match := [c for c in self.warm_model if c.action is a]:
            return min(match, key=lambda c: c.rank) if self.hash_order else match[0]
        if match := [c for c in self.warm_container if c.action is a]:
            return min(match, key=lambda c: c.since)
        if self.has_space(a.memory_mb):
            return self.new(a)
        if self._evict(a.memory_mb):
            c = self.new(a)
            c.recreated = True
            return c
        return None

    def take(self, c: Container):
        cls = super().take(c)
        if cls != "warm":
            self._note_free_size(len(self.warm_model) + 1)
        if cls == "prewarm":
            self._start_prewarm(c.action)
        elif cls == "cold":
            self.cold_starts[c.action] = self.cold_starts.get(c.action, 0) + 1
            if c.recreated:
                cls = "recreated"
        return cls

    def _evict(self, memory_mb: int):
        # the new container is then created without another space check
        idle = list(self.warm_model)
        need = min(memory_mb, sum(c.action.memory_mb for c in idle))
        evicted = []
        while need > 0 and idle and sum(c.action.memory_mb for c in idle) >= need:
            c = min(idle, key=lambda c: c.last_used)
            idle.remove(c)
            evicted.append(c)
            need -= c.action.memory_mb
        self._account()
        for c in evicted:
            self.remove(c)
            self.evicted += 1
        return bool(evicted)

    def on_idle(self, c: Container):
        if c.state == "warm_model":
            self._note_free_size(len(self.warm_model))
            self.arm(c, self.pause_grace)
            self._retry_buffer()

    def on_timeout(self, c: Container):
        self.remove(c)
        self.removed += 1
        self._retry_buffer()

    def _set_pause_grace(self, pause_grace: float):
        now = self.env.now
        self.log.append(f"t={now:.1f} lifecycle_policy_updated field=pauseGrace old={self.pause_grace} "
                        f"new={pause_grace} free={len(self.warm_model)} busy={len(self.busy)}")
        old, self.pause_grace = self.pause_grace, pause_grace
        for c in list(self.warm_model):
            waited = now - c.since
            remaining = max(0.0, pause_grace - waited)
            event = "warm_idle_retroactive_expire" if remaining == 0 else "warm_idle_timeout_rescheduled"
            self.log.append(f"t={now:.1f} {event} container={c.id} action={c.action.name} waited={waited:.1f} "
                            f"old={old} new={pause_grace} remaining={remaining:.1f}")
            self.arm(c, remaining)

    def _start_prewarm(self, a: Action):
        if self.has_space(a.memory_mb):
            self.start(a)
        else:
            self.prewarm_skipped += 1

    def _ttl(self, p: Container):
        return self.prewarm_ttl if self.prewarm_ttl is not None else self.cfg.reactive.ttl_s

    def _counts(self, a: Action, min_count=None, max_count=None):
        lo = min_count if min_count is not None else self.prewarm_min
        hi = max_count if max_count is not None else self.prewarm_max
        r = self.cfg.reactive
        return (lo if lo is not None else r.min_count, hi if hi is not None else r.max_count)

    def _stems(self, a: Action):
        return sum(p.action is a for p in self.warm_container + self.starting)

    def _backfill(self):
        for a in self.cfg.actions:
            for _ in range(self._counts(a)[0] - self._stems(a)):
                self._start_prewarm(a)

    def _prewarm_check(self):
        yield self.env.timeout(self.cfg.prewarm_check_init_s)
        while True:
            expired = self._expire_prewarms()
            r = self.cfg.reactive
            for a in self.cfg.actions:
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

    def _expire_prewarms(self):
        expired = [p for p in self.warm_container if p.since + self._ttl(p) < self.env.now]
        self._account()
        for p in expired:
            self.remove(p)
            self.prewarm_expired += 1
        return expired

    def set_policy(self, pause_grace: float | None = None, prewarm_ttl: float | None = None,
                   min_count: int | None = None, max_count: int | None = None):
        """PATCH /lifecycle."""
        if pause_grace is not None and pause_grace < 0:
            raise ValueError("pause_grace must be >= 0")
        if prewarm_ttl is not None and prewarm_ttl <= 0:
            raise ValueError("prewarm_ttl must be > 0")
        if min_count is not None and min_count < 0 or max_count is not None and max_count < 1:
            raise ValueError("min_count must be >= 0 and max_count >= 1")
        for a in self.cfg.actions:
            lo, hi = self._counts(a, min_count, max_count)
            if lo > hi or self.cfg.reactive.increment > hi:
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
                # starting stem cells count but stay
                started = sorted((p for p in self.warm_container if p.action is a), key=lambda p: p.since)
                starting = sum(p.action is a for p in self.starting)
                excess = started[:max(0, len(started) + starting - min(lowered))]
                self._account()
                for p in excess:
                    self.remove(p)
                    self.prewarm_removed += 1
                    self.log.append(f"t={now:.1f} prewarm_retroactive_remove container={p.id} action={a.name} "
                                    f"age={now - p.since:.1f} limit={min(lowered)}")
                removed |= bool(excess)
            self._backfill()
            if removed:
                self._retry_buffer()
        if prewarm_ttl is not None:
            self.log.append(f"t={now:.1f} lifecycle_policy_updated field=prewarmTtl old={self.prewarm_ttl} "
                            f"new={prewarm_ttl} prewarms={len(self.warm_container)}")
            self.prewarm_ttl = prewarm_ttl
            expired = self._expire_prewarms()
            for p in expired:
                self.log.append(f"t={now:.1f} prewarm_retroactive_expire container={p.id} action={p.action.name} "
                                f"age={now - p.since:.1f} ttl={prewarm_ttl}")
            if expired:
                self._backfill()
                self._retry_buffer()
        if pause_grace is not None:
            self._set_pause_grace(pause_grace)
