import random
from collections import deque
from dataclasses import dataclass

import simpy

from .config import Action, Config
from .platform import Container, Platform


@dataclass(frozen=True)
class TheoryConfig(Config):
    model_timeout_s: float = 10.0  # warm model -> warm container
    container_timeout_s: float = 600.0  # warm container -> Null


class Theory(Platform):
    """The theoretical model: an idle warm model becomes a warm container after one timeout and the warm container
    is removed after another; a request that finds no container and no free memory waits in a FIFO queue."""

    def __init__(self, env: simpy.Environment, cfg: TheoryConfig, rng: random.Random):
        super().__init__(env, cfg, rng)
        self.timeout = {"warm_model": cfg.model_timeout_s, "warm_container": cfg.container_timeout_s}
        self.queue = deque()
        self.max_queue = self.demoted = self.removed = 0

    def request(self, action: Action):
        if self.queue:
            self.unplaced(self.job(action))
        else:
            super().request(action)

    def place(self, a: Action):
        for c in self.warm_model + self.warm_container:
            if c.action is a:
                return c
        return self.new(a) if self.has_space(a.memory_mb) else None

    def unplaced(self, job: tuple):
        self.queue.append(job)
        self.max_queue = max(self.max_queue, len(self.queue))

    def _retry_queue(self):
        while self.queue and self.dispatch(self.queue[0]):
            self.queue.popleft()

    def idle_time(self, c: Container):
        return self.timeout[c.state]

    def on_idle(self, c: Container):
        self.arm(c, self.idle_time(c))
        self._retry_queue()

    def on_timeout(self, c: Container):
        if c.state == "warm_model":
            self.demote(c)
            self.demoted += 1
        else:
            self.remove(c)
            self.removed += 1
            self._retry_queue()

    def set_policy(self, model_timeout: float | None = None, container_timeout: float | None = None):
        new = {"warm_model": model_timeout, "warm_container": container_timeout}
        if any(t is not None and t < 0 for t in new.values()):
            raise ValueError("timeouts must be >= 0")
        for state, t in new.items():
            if t is not None:
                self.timeout[state] = t
                for c in getattr(self, state):  # idle containers keep the time already waited
                    self.arm(c, max(0.0, t - (self.env.now - c.since)))
