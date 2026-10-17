import random

import gymnasium as gym
import numpy as np
import simpy

from .config import APPS, Config, lab_actions
from .invoker import Invoker
from .workload import read_trace, trace_arrivals

GRACES = (10.0, 60.0, 360.0, 1800.0)
TTLS = (10.0, 60.0, 300.0, 600.0)
OBS = ("ready", "busy", "prewarm", "buffer", "grace", "ttl", "requests")


class OwEnv(gym.Env):
    """One app on one invoker; every step applies (grace, ttl) and runs `interval_s`.

    Observation: only what the testbed exposes (Prometheus pool gauges, GET /lifecycle, the request
    count of the last step). Reward: -[alpha * GB*s / m_ref + (1 - alpha) * latency / l_ref] for the step.
    """

    def __init__(self, app: str, trace: str, alpha: float, m_ref: float, l_ref: float,
                 days=tuple(range(10)), n_days: int = 1, interval_s: float = 300.0,
                 grace_s: float = 10.0, ttl_s: float = 600.0):
        self.app, self.alpha, self.m_ref, self.l_ref = app, alpha, m_ref, l_ref
        self.counts = read_trace(trace)
        self.days, self.n_days, self.interval_s = tuple(days), n_days, interval_s
        self.grace0, self.ttl0 = grace_s, ttl_s
        self.n_steps = round(n_days * 86400 / interval_s)
        self.action_space = gym.spaces.MultiDiscrete([len(GRACES), len(TTLS)])
        self.observation_space = gym.spaces.Box(0.0, np.inf, (len(OBS),), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.day = options.get("day", self.days[self.np_random.integers(len(self.days))])
        self.sim_seed = options.get("sim_seed", int(self.np_random.integers(2**31)))
        start = self.day * 1440
        counts = {m - start: n for m, n in self.counts.items() if start <= m < start + self.n_days * 1440}
        self.env = simpy.Environment()
        cfg = Config(pause_grace_s=self.grace0, prewarm_ttl_s=self.ttl0, actions=lab_actions((self.app,)))
        self.inv = Invoker(self.env, cfg, random.Random(self.sim_seed))
        (a,) = cfg.actions
        rng = random.Random(self.sim_seed * 1000 + APPS.index(a.name) + 1)  # as tools/run_fixed.simulate
        self.arrivals = []  # arrival times so far: requests the controller has already seen
        self.env.process(self._feed(a, trace_arrivals(rng, counts)))
        self.t, self.n_rec, self.n_arr, self.mem = 0, 0, 0, 0.0
        return self._obs(0), {"day": self.day, "sim_seed": self.sim_seed}

    def _feed(self, a, times):
        for t in times:
            yield self.env.timeout(t - self.env.now)
            self.arrivals.append(self.env.now)
            self.inv.request(a)

    def _obs(self, requests: int):
        inv = self.inv
        return np.array([len(inv.free), len(inv.busy), len(inv.prewarmed) + len(inv.prewarm_starting),
                         len(inv.buffer), inv.pause_grace, inv.prewarm_ttl, requests], np.float32)

    def step(self, action):
        grace, ttl = GRACES[action[0]], TTLS[action[1]]
        inv = self.inv
        # PATCH only what changes: a no-op PATCH would still expire stem cells off the 1 s check
        inv.set_policy(pause_grace=grace if grace != inv.pause_grace else None,
                       prewarm_ttl=ttl if ttl != inv.prewarm_ttl else None)
        self.t += 1
        self.env.run(until=self.t * self.interval_s)

        recs = inv.records[self.n_rec:]
        self.n_rec = len(inv.records)
        requests = len(self.arrivals) - self.n_arr
        self.n_arr = len(self.arrivals)
        mem = inv.memory_seconds()
        gbs = (mem["total"] - self.mem) / 1024
        self.mem = mem["total"]
        latency = sum(r[3] for r in recs)
        done = self.t >= self.n_steps
        # requests seen but not started yet (buffer or startup): charge the wait so far, no future draws
        started = {r[0] for r in inv.records}
        pending = [self.env.now - t for t in self.arrivals if t not in started] if done else []
        terminal = -(1 - self.alpha) * sum(pending) / self.l_ref
        reward = -(self.alpha * gbs / self.m_ref + (1 - self.alpha) * latency / self.l_ref) + terminal
        info = {"gbs": gbs, "latency": latency, "started": len(recs), "requests": requests,
                "classes": {c: sum(r[2] == c for r in recs) for c in ("warm", "prewarm", "cold", "recreated")},
                "terminal_penalty": terminal, "pending": len(pending), "pending_wait": sum(pending)}
        return self._obs(requests), reward, done, False, info
