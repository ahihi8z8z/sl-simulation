import math
import random

import gymnasium as gym
import numpy as np
import simpy

from core.config import APPS, lab_actions
from core.monitor import Monitor
from core.openwhisk import OpenWhisk, OpenWhiskConfig, fixed_stems
from core.workload import read_trace, trace_arrivals

GRACES = (15.0, 40.0, 100.0, 240.0, 600.0, 1200.0, 2400.0)
STEMS = (0, 1, 2)
ACTIONS = [(g, p) for g in GRACES for p in STEMS]  # index = grace index * 3 + stems, as the testbed controller
OBS = ("req5", "warm", "pre", "grace", "stems", "sin", "cos", "used")


class OwEnv(gym.Env):
    """One app on one invoker; every step applies (pause grace, stem cells) and runs `interval_s`.

    Observation: log1p requests of the last step, log1p warm models (idle + busy), log1p stem cells (started +
    starting), the action in force (grace index / 6, stems / 2), sin and cos of the time of day, 10 * used-RAM
    fraction of the last step.
    Reward: -[alpha * used GB*s / m_ref + (1 - alpha) * latency / l_ref] for the step.
    """

    def __init__(self, app: str, trace: str, alpha: float, m_ref: float, l_ref: float,
                 days=tuple(range(10)), n_days: int = 1, interval_s: float = 300.0,
                 grace_s: float = 240.0, stems: int = 0, stem_ttl_s: float = 3600.0):
        self.app, self.alpha, self.m_ref, self.l_ref = app, alpha, m_ref, l_ref
        self.counts = read_trace(trace)
        self.days, self.n_days, self.interval_s = tuple(days), n_days, interval_s
        self.a0, self.stem_ttl_s = (grace_s, stems), stem_ttl_s
        self.n_steps = round(n_days * 86400 / interval_s)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(-1.0, np.inf, (len(OBS),), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.day = options.get("day", self.days[self.np_random.integers(len(self.days))])
        self.sim_seed = options.get("sim_seed", int(self.np_random.integers(2**31)))
        start = self.day * 1440
        counts = {m - start: n for m, n in self.counts.items() if start <= m < start + self.n_days * 1440}
        self.env = simpy.Environment()
        grace, p = self.a = self.a0
        cfg = OpenWhiskConfig(pause_grace_s=grace, actions=lab_actions((self.app,)), **fixed_stems(p, self.stem_ttl_s))
        self.inv = OpenWhisk(self.env, cfg, random.Random(self.sim_seed))
        (a,) = cfg.actions
        rng = random.Random(self.sim_seed * 1000 + APPS.index(a.name) + 1)  # as tools/run_fixed
        self.env.process(self._feed(a, trace_arrivals(rng, counts)))
        self.monitor = Monitor(self.inv)
        self.t = 0
        return self._obs(self.monitor.step()["live"]), {"day": self.day, "sim_seed": self.sim_seed}

    def _feed(self, a, times):
        for t in times:
            yield self.env.timeout(t - self.env.now)
            self.inv.request(a)

    def _obs(self, live: dict):
        grace, p = self.a
        tod = 2 * math.pi * self.env.now / 86400
        return np.array([math.log1p(live["req5"]), math.log1p(live["warm"]), math.log1p(live["pre"]),
                         GRACES.index(grace) / (len(GRACES) - 1), p / STEMS[-1], math.sin(tod), math.cos(tod),
                         10 * live["u_last"]], np.float32)

    def step(self, action):
        grace, p = ACTIONS[action]
        inv = self.inv
        if p != self.a[1]:
            inv.set_policy(min_count=p, max_count=max(p, 1))
        if grace != self.a[0]:
            inv.set_policy(pause_grace=grace)
        self.a = (grace, p)
        self.t += 1
        self.env.run(until=self.t * self.interval_s)
        m = self.monitor.step()
        recs, gbs_used = m["started"], m["gbs_used"]
        latency = sum(r[3] for r in recs)
        done = self.t >= self.n_steps
        # requests seen but not started yet (buffer or startup): charge the wait so far, no future draws
        started = {r[0] for r in inv.records}
        pending = [self.env.now - t for t in inv.arrivals if t not in started] if done else []
        terminal = -(1 - self.alpha) * sum(pending) / self.l_ref
        reward = -(self.alpha * gbs_used / self.m_ref + (1 - self.alpha) * latency / self.l_ref) + terminal
        info = {"gbs_used": gbs_used, "latency": latency, "started": len(recs), "requests": len(m["arrivals"]),
                "classes": {c: sum(r[2] == c for r in recs) for c in ("warm", "prewarm", "cold", "recreated")},
                "terminal_penalty": terminal, "pending": len(pending), "pending_wait": sum(pending)}
        return self._obs(m["live"]), reward, done, False, info
