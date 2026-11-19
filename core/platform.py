import math
import random
import statistics

import simpy

from .config import Action, Config

START = {None: "cold", "warm_container": "prewarm", "warm_model": "warm"}  # a request's start class


def draw(rng: random.Random, samples: tuple, const: float):
    return rng.choice(samples) if samples else const


def lognormal(samples: tuple):
    """(mu, sigma) of the log-normal with the mean and variance of the samples."""
    m, v = statistics.mean(samples), statistics.variance(samples)
    s2 = math.log(1 + v / m ** 2)
    return math.log(m) - s2 / 2, math.sqrt(s2)


class Container:
    def __init__(self, cid: int, action: Action):
        self.id, self.action = cid, action
        self.state = None  # None (Null), starting, warm_container, warm_model or busy
        self.since = None  # when it entered its idle state
        self.last_used = None
        self.epoch = 0  # bumped on every state change, so an older timer is ignored
        self.sched = None  # while busy: [(t_from, phase)] of the request it serves (start / init / run)


class Platform:
    """Mechanics of a serverless platform with one memory budget. Container states:
    Null -> starting -> warm_container (app up, not initialized) -> warm_model (initialized, idle) <-> busy.
    A subclass is a platform's policy: it decides in place, unplaced, on_idle, on_timeout and set_policy, and
    acts with start, new, demote, remove and arm."""

    post_run_s = 0.0  # a container stays busy this long after a run

    def __init__(self, env: simpy.Environment, cfg: Config, rng: random.Random):
        self.env, self.cfg, self.rng = env, cfg, rng
        # one stream per action for request service times, drawn at arrival: the same request gets
        # the same times whatever the policy
        self.svc_rng = {a: random.Random(rng.random()) for a in cfg.actions}
        self.fit = {}  # (action, transition) -> (mu, sigma)
        if cfg.transition == "lognormal":
            self.fit = {(a, k): lognormal(getattr(a, k + "_samples")) for a in cfg.actions
                        for k in ("pod_start", "init", "init_cold") if len(getattr(a, k + "_samples")) > 1}
        self.starting, self.warm_container, self.warm_model, self.busy = [], [], [], []  # each in arrival order
        self.arrivals = []  # arrival time of every request
        self.records = []  # (arrival, action, start class, latency, queue wait), once the run starts
        self.peak_busy = 0
        self.state_s = {"warm_model": 0.0, "warm_container": 0.0, "starting": 0.0, "busy": 0.0}  # container*s per state
        self.used_s = {"start": 0.0, "stem": 0.0, "init": 0.0, "run": 0.0, "ready": 0.0}  # used MB*s per phase
        self._mem_t = 0.0
        self.log = []
        self._ids = 0

    # decisions

    def place(self, a: Action) -> Container | None:
        """The idle or new container that serves a request of `a`, or None."""
        raise NotImplementedError

    def unplaced(self, job: tuple):
        """`place` found no container for this request."""
        raise NotImplementedError

    def on_idle(self, c: Container):
        """`c` just became a warm container or a warm model."""
        raise NotImplementedError

    def on_timeout(self, c: Container):
        """The timer armed on idle `c` ran out."""
        raise NotImplementedError

    def set_policy(self, **knobs):
        raise NotImplementedError

    # mechanics

    def request(self, action: Action):
        job = self.job(action)
        if not self.dispatch(job):
            self.unplaced(job)

    def job(self, action: Action):
        self.arrivals.append(self.env.now)
        r = self.svc_rng[action]
        init_cold = action.init_s if action.init_cold_s is None else action.init_cold_s
        times = (self.transition(r, action, "pod_start", action.pod_start_s),
                 self.transition(r, action, "init", action.init_s), draw(r, action.run_samples, action.run_s),
                 self.transition(r, action, "init_cold", init_cold))
        return action, self.env.now, times

    def transition(self, rng: random.Random, a: Action, k: str, const: float):
        fit = self.fit.get((a, k))
        return rng.lognormvariate(*fit) if fit else draw(rng, getattr(a, k + "_samples"), const)

    def dispatch(self, job: tuple):
        c = self.place(job[0])
        if c:
            self.serve(c, job)
        return c is not None

    def has_space(self, memory_mb: int):
        used = sum(c.action.memory_mb for c in self.busy + self.warm_model + self.warm_container + self.starting)
        return used + memory_mb <= self.cfg.user_memory_mb

    def new(self, a: Action):
        """A Null container, to return from `place`."""
        self._ids += 1
        return Container(self._ids, a)

    def start(self, a: Action):
        """Null -> starting -> warm_container, ahead of any request."""
        self._account()
        c = self.new(a)
        self._move(c, "starting")
        self.env.process(self._start(c))
        return c

    def _start(self, c: Container):
        yield self.env.timeout(self.transition(self.rng, c.action, "pod_start", c.action.pod_start_s))
        self._idle(c, "warm_container")

    def take(self, c: Container):
        """Idle or Null -> busy; returns the start class."""
        self._account()
        cls = START[c.state]
        self._move(c, "busy")
        return cls

    def serve(self, c: Container, job: tuple):
        a, t0, (pod_start, init, run, init_cold) = job
        cls = self.take(c)
        now = self.env.now
        c.sched = ([(now, "run")] if cls == "warm" else [(now, "init"), (now + init, "run")] if cls == "prewarm"
                   else [(now, "start"), (now + pod_start, "init"), (now + pod_start + init_cold, "run")])
        startup = None if cls == "warm" else init if cls == "prewarm" else pod_start + init_cold
        self.env.process(self._serve(c, a, t0, cls, startup, run))

    def _serve(self, c: Container, a: Action, t0: float, cls: str, startup: float | None, run: float):
        wait = self.env.now - t0
        c.last_used = self.env.now
        if startup is not None:
            yield self.env.timeout(startup)
        self.records.append((t0, a.name, cls, self.env.now - t0, wait))  # latency ends when the run starts
        yield self.env.timeout(run + self.post_run_s)
        self._idle(c, "warm_model")

    def demote(self, c: Container):
        """warm_model -> warm_container."""
        self._idle(c, "warm_container")

    def remove(self, c: Container):
        """Idle -> Null."""
        self._account()
        self._move(c, None)

    def arm(self, c: Container, delay: float):
        """Call on_timeout(c) after `delay`, unless `c` changes state or is armed again first."""
        c.epoch += 1
        self.env.process(self._timer(c, c.epoch, delay))

    def _timer(self, c: Container, epoch: int, delay: float):
        yield self.env.timeout(delay)
        if c.epoch == epoch:
            self.on_timeout(c)

    def _idle(self, c: Container, state: str):
        self._account()
        self._move(c, state)
        c.since = self.env.now
        self.on_idle(c)

    def _move(self, c: Container, state: str | None):
        if c.state:
            getattr(self, c.state).remove(c)
        if state:
            getattr(self, state).append(c)
        c.state = state
        c.epoch += 1
        self.peak_busy = max(self.peak_busy, len(self.busy))

    # memory

    def _account(self):
        """Add the time since the last change: containers per state, used RAM per phase."""
        t0, t1 = self._mem_t, self.env.now
        dt = t1 - t0
        self._mem_t = t1
        for state in self.state_s:
            self.state_s[state] += dt * len(getattr(self, state))
        if dt <= 0:
            return
        for phase, pool in (("ready", self.warm_model), ("stem", self.warm_container), ("start", self.starting)):
            self.used_s[phase] += dt * sum(c.action.ram(phase) for c in pool)
        for c in self.busy:
            sched = c.sched or [(t0, "run")]
            for i, (lo, phase) in enumerate(sched):
                hi = sched[i + 1][0] if i + 1 < len(sched) else t1
                span = min(hi, t1) - max(lo, t0)
                if span > 0:
                    self.used_s[phase] += span * c.action.ram(phase)

    def container_seconds(self):
        self._account()
        return dict(self.state_s, total=sum(self.state_s.values()))

    def used_memory_seconds(self):
        self._account()
        return dict(self.used_s, total=sum(self.used_s.values()))
