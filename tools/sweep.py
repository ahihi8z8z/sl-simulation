import argparse
import csv
import json
import math
import os
import random
from multiprocessing import Pool

import simpy

from core.config import lab_actions
from core.theory import Theory, TheoryConfig

INF = math.inf
SLOTS = 15  # containers the testbed holds; the paper's figures have 60
# the paper's application names -> the measured profiles in data/profiles (mean times, used RAM per phase)
APPS = {"llm_2b": "gemma", "yolo_x": "yolo"}
PROFILES = {k: lab_actions((v,))[0] for k, v in APPS.items()}
# the paper's arrival rates divided by 4, as the system is 4 times smaller: the same load per container
RATES = {"llm_2b": [2.0 ** k for k in range(-8, 1)], "yolo_x": [2.0 ** k for k in range(-2, 7)]}
ON = 0.001  # the "theta ~ 0" of the 2-state ON case
FIELDS = ["app", "case", "rate", "theta3", "theta4", "seed", "duration_s", "requests", "served", "queued",
          "wait_s", "latency_s", "latency_p95_s", "ram_gbs_per_req", "cold", "prewarm", "warm"]

PHASES = ("pod_start", "init_prewarm", "run", "init_cold")  # the order of a job's times


def fit(a):
    """Log-normal (mu, sigma) of each transition time: the maximum-likelihood fit to the profile samples."""
    out = {}
    for phase, xs in zip(PHASES, (a.pod_start_samples, a.init_samples, a.run_samples, a.init_cold_samples)):
        logs = [math.log(x) for x in xs]
        mu = sum(logs) / len(logs)
        out[phase] = (mu, math.sqrt(sum((v - mu) ** 2 for v in logs) / len(logs)))
    return out


class Markov(Theory):
    """The paper's 4-state lifecycle as a simulation: Poisson arrivals, transition and service times from
    the log-normal fitted to the profile samples, and idle containers that fall back WM -> WC after a fixed
    1/theta4 and WC -> Null after a fixed 1/theta3 (inf = at once), as sl-sim-lite's statesweep maps the rate."""

    def __init__(self, env, cfg, rng, theta3, theta4):
        super().__init__(env, cfg, rng)
        self.theta = {"warm_container": theta3, "warm_model": theta4}
        (a,) = cfg.actions
        self.fit = fit(a)

    def job(self, a):
        self.arrivals.append(self.env.now)
        r = self.svc_rng[a]
        return a, self.env.now, tuple(r.lognormvariate(*self.fit[p]) for p in PHASES)

    def idle_time(self, c):
        theta = self.theta[c.state]
        return 0.0 if theta == INF else 1.0 / theta

    # One app and up to 60 containers: count slots and add up used RAM without walking the pools. A request's
    # startup and run RAM x time is added when it is placed, an idle pool's as time passes.

    def has_space(self, memory_mb):
        n = len(self.busy) + len(self.warm_model) + len(self.warm_container) + len(self.starting)
        return (n + 1) * memory_mb <= self.cfg.user_memory_mb

    def serve(self, c, job):
        a, _, (start, init, run, init_cold) = job
        super().serve(c, job)
        phases = {"run": run}
        if len(c.sched) == 2:
            phases["init"] = init
        if len(c.sched) == 3:
            phases.update(start=start, init=init_cold)
        for phase, seconds in phases.items():
            self.used_s[phase] += seconds * a.ram(phase)

    def _account(self):
        dt, self._mem_t = self.env.now - self._mem_t, self.env.now
        (a,) = self.cfg.actions
        self.used_s["ready"] += dt * len(self.warm_model) * a.ram("ready")
        self.used_s["stem"] += dt * len(self.warm_container) * a.ram("stem")


def grid(n):
    """Backward rates from the 2-state ON value to 100 /s, evenly spaced in log."""
    lo, hi = math.log10(ON), 2.0
    return [float(f"{10 ** (lo + (hi - lo) * i / (n - 1)):.4g}") for i in range(n)]


def cases(n3, n4):
    """Table "Evaluation test cases": (case, theta3, theta4)."""
    out = [("2-state ON", ON, ON), ("2-state OFF", INF, INF)]
    out += [("3-state WC", t, INF) for t in grid(n3)]
    out += [("3-state WM", INF, t) for t in grid(n3)]
    out += [("4-state", t3, t4) for t3 in grid(n4) for t4 in grid(n4)]
    return out


def run(point):
    app, case, rate, theta3, theta4, seed, requests, turnovers = point
    a = PROFILES[app]
    env = simpy.Environment()
    cfg = TheoryConfig(user_memory_mb=SLOTS * a.memory_mb, actions=(a,))
    p = Markov(env, cfg, random.Random(seed), theta3, theta4)
    arrivals = random.Random(seed * 1000 + 1)  # own stream: the same arrivals in every case

    def feed():
        while True:
            yield env.timeout(arrivals.expovariate(rate))
            p.request(a)

    env.process(feed())
    # long enough for `requests` arrivals and for `turnovers` lifetimes of the slowest idle state
    slow = max([1 / t for t in (theta3, theta4) if t != INF] + [a.run_s])
    duration = max(requests / rate, turnovers * slow)
    warmup = max(0.1 * duration, 3 * slow)
    env.run(until=warmup)
    n0, r0, m0 = len(p.arrivals), len(p.records), p.used_memory_seconds()["total"]
    env.run(until=warmup + duration)
    recs = p.records[r0:]
    reqs, served = len(p.arrivals) - n0, len(recs)
    cls = [r[2] for r in recs]
    lat = sorted(r[3] for r in recs)
    return dict(app=app, case=case, rate=rate, theta3=theta3, theta4=theta4, seed=seed, duration_s=duration,
                requests=reqs, served=served, queued=len(p.queue), wait_s=sum(r[4] for r in recs) / served,
                latency_s=sum(lat) / served, latency_p95_s=lat[min(served - 1, int(0.95 * served))],
                ram_gbs_per_req=(p.used_memory_seconds()["total"] - m0) / 1024 / served,
                cold=cls.count("cold"), prewarm=cls.count("prewarm"), warm=cls.count("warm"))


def main():
    ap = argparse.ArgumentParser(description="Simulate the paper's evaluation: every test case at every arrival rate.")
    ap.add_argument("--apps", default="yolo_x,llm_2b")
    ap.add_argument("--n3", type=int, default=41, help="theta values per 3-state case, log-spaced in [0.001, 100]")
    ap.add_argument("--n4", type=int, default=11, help="theta values per axis of the 4-state case")
    ap.add_argument("--requests", type=int, default=10000, help="least arrivals per run")
    ap.add_argument("--turnovers", type=float, default=10, help="least run length, in mean idle lifetimes")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--out", default="sweep/results")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    points = [(app, case, rate, t3, t4, seed, args.requests, args.turnovers)
              for seed in map(int, args.seeds.split(",")) for app in args.apps.split(",")
              for rate in RATES[app] for case, t3, t4 in cases(args.n3, args.n4)]
    points.sort(key=lambda p: -max(p[6], p[2] * p[7] / min(p[3], p[4], 1)))  # longest runs first
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(dict(vars(args), slots=SLOTS, theta_on=ON, profiles={
            k: dict(profile=APPS[k], pod_start_s=a.pod_start_s, init_s=a.init_s, init_cold_s=a.init_cold_s,
                    run_s=a.run_s, ram_mb=a.ram_mb)
            for k, a in PROFILES.items()}), f, indent=1)
    with Pool(args.jobs) as pool, open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run, points, chunksize=1), 1):
            w.writerow(row)
            if i % 100 == 0 or i == len(points):
                f.flush()
                print(f"{i}/{len(points)}", flush=True)


if __name__ == "__main__":
    main()
