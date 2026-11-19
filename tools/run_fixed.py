import argparse
import random
from collections import Counter

import simpy

from core.config import APPS, lab_actions
from core.openwhisk import OpenWhisk, OpenWhiskConfig, fixed_stems
from core.workload import poisson, read_trace, trace_arrivals


def feed(env, inv, action, times):
    for t in times:
        yield env.timeout(t - env.now)
        inv.request(action)


def pct(xs, p):
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Run one app on the OpenWhisk model with a fixed policy; print a summary.")
    ap.add_argument("--app", required=True, choices=APPS)
    ap.add_argument("--grace", type=float, default=10.0, help="pause grace (s)")
    ap.add_argument("--stems", type=int, default=None, help="fixed stem cell count; default: OpenWhisk's reactive stem cells")
    ap.add_argument("--trace", default=None, help="minute-count CSV; default: Poisson arrivals at --rate")
    ap.add_argument("--rate", type=float, default=0.2, help="Poisson rate (1/s)")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    env = simpy.Environment()
    (a,) = lab_actions((args.app,))
    cfg = OpenWhiskConfig(pause_grace_s=args.grace, actions=(a,), **({} if args.stems is None else fixed_stems(args.stems)))
    inv = OpenWhisk(env, cfg, random.Random(args.seed))
    rng = random.Random(args.seed * 1000 + APPS.index(a.name) + 1)  # own stream: same arrivals whatever the policy
    env.process(feed(env, inv, a, trace_arrivals(rng, read_trace(args.trace)) if args.trace else poisson(rng, args.rate)))
    env.run(until=args.hours * 3600)

    cls = Counter(r[2] for r in inv.records)
    lat = sorted(r[3] for r in inv.records)
    n = len(lat) or 1
    print(f"requests={len(inv.arrivals)} started={len(lat)} " + " ".join(f"{k}={cls[k]}" for k in ("cold", "prewarm", "warm", "recreated")))
    print(f"latency s: mean={sum(lat) / n:.2f} p50={pct(lat, 50):.2f} p95={pct(lat, 95):.2f} p99={pct(lat, 99):.2f} "
          f"queue wait mean={sum(r[4] for r in inv.records) / n:.2f}")
    print("used RAM GB*s: " + " ".join(f"{k}={v / 1024:.0f}" for k, v in inv.used_memory_seconds().items()))
    print(f"containers: created={inv._ids} removed={inv.removed} evicted={inv.evicted} stems expired={inv.prewarm_expired}")


if __name__ == "__main__":
    main()
