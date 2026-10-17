import csv
import random
import sys


def poisson_onoff(rng: random.Random, rate_per_s: float, mean_on_s: float, mean_off_s: float):
    """Arrival times of a Poisson process that is switched on and off; phase lengths are exponential."""
    t = 0.0
    while True:
        end = t + rng.expovariate(1 / mean_on_s)
        while True:
            t += rng.expovariate(rate_per_s)
            if t >= end:
                break
            yield t
        t = end + rng.expovariate(1 / mean_off_s)


def read_trace(path: str) -> dict[int, float]:
    """CSV rows (minute, count); a header row and missing minutes are allowed."""
    counts = {}
    with open(path) as f:
        for row in csv.reader(f):
            try:
                counts[int(row[0])] = float(row[1])
            except ValueError:
                continue
    return counts


def trace_arrivals(rng: random.Random, counts: dict[int, float]):
    """Arrival times: inside minute m, a Poisson process with mean count[m] arrivals."""
    for m in sorted(counts):
        n = counts[m]
        if n <= 0:
            continue
        t, end = 60.0 * m, 60.0 * (m + 1)
        while True:
            t += rng.expovariate(n / 60)
            if t >= end:
                break
            yield t


def read_arrivals(path: str) -> dict[str, list[float]]:
    """CSV rows (t, action) of a fixed schedule; returns sorted times per action."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out.setdefault(row["action"], []).append(float(row["t"]))
    return {a: sorted(ts) for a, ts in out.items()}


if __name__ == "__main__":
    rng = random.Random(1)
    if len(sys.argv) > 1:
        counts = read_trace(sys.argv[1])
        times = list(trace_arrivals(rng, counts))
        got = {}
        for t in times:
            got[int(t // 60)] = got.get(int(t // 60), 0) + 1
        print("minute trace sim")
        for m in sorted(counts):
            print(f"{m:6d} {counts[m]:5g} {got.get(m, 0):3d}")
        print(f"total trace={sum(counts.values()):g} sim={len(times)}")
    else:
        gen = poisson_onoff(rng, rate_per_s=0.5, mean_on_s=60, mean_off_s=120)
        times = [next(gen) for _ in range(2000)]
        print("first:", " ".join(f"{t:.1f}" for t in times[:10]))
        print(f"mean rate={len(times) / times[-1]:.3f}/s (expected {0.5 * 60 / 180:.3f})")
