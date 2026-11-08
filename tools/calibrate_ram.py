"""Used RAM (working set) per container phase from testbed runs, for owsim's used-memory metric.

Each 5 s working-set sample of a gemma pod (resources.csv: memory.current - inactive_file, the testbed's u_mem_used
input) gets the phase the simulator models at that instant:
  start = pod starting (prewarm_starting, or a cold pod before its activation starts)
  stem  = started stem cell (prewarmed)
  init  = /init of an activation (activation start .. start + initTime)
  run   = /run and the log collection after it (rest of the busy interval)
  ready = warm idle (ready)
The mean of a phase's samples is its time-averaged RAM (samples are evenly spaced), so sum(phase seconds x mean MB)
reproduces the testbed integral. `--check` replays each run's own container / activation timeline with RAM fitted on
the other experiments (leave-one-experiment-out) and compares it with the run's measured u_mem_used.
Apps without testbed runs (yolo) take the pod RAM per lifecycle state of a 2026_lifecycle OpenWhisk profile
(lifecycle_result.txt) instead: --profile <file> --app yolo. For gemma that profile reproduces the testbed's u_mem_used
within -4.8..+1.9 % over the 12 runs of trace 20606.
Usage: python tools/calibrate_ram.py [--write] [--check] <exp_dir>:<arm> ...   (exp_dir holds protocol.json, runs/)
       python tools/calibrate_ram.py --profile <lifecycle_result.txt> --app yolo [--write]"""
import argparse
import csv
import json
import os
import re
import statistics

PHASES = ("start", "stem", "init", "run", "ready")
STATE = {"prewarm_starting": "start", "prewarmed": "stem", "ready": "ready"}
CAL = os.path.join(os.path.dirname(__file__), "..", "owsim", "calibration.json")
HOLD = 5.0


def load(exp, arm):
    d = os.path.join(exp, "runs", arm)
    pr = json.load(open(os.path.join(exp, "protocol.json")))
    a = pr["warmup_min"] * 60
    b = a + pr["n_steps"] * pr["tc_s"]
    t0 = json.load(open(os.path.join(d, "run.json")))["t0_wall"]
    off = statistics.mean(json.loads(l)["offset"] for l in open(os.path.join(d, "clock.jsonl")))
    pods = {r["uid"]: r for r in csv.DictReader(open(os.path.join(d, "pods.csv"))) if "gemma" in r["name"]}
    iv = {}
    for r in csv.DictReader(open(os.path.join(d, "containers.csv"))):
        iv.setdefault(r["container_id"], []).append((r["state"], float(r["t_from"]), float(r["t_to"])))
    acts = {}
    for l in open(os.path.join(d, "activations.jsonl")):
        x = json.loads(l)
        if x.get("start") and x.get("container"):
            s = x["start"] / 1000 - t0
            acts.setdefault(x["container"], []).append((s, s + (x.get("init_ms") or 0) / 1000))
    return dict(d=d, a=a, b=b, t0=t0, off=off, pods=pods, iv=iv, acts=acts, cap_gb=pr["user_memory_mb"] / 1024)


def phase(r, name, t):
    st = next((s for s, lo, hi in r["iv"].get(name, ()) if lo <= t < hi), None)
    if st is None or st in STATE:
        return STATE.get(st)
    # busy: the activation that holds this instant (last one started at or before t)
    started = [(s, e) for s, e in r["acts"].get(name, ()) if s <= t]
    if not started:
        return "start"  # cold pod still starting
    s, e = max(started)
    return "init" if t < e else "run"


def samples(r):
    """(phase, MB) of every working-set sample inside the window."""
    out = []
    for line in open(os.path.join(r["d"], "resources.csv")):
        v = line.split()
        if len(v) != 5 or v[2] not in r["pods"]:
            continue
        t = float(v[1]) - r["off"] - r["t0"]
        if r["a"] <= t < r["b"]:
            p = phase(r, r["pods"][v[2]]["name"], t)
            if p:
                out.append((p, int(v[4]) / 2 ** 20))
    return out


def phase_seconds(r):
    """Seconds spent in each phase inside the window, from the run's container / activation timeline."""
    sec = dict.fromkeys(PHASES, 0.0)
    clip = lambda lo, hi: max(0.0, min(hi, r["b"]) - max(lo, r["a"]))
    for name, ivs in r["iv"].items():
        for st, lo, hi in ivs:
            if st in STATE:
                sec[STATE[st]] += clip(lo, hi)
                continue
            # busy interval: split at the activations' start and end of init
            cuts = [lo]
            kinds = ["start"]
            for s, e in sorted(r["acts"].get(name, ())):
                if lo <= s < hi:
                    cuts += [s, min(e, hi)]
                    kinds += ["init", "run"]
            cuts.append(hi)
            for k, x, y in zip(kinds, cuts, cuts[1:]):
                sec[k] += clip(x, y)
    return sec


def measured(r):
    p = os.path.join(r["d"], "cost_used.json")
    return json.load(open(p))["u_mem_used"] if os.path.exists(p) else None


PROFILE = {"null_to_warm_cpu": "start", "warm_cpu": "stem", "warm_cpu_to_warm_model": "init", "active": "run",
           "warm_model": "ready"}


def profile(path):
    """Phase -> mean pod RAM (MB) from a 2026_lifecycle OpenWhisk lifecycle_result.txt ("Pod RAM (MB): mean=...")."""
    out, state = {}, None
    for line in open(path):
        if m := re.match(r"── (\w+) \(", line):
            state = m.group(1)
        elif state in PROFILE and (m := re.search(r"Pod RAM\s+\(MB\):\s+mean=([\d.]+)", line)):
            out[PROFILE[state]] = float(m.group(1))
    missing = set(PHASES) - set(out)
    assert not missing, f"{path}: no Pod RAM for {missing}"
    return out


def write(app, ram):
    cal = json.load(open(CAL))
    cal[app]["ram_mb"] = {p: round(ram[p], 1) for p in PHASES}
    json.dump(cal, open(CAL, "w"))
    print(f"wrote {app} ram_mb {cal[app]['ram_mb']} to", os.path.normpath(CAL))


def fit(runs):
    pool = {}
    for r in runs:
        for p, mb in r["samples"]:
            pool.setdefault(p, []).append(mb)
    return {p: dict(mean=statistics.mean(v), median=statistics.median(v), n=len(v)) for p, v in pool.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="store the pooled means in owsim/calibration.json (gemma ram_mb)")
    ap.add_argument("--check", action="store_true", help="leave-one-experiment-out replay vs measured u_mem_used")
    ap.add_argument("--profile", help="lifecycle_result.txt of an OpenWhisk profile (instead of testbed runs)")
    ap.add_argument("--app", default="gemma")
    ap.add_argument("runs", nargs="*")
    args = ap.parse_args()
    if args.profile:
        ram = profile(args.profile)
        print({p: ram[p] for p in PHASES})
        if args.write:
            write(args.app, ram)
        raise SystemExit
    runs = []
    for spec in args.runs:
        exp, arm = spec.rsplit(":", 1)
        r = load(exp, arm)
        r.update(exp=os.path.basename(exp.rstrip("/")), arm=arm, samples=samples(r))
        runs.append(r)
    ram = fit(runs)
    print(f"{'phase':6s} {'n':>6s} {'mean MB':>8s} {'median MB':>9s}")
    for p in PHASES:
        if p in ram:
            print(f"{p:6s} {ram[p]['n']:6d} {ram[p]['mean']:8.1f} {ram[p]['median']:9.1f}")
    if args.check:
        print(f"\n{'run':28s} {'measured':>9s} {'replay':>9s} {'err':>7s}   (RAM fitted without this experiment)")
        for r in runs:
            other = fit([x for x in runs if x["exp"] != r["exp"]]) or ram
            sec = phase_seconds(r)
            u = sum(sec[p] * other[p]["mean"] for p in PHASES if p in other) / 1024 / (r["cap_gb"] * (r["b"] - r["a"]))
            m = measured(r)
            err = f"{100 * (u / m - 1):+6.1f}%" if m else "   n/a"
            print(f"{r['exp'] + ':' + r['arm']:28s} {m if m is not None else float('nan'):9.4f} {u:9.4f} {err}")
    if args.write:
        write(args.app, {p: ram[p]["mean"] for p in PHASES})
