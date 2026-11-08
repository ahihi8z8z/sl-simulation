"""Checks owsim's used-memory metric against the testbed, through a testbed bundle's own refsim.

For a bundle directory (protocol.json, refsim.py, verify.py, src/owsim, sim_ref/, runs/<arm>/cost_used.json):
  1. copies the bundle (without runs/) to --work with src/owsim replaced by this repo's owsim, and runs its verify.py:
     the reserved-memory / latency results must still reproduce sim_ref bit for bit;
  2. runs refsim for the given arms x 20 invoker seeds and integrates used RAM over the window
     (Invoker.used_memory_seconds between the end of the warm-up and the end of the window), as u_mem_used;
  3. compares the sim mean with the testbed run's u_mem_used (cost_used.json from testbed_common/cost.py).
Usage: python tools/check_used_sim.py <bundle_dir> <ARM>=<run_subdir> ... [--seeds 20] --work <scratch dir>"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys

OWSIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "owsim")

ap = argparse.ArgumentParser()
ap.add_argument("bundle")
ap.add_argument("arms", nargs="+", help="ARM=runs/<arm> (testbed run of that arm in the bundle)")
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--work", required=True)
args = ap.parse_args()

name = os.path.basename(os.path.abspath(args.bundle))
W = os.path.join(args.work, name)
shutil.rmtree(W, ignore_errors=True)
shutil.copytree(args.bundle, W, ignore=shutil.ignore_patterns("runs", "*.tar.gz", "__pycache__", "fig*"))
shutil.rmtree(os.path.join(W, "src", "owsim"))
shutil.copytree(OWSIM, os.path.join(W, "src", "owsim"), ignore=shutil.ignore_patterns("__pycache__"))
v = subprocess.run([sys.executable, "-I", "verify.py"], cwd=W, capture_output=True, text=True)
tail = (v.stdout + v.stderr).strip().splitlines()[-1:]
print(f"{name}: verify.py with the new owsim -> {tail[0] if tail else '?'}")

sys.path.insert(0, W)
os.chdir(W)
import refsim  # noqa: E402  (the copied bundle's refsim, importing the new owsim from src/)

PR = refsim.PR
T = PR["n_steps"] * PR["tc_s"]
SIMS = []
_init = refsim.Sim.__init__


def init(self, *a, **k):
    _init(self, *a, **k)  # ends at the end of the warm-up
    self.used0 = self.inv.used_memory_seconds()["total"]
    SIMS.append(self)


refsim.Sim.__init__ = init
ref = {}
for l in open(os.path.join(W, "sim_ref", "sim_ref.jsonl")):
    x = json.loads(l)
    ref[(x["arm"], x["inv_seed"])] = x
print(f"{'arm':6s} {'testbed':>8s} {'sim mean':>9s} {'sim sd':>7s} {'sim min-max':>15s} {'err':>7s}   reserved: testbed / sim")
for spec in args.arms:
    arm, sub = spec.split("=")
    U, R = [], []
    for seed in range(args.seeds):
        out = refsim.run(arm, seed)
        s = SIMS[-1]
        assert abs(out["lat"] - ref[(arm, seed)]["lat"]) < 1e-9, (arm, seed)  # same run as sim_ref
        U.append((s.inv.used_memory_seconds()["total"] - s.used0) / (PR["user_memory_mb"] * T))
        R.append(ref[(arm, seed)].get("u_mem", ref[(arm, seed)].get("gbs", 0) * 1024 / (PR["user_memory_mb"] * T)))
    tb = json.load(open(os.path.join(args.bundle, sub, "cost_used.json")))
    m = statistics.mean(U)
    print(f"{arm:6s} {tb['u_mem_used']:8.4f} {m:9.4f} {statistics.pstdev(U):7.4f} {min(U):7.4f}-{max(U):7.4f} "
          f"{100 * (m / tb['u_mem_used'] - 1):+6.1f}%   {tb['u_mem']:.4f} / {statistics.mean(R):.4f}")
