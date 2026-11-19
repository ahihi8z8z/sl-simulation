import json
import os
from dataclasses import dataclass, field, replace

PROFILES = os.path.join(os.path.dirname(__file__), "..", "data", "profiles")  # <app>.json: measured times and RAM
APPS = ("gemma", "yolo")  # fixed order: also gives each app its own workload random stream


@dataclass(frozen=True, eq=False)  # identity hash: hashing the sample tuples dominated run time
class Action:
    name: str
    kind: str
    memory_mb: int
    pod_start_s: float  # Null -> warm container
    init_s: float       # warm container -> warm model (/init)
    run_s: float        # busy (/run)
    init_cold_s: float | None = None  # /init right after a cold pod start; None = init_s
    # measured seconds to resample from; empty = use the constant above
    pod_start_samples: tuple[float, ...] = ()
    init_samples: tuple[float, ...] = ()
    init_cold_samples: tuple[float, ...] = ()
    run_samples: tuple[float, ...] = ()
    # used RAM (working set, MB) per phase: start (starting), stem (warm container), init (/init), run (/run),
    # ready (warm model); empty = the reservation memory_mb in every phase
    ram_mb: dict = field(default_factory=dict)

    def ram(self, phase: str) -> float:
        return self.ram_mb.get(phase, self.memory_mb)


def lab_actions(apps: tuple[str, ...] = APPS, profiles: str | None = PROFILES):
    actions = tuple(a for a in (
        Action("gemma", "llama:gemma2-2b", 4096, pod_start_s=1.3, init_s=1.4, run_s=40 / 10.5),
        Action("yolo", "yolo:yolo26x", 4096, pod_start_s=1.8, init_s=2.2, run_s=0.55),
    ) if a.name in apps)
    if not profiles:
        return actions
    out = []
    for a in actions:
        with open(os.path.join(profiles, f"{a.name}.json")) as f:
            p = json.load(f)
        c = {k: tuple(v) for k, v in p.items() if k != "ram_mb"}
        out.append(replace(a, ram_mb=dict(p.get("ram_mb", {})), pod_start_s=sum(c["pod_start"]) / len(c["pod_start"]),
                           init_s=sum(c["init_prewarm"]) / len(c["init_prewarm"]),
                           init_cold_s=sum(c["init_cold"]) / len(c["init_cold"]), run_s=sum(c["run"]) / len(c["run"]),
                           pod_start_samples=c["pod_start"], init_samples=c["init_prewarm"],
                           init_cold_samples=c["init_cold"], run_samples=c["run"]))
    return tuple(out)


@dataclass(frozen=True)
class Config:
    user_memory_mb: int = 61440  # 15 pods of 4 GiB + 4 CPU: 3 per worker (16 CPU) x 5 workers
    actions: tuple[Action, ...] = field(default_factory=lab_actions)
    transition: str = "empirical"  # pod start and init times: a measured sample, or "lognormal" with their mean and variance


if __name__ == "__main__":
    c = Config()
    print(f"user_memory_mb={c.user_memory_mb}")
    for a in c.actions:
        print(f"{a.name} {a.kind} {a.memory_mb}MB pod_start={a.pod_start_s:.3f}s init={a.init_s:.3f}s "
              f"init_cold={a.init_cold_s or a.init_s:.3f}s run={a.run_s:.3f}s "
              f"samples={len(a.pod_start_samples)}/{len(a.init_samples)}/{len(a.init_cold_samples)}/{len(a.run_samples)}")
