import json
import os
from dataclasses import dataclass, field, replace

CALIBRATION = os.path.join(os.path.dirname(__file__), "calibration.json")
APPS = ("gemma", "yolo")  # fixed order: also gives each app its own workload random stream


@dataclass(frozen=True)
class Reactive:
    min_count: int = 0
    max_count: int = 1
    ttl_s: float = 600.0
    threshold: int = 1
    increment: int = 1


@dataclass(frozen=True, eq=False)  # identity hash: hashing the sample tuples dominated run time
class Action:
    name: str
    kind: str
    memory_mb: int
    pod_start_s: float  # Null -> Warm CPU (stem cell ready)
    init_s: float       # Warm CPU -> Warm Model (/init) on a stem cell
    run_s: float        # Active (/run)
    stem_cells: int = 1  # initialCount
    reactive: Reactive = Reactive()
    init_cold_s: float | None = None  # /init right after a cold pod start; None = init_s
    # measured seconds to resample from; empty = use the constant above
    pod_start_samples: tuple[float, ...] = ()
    init_samples: tuple[float, ...] = ()
    init_cold_samples: tuple[float, ...] = ()
    run_samples: tuple[float, ...] = ()


def lab_actions(apps: tuple[str, ...] = APPS, calibration: str | None = CALIBRATION):
    actions = tuple(a for a in (
        Action("gemma", "llama:gemma2-2b", 4096, pod_start_s=1.3, init_s=1.4, run_s=40 / 10.5),
        Action("yolo", "yolo:yolo26x", 4096, pod_start_s=1.8, init_s=2.2, run_s=0.55),
    ) if a.name in apps)
    if not calibration:
        return actions
    with open(calibration) as f:
        cal = json.load(f)
    out = []
    for a in actions:
        c = {k: tuple(v) for k, v in cal[a.name].items()}
        out.append(replace(a, pod_start_s=sum(c["pod_start"]) / len(c["pod_start"]),
                           init_s=sum(c["init_prewarm"]) / len(c["init_prewarm"]),
                           init_cold_s=sum(c["init_cold"]) / len(c["init_cold"]), run_s=sum(c["run"]) / len(c["run"]),
                           pod_start_samples=c["pod_start"], init_samples=c["init_prewarm"],
                           init_cold_samples=c["init_cold"], run_samples=c["run"]))
    return tuple(out)


@dataclass(frozen=True)
class Config:
    user_memory_mb: int = 61440  # 15 pods of 4 GiB + 4 CPU: 3 per worker (16 CPU) x 5 workers
    pause_grace_s: float = 10.0  # whisk.container-proxy.timeouts.pause-grace
    log_collect_s: float = 2.0  # after a run, before Ready: KubernetesContainer waits waitForLogs for the sentinel
    prewarm_ttl_s: float | None = None  # None = each stem cell's reactive ttl
    prewarm_min_count: int | None = None  # None = each stem cell's reactive minCount
    prewarm_max_count: int | None = None  # None = each stem cell's reactive maxCount
    prewarm_check_init_s: float = 1.0
    prewarm_check_s: float = 1.0
    prewarm_check_variance_s: float = 1.0
    actions: tuple[Action, ...] = field(default_factory=lab_actions)


if __name__ == "__main__":
    c = Config()
    print(f"user_memory_mb={c.user_memory_mb} pause_grace_s={c.pause_grace_s} "
          f"prewarm_ttl_s={c.prewarm_ttl_s} prewarm_check_s={c.prewarm_check_s}")
    for a in c.actions:
        print(f"{a.name} {a.kind} {a.memory_mb}MB stem_cells={a.stem_cells} {a.reactive}")
        print(f"  pod_start={a.pod_start_s:.3f}s init={a.init_s:.3f}s init_cold={a.init_cold_s or a.init_s:.3f}s run={a.run_s:.3f}s "
              f"samples={len(a.pod_start_samples)}/{len(a.init_samples)}/{len(a.init_cold_samples)}/{len(a.run_samples)}")
