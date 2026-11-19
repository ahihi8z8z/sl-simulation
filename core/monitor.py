from .platform import Platform


class Monitor:
    """A platform as the testbed shows it: the pool counts, and at each decision the live numbers of the last step."""

    def __init__(self, platform: Platform, cold_window_s: float = 600.0):
        self.p, self.cold_window_s = platform, cold_window_s
        self.t = platform.env.now
        self.n_arr = self.n_rec = 0
        self.used = 0.0

    def pool(self):
        p = self.p
        return dict(free=len(p.warm_model), busy=len(p.busy), prewarmed=len(p.warm_container),
                    prewarm_starting=len(p.starting))

    def step(self):
        """What happened since the last call; `live` has the testbed controller's inputs."""
        p, now = self.p, self.p.env.now
        arrivals, started = p.arrivals[self.n_arr:], p.records[self.n_rec:]
        self.n_arr, self.n_rec = len(p.arrivals), len(p.records)
        used = p.used_memory_seconds()["total"]
        gbs_used = (used - self.used) / 1024
        pool, cap, dt = self.pool(), p.cfg.user_memory_mb, now - self.t
        cold = 0
        for r in reversed(p.records):  # in order of run start
            if r[0] + r[3] <= now - self.cold_window_s:
                break
            cold += r[2] in ("cold", "recreated") and r[0] > now - self.cold_window_s
        live = dict(req5=len(arrivals), conc5=p.peak_busy, cold10=cold,
                    warm=pool["free"] + pool["busy"], pre=pool["prewarmed"] + pool["prewarm_starting"],
                    lat_last=sum(r[3] for r in started) / len(started) if started else 0.0,
                    u_last=gbs_used * 1024 / (cap * dt) if dt else 0.0,
                    free_frac=1 - sum(c.action.memory_mb for c in p.busy + p.warm_model + p.warm_container + p.starting) / cap)
        self.t, self.used = now, used
        p.peak_busy = pool["busy"]
        return dict(live=live, arrivals=arrivals, started=started, gbs_used=gbs_used)
