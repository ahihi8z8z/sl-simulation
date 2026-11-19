import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# the paper's figures: file name, results column, axis label
KPIS = [("block", "block_prob", "Blocking Probability"), ("latency", "latency_s", "Startup latency (s)"),
        ("ram", "ram_gbs_per_req", "RAM Per Request (GB·s)")]
# case, label, line and marker as in the paper
CASES = [("2-state ON", "2-states ON", dict(color="tab:blue", ls="--", marker="s")),
         ("2-state OFF", "2-states OFF", dict(color="tab:orange", ls=":", marker="^")),
         ("3-state WC", "3-states WC", dict(color="tab:green", ls="-.", marker="x")),
         ("3-state WM", "3-states WM", dict(color="tab:purple", ls="--", marker="D")),
         ("4-state", "4-states", dict(color="tab:red", ls="-", marker="o", lw=3, ms=10, mfc="none", mew=2.5))]
FRONT_MARKERS = {"2-state ON": "s", "2-state OFF": "v", "3-state WC": "^", "3-state WM": "D", "4-state": "o"}


def save(fig, out, name):
    os.makedirs(os.path.dirname(os.path.join(out, name)), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def mean_rows(rows):
    """One row per (app, case, rate, theta3, theta4): the mean over seeds."""
    groups = {}
    for r in rows:
        groups.setdefault((r["app"], r["case"], r["rate"], r["theta3"], r["theta4"]), []).append(r)
    cols = [c for _, c, _ in KPIS]
    return [dict(app=k[0], case=k[1], rate=k[2], theta3=k[3], theta4=k[4],
                 **{c: sum(r[c] for r in g) / len(g) for c in cols}) for k, g in groups.items()]


def theta_sweep(rows, out, app, case, theta, rates):
    """KPIs against one backward rate, a line per arrival rate (Fig. "Backward Rate Impacts on KPIs")."""
    sub = "3" if theta == "theta3" else "4"
    for name, col, label in KPIS:
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        for rate, ls in zip(rates, ("-", "-.", ":")):
            pts = sorted((r[theta], r[col]) for r in rows if (r["app"], r["case"], r["rate"]) == (app, case, rate))
            best = min(pts, key=lambda p: p[1])[0]
            ax.plot(*zip(*pts), color="green", ls=ls, lw=4,
                    label=rf"$\lambda$ = {rate:g}, best $\theta_{sub}$ = {best:.2g}")
        ax.set_xscale("log")
        ax.set_xlabel(rf"$\theta_{sub}$ Value", fontsize=16)
        ax.set_ylabel(label, fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(True)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=14, labelspacing=0.2, handlelength=4)
        save(fig, out, f"3D-{app}-{'wc' if sub == '3' else 'wm'}/{name}")


def best_ram(rows, app):
    """Per case and arrival rate, the run with the lowest RAM per request over the swept backward rates."""
    best = {}
    for r in rows:
        if r["app"] == app:
            k = (r["case"], r["rate"])
            if k not in best or r["ram_gbs_per_req"] < best[k]["ram_gbs_per_req"]:
                best[k] = r
    return best


def rate_sweep(rows, out, app):
    """KPIs against the arrival rate at the most RAM-efficient backward rates (Fig. "Number of States ...")."""
    best = best_ram(rows, app)
    rates = sorted({k[1] for k in best})
    for name, col, label in KPIS:
        fig, ax = plt.subplots(figsize=(4.4, 4.0))
        for case, text, style in CASES:
            ax.plot(rates, [best[case, r][col] for r in rates], label=text, **{"lw": 2, "ms": 7, **style})
        ax.set_xscale("log", base=2)
        ax.set_xticks(rates)
        ax.set_xlabel(r"Arrival rate  ($\lambda$)", fontsize=16)
        ax.set_ylabel(label, fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(True)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=14,
                  labelspacing=0.2, columnspacing=1.0)
        save(fig, out, f"{app}-app/{name}")
    return best


def front(points):
    """The non-dominated (ram, latency) points, by rising RAM."""
    out = []
    for ram, lat in sorted(points):
        if not out or lat < out[-1][1]:
            out.append((ram, lat))
    return out


def pareto(rows, out, app, rate):
    """Latency-RAM Pareto front of every case at one arrival rate (Fig. "Pareto fronts")."""
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for case, text, style in CASES:
        pts = front([(r["ram_gbs_per_req"], r["latency_s"]) for r in rows
                     if (r["app"], r["case"], r["rate"]) == (app, case, rate)])
        if len(pts) > 1:
            ax.step(*zip(*pts), where="post", color=style["color"], lw=1.5)
        ax.scatter(*zip(*pts), marker=FRONT_MARKERS[case], color=style["color"], edgecolor="black", linewidth=0.5,
                   s=45, zorder=3, label=text)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel("RAM Per Request (GB·s)", fontsize=13)
    ax.set_ylabel("Startup latency (s)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10)
    save(fig, out, f"pareto/{app}")


def main():
    ap = argparse.ArgumentParser(description="The paper's evaluation figures from tools/sweep results.")
    ap.add_argument("--dir", default="sweep/results")
    args = ap.parse_args()
    with open(os.path.join(args.dir, "results.csv")) as f:
        rows = mean_rows([{k: v if k in ("app", "case") else float(v) for k, v in r.items()}
                          for r in csv.DictReader(f)])
    out = os.path.join(args.dir, "figures")
    for app in sorted({r["app"] for r in rows}):
        if app == "yolo_x":
            theta_sweep(rows, out, app, "3-state WM", "theta4", (0.5, 2.0, 8.0))  # the paper: 2, 8, 32 on 60 containers
        best = rate_sweep(rows, out, app)
        pareto(rows, out, app, 0.25)  # the paper: 1 req/s on 60 containers
        with open(os.path.join(args.dir, f"best_ram_{app}.csv"), "w", newline="") as f:  # the plotted points
            w = csv.writer(f)
            w.writerow(["case", "rate", "theta3", "theta4", "block_prob", "latency_s", "ram_gbs_per_req"])
            for (case, rate), r in sorted(best.items()):
                w.writerow([case, rate, r["theta3"], r["theta4"], r["block_prob"], r["latency_s"], r["ram_gbs_per_req"]])
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
