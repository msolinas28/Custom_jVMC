"""Peak memory and time of one SR step (a, b) and one MinSR step (c, d), jVMC_exp vs NetKet, on 1 GPU.
Data from Scripts/compare_netket_sr.py and Scripts/compare_netket_minsr.py."""
import json

import matplotlib.pyplot as plt
import numpy as np

import plot_config as cfg

SR_DATA = cfg.DATA_DIR / "compare_netket_sr"
MINSR_DATA = cfg.DATA_DIR / "compare_netket_minsr"
SERIES = {
    "jvmc-dense": ("jVMC, full Jacobian", cfg.COLORS["jvmc_dense"], "o"),
    "jvmc-batched": ("jVMC, batched Jacobian", cfg.COLORS["jvmc_batched"], "s"),
    "netket": ("NetKet, VMC_SR", cfg.COLORS["netket"], "^"),
}

def load_points(data, x):
    # Points of every series, sorted by x
    return {
        series: sorted((json.loads(p.read_text()) for p in data.glob(f"{series}_*.json")), key=lambda p: p[x])
        for series in SERIES
    }

def plot_step(ax_mem, ax_time, points, x, step):
    measured = [p for pts in points.values() for p in pts if "times" in p]
    limit = measured[0]["bytes_limit"] / 1e9

    for series, (label, color, marker) in SERIES.items():
        ok = [p for p in points[series] if "times" in p]
        n = [p[x] for p in ok]
        ax_mem.plot(n, [p["peak_bytes"] / 1e9 for p in ok], marker=marker, color=color, label=label)
        ax_time.plot(n, [np.median(p["times"][step]) for p in ok], marker=marker, color=color, label=label)

    # Memory available to JAX, where the series that run out of memory end
    ax_mem.axhline(limit, color=cfg.COLORS["guide"])
    ax_mem.annotate("JAX memory limit", (min(p[x] for p in measured), limit), xytext=(0, -4), textcoords="offset points",
                    va="top", color=cfg.COLORS["guide"])
    oom = [p[x] for pts in points.values() for p in pts if p.get("oom")]
    if oom:
        ax_mem.plot(oom, [limit] * len(oom), "x", color="k", ms=10, mew=2, label="out of memory")

    ax_mem.set_ylabel("Peak device memory [GB]")
    ax_time.set_ylabel("Time [s]")

cfg.setup()

fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
ax_sr_mem, ax_sr_time, ax_minsr_mem, ax_minsr_time = axes

plot_step(ax_sr_mem, ax_sr_time, load_points(SR_DATA, "n_samples"), "n_samples", "sr_step")
plot_step(ax_minsr_mem, ax_minsr_time, load_points(MINSR_DATA, "n_parameters"), "n_parameters", "minsr_step")

# With more samples than the batch size, the batched Jacobian is built in batches
batch_size = json.loads((SR_DATA / "config.json").read_text())["batch_size"]
ax_sr_mem.axvline(batch_size, color=cfg.COLORS["guide"], ls="--")
ax_sr_mem.annotate("batch size of the\nbatched Jacobian", (batch_size, 0.5), xycoords=("data", "axes fraction"),
                   xytext=(-4, 0), textcoords="offset points", rotation=90, ha="right", va="center", color=cfg.COLORS["guide"])

for ax, letter, xlabel in zip(axes, "abcd", ["Number of samples"] * 2 + ["Number of parameters"] * 2):
    ax.set_title(f"({letter})", loc="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(2.0 ** np.arange(10, 21, 2))
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    cfg.plain_log_labels(ax.yaxis)

# One entry per series over the four panels
legend = {}
for ax in axes:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        legend.setdefault(label, handle)
fig.legend(legend.values(), legend.keys(), loc="upper center", ncol=len(legend), frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.91))
cfg.save(fig, "netket_comparison_sr_minsr_1gpu")
