"""Peak memory and time of one SR step, jVMC_exp vs NetKet, on 1 GPU. Data from Scripts/compare_netket_sr.py."""
import json

import matplotlib.pyplot as plt
import numpy as np

import plot_config as cfg

DATA = cfg.DATA_DIR / "compare_netket_sr"
SERIES = {
    "jvmc-dense": ("jVMC, full Jacobian", cfg.COLORS["jvmc_dense"], "o"),
    "jvmc-batched": ("jVMC, batched Jacobian", cfg.COLORS["jvmc_batched"], "s"),
    "netket": ("NetKet, VMC_SR", cfg.COLORS["netket"], "^"),
}

cfg.setup()

config = json.loads((DATA / "config.json").read_text())
# Points of every series, sorted by the number of samples
points = {
    series: sorted((json.loads(p.read_text()) for p in DATA.glob(f"{series}_*.json")), key=lambda p: p["n_samples"])
    for series in SERIES
}
measured = [p for pts in points.values() for p in pts if "times" in p]
limit = measured[0]["bytes_limit"] / 1e9
n_min = min(p["n_samples"] for p in measured)

fig, (ax_mem, ax_time) = plt.subplots(1, 2, figsize=(13, 5))

for series, (label, color, marker) in SERIES.items():
    ok = [p for p in points[series] if "times" in p]
    n = [p["n_samples"] for p in ok]
    ax_mem.plot(n, [p["peak_bytes"] / 1e9 for p in ok], marker=marker, color=color, label=label)
    ax_time.plot(n, [np.median(p["times"]["sr_step"]) for p in ok], marker=marker, color=color, label=label)

# Memory available to JAX, where the series that run out of memory end
ax_mem.axhline(limit, color=cfg.COLORS["guide"])
ax_mem.annotate("JAX memory limit", (n_min, limit), xytext=(0, -4), textcoords="offset points",
                va="top", color=cfg.COLORS["guide"])
oom = [p["n_samples"] for pts in points.values() for p in pts if p.get("oom")]
ax_mem.plot(oom, [limit] * len(oom), "x", color="k", ms=10, mew=2, label="out of memory")

# With more samples than the batch size, the batched Jacobian is built in batches
ax_mem.axvline(config["batch_size"], color=cfg.COLORS["guide"], ls="--")
ax_mem.annotate("batch size of the\nbatched Jacobian", (config["batch_size"], 0.5), xycoords=("data", "axes fraction"),
                xytext=(-4, 0), textcoords="offset points", rotation=90, ha="right", va="center", color=cfg.COLORS["guide"])

ax_mem.set_ylabel("Peak device memory [GB]")
ax_mem.set_title("(a)", loc="left")
ax_time.set_ylabel("Time [s]")
ax_time.set_title("(b)", loc="left")
for ax in (ax_mem, ax_time):
    ax.set_xscale("log", base=2)
    ax.set_xticks(2.0 ** np.arange(10, 21, 2))
    ax.set_yscale("log")
    ax.set_xlabel("Number of samples")
    cfg.plain_log_labels(ax.yaxis)

fig.legend(*ax_mem.get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92))
cfg.save(fig, "netket_comparison_sr_1gpu")
