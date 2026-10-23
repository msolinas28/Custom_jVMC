"""Peak memory and time of one MinSR step, jVMC_exp vs NetKet, on 4 GPUs. Data from Scripts/compare_netket_minsr.py."""
import json

import matplotlib.pyplot as plt
import numpy as np

import plot_config as cfg

DATA = cfg.DATA_DIR / "results_minsr_chunk_and_compiled_batch_bathingon_4gpus"
SERIES = {
    "jvmc-dense": ("jVMC_exp, dense Jacobian", cfg.COLORS["jvmc_dense"], "o"),
    "jvmc-batched": ("jVMC_exp, batched Jacobian", cfg.COLORS["jvmc_batched"], "s"),
    "netket": ("NetKet, NTK", cfg.COLORS["netket"], "^"),
    "netket-onthefly": ("NetKet, NTK on the fly", cfg.COLORS["netket_onthefly"], "D"),
}

cfg.setup()

# Points of every series, sorted by the number of parameters
points = {
    series: sorted((json.loads(p.read_text()) for p in DATA.glob(f"{series}_*.json")), key=lambda p: p["n_parameters"])
    for series in SERIES
}
measured = [p for pts in points.values() for p in pts]
# Per device; the peak memory is the largest over the devices
limit = measured[0]["bytes_limit"] / 1e9
n_min = min(p["n_parameters"] for p in measured)

fig, (ax_mem, ax_time) = plt.subplots(1, 2, figsize=(13, 5.3))

for series, (label, color, marker) in SERIES.items():
    n = [p["n_parameters"] for p in points[series]]
    ax_mem.plot(n, [p["peak_bytes"] / 1e9 for p in points[series]], marker=marker, color=color, label=label)
    ax_time.plot(n, [np.median(p["times"]["minsr_step"]) for p in points[series]], marker=marker, color=color, label=label)

# Memory available to JAX
ax_mem.axhline(limit, color=cfg.COLORS["guide"])
ax_mem.annotate("JAX memory limit", (n_min, limit), xytext=(0, -4), textcoords="offset points",
                va="top", color=cfg.COLORS["guide"])

ax_mem.set_ylabel("Peak device memory [GB]")
ax_mem.set_title("(a) Peak memory of the process", loc="left")
ax_time.set_ylabel("Time [s]")
ax_time.set_title("(b) Time of one MinSR step", loc="left")
for ax in (ax_mem, ax_time):
    ax.set_xscale("log", base=2)
    ax.set_xticks(2.0 ** np.arange(10, 21, 2))
    ax.set_yscale("log")
    ax.set_xlabel("Number of parameters")
    cfg.plain_log_labels(ax.yaxis)

# Columns: jVMC_exp, NetKet
fig.legend(*ax_mem.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.88))
cfg.save(fig, "netket_comparison_minsr_4gpus")
