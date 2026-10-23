"""Time of sampling and expect <H>, jVMC_exp vs NetKet, on 1 GPU. Data from Scripts/compare_netket.py."""
import json

import matplotlib.pyplot as plt
import numpy as np

import plot_config as cfg

DATA = cfg.DATA_DIR / "compare_netket"
# Variants of the results that are plotted, in this order
SERIES = {
    "jvmc": ("jVMC", cfg.COLORS["jvmc_dense"]),
    "ising": ("NetKet, Ising (JAX)", cfg.COLORS["netket"]),
    "local": ("NetKet, LocalOperator (JAX)", cfg.COLORS["netket_onthefly"]),
}
STAGES = {
    "sampling": "(a) Sampling",
    "expect": r"(b) Expect $\langle H \rangle$",
}

cfg.setup()

variants = {}
for library in ("jvmc", "netket"):
    path = DATA / f"{library}.json"
    if path.exists():
        variants |= json.loads(path.read_text())["variants"]
series = [s for s in SERIES if s in variants]

# One panel per stage, each with its own scale
fig, axes = plt.subplots(1, len(STAGES), figsize=(13, 5))
for ax, (stage, title) in zip(axes, STAGES.items()):
    times = [np.asarray(variants[s]["times"][stage]) for s in series]
    q25, median, q75 = (np.array([np.percentile(t, q) for t in times]) for q in (25, 50, 75))
    unit, scale = ("ms", 1e3) if q75.max() < 1 else ("s", 1)

    for i, s in enumerate(series):
        label, color = SERIES[s]
        # Median and interquartile range of the repetitions
        ax.bar(i, median[i] * scale, 0.7, color=color, label=label,
               yerr=[[(median[i] - q25[i]) * scale], [(q75[i] - median[i]) * scale]], ecolor="k", capsize=4)
        text = f"{median[i] * scale:.3g}"
        if "jvmc" in series and s != "jvmc":
            text += f"\n({median[i] / median[series.index('jvmc')]:.2f}x)"
        ax.annotate(text, (i, q75[i] * scale), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")

    ax.set_title(title, loc="left")
    ax.set_ylabel(f"Time [{unit}]")
    ax.set_xticks([])
    ax.margins(y=0.25)

fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(series), frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92))
cfg.save(fig, "netket_comparison")
