"""Plotting configuration shared by all paper figures.

Every plotting script starts with

    import plot_config as cfg
    cfg.setup()

and takes colors from ``cfg.COLORS`` so that the same method has the same color in every figure.
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib import ticker

PAPER_DIR = Path(__file__).resolve().parent
DATA_DIR = PAPER_DIR / "Data"
FIGURES_DIR = PAPER_DIR / "Figures"

FONTSIZE = 12

# jVMC_exp in greens, reference methods in purple
COLORS = {
    "vmc": "#30ca92",
    "vmc_light": "#5fc9a3",
    "fci": "#8d82f4",
    "jvmc_dense": "#1d8a63",
    "jvmc_batched": "#30ca92",
    "netket": "#8d82f4",
    "netket_onthefly": "#5b4cc4",
    # Reference lines and their annotations
    "guide": "#808080",
}

def setup():
    mpl.rcParams.update({
        "font.size": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        # Embed TrueType fonts in the PDF instead of Type 3, as most journals require
        "pdf.fonttype": 42,
    })

class _PlainLogFormatter(ticker.LogFormatter):
    def __call__(self, x, pos=None):
        return f"{x:g}" if super().__call__(x, pos) else ""

def plain_log_labels(axis):
    # Plain numbers (0.5 rather than 5e-01) on a log axis, also at some minor ticks when it spans up to 3 decades
    axis.set_major_formatter(_PlainLogFormatter(labelOnlyBase=False))
    axis.set_minor_formatter(_PlainLogFormatter(labelOnlyBase=False, minor_thresholds=(3, 0.5)))

def save(fig, name):
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
