"""VMC optimization of the 2D quantum dot compared with FCI. Data from Scripts/quantum_dot_2d.py."""
import h5py
import matplotlib.pyplot as plt
import numpy as np

import plot_config as cfg

cfg.setup()

with h5py.File(cfg.DATA_DIR / "quantum_dot_2d.h5", "r") as f:
    steps = f["vmc/steps"][()] + 1
    energy = f["vmc/energy"][()]
    variance = f["vmc/variance"][()]
    E_c_fci = f["fci/E_c"][()]
    E_fci = f["fci/energy"][()]

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(steps, energy, label="VMC", color=cfg.COLORS["vmc"])
for E_c, E, alpha in zip(E_c_fci, E_fci, np.linspace(0.3, 1, len(E_fci))):
    ax.axhline(E, color=cfg.COLORS["fci"], ls="--", alpha=alpha, label=f"FCI, $E_c = {E_c}$")
ax.set_xlabel("Steps")
ax.set_ylabel("Energy")
ax.set_xlim(1, steps[-1])
ax.legend(loc="upper left", bbox_to_anchor=(0.1, 1))

# Inset position as [x0, y0, width, height], in fractions of the main axes
inset = ax.inset_axes([0.48, 0.47, 0.5, 0.5])
inset.plot(steps, variance, color=cfg.COLORS["vmc_light"])
inset.set_yscale("log")
inset.set_xlabel("Steps")
inset.set_ylabel("Energy variance")
inset.set_xlim(1, steps[-1])

fig.tight_layout()
cfg.save(fig, "quantum_dot_2d")
