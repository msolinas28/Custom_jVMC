import sys
sys.path.append(sys.path[0] + "/../../..")
import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = ("jvmc_exp", "jvmc")
LABELS = {"jvmc_exp": "jVMC_exp", "jvmc": "jVMC (original)"}
N_CHECK_CONFIGS = 64

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--L", type=int, default=80, help="Number of spins")
parser.add_argument("--g", type=float, default=1.0, help="Transverse field")
parser.add_argument("--num-hidden", type=int, default=None, help="Hidden units of the RBM (default: L)")
parser.add_argument("--bias", action="store_true", help="Use the bias of the hidden units")
parser.add_argument("--n-samples", type=int, default=2**12,
                    help="Rounded up to a multiple of the number of chains, as both libraries do")
parser.add_argument("--n-chains", type=int, default=2**10)
parser.add_argument("--therm-sweeps", type=int, default=20, help="Thermalization sweeps at every sampling")
parser.add_argument("--batch-size", type=int, default=None,
                    help="Maximum number of network evaluations per call in both libraries "
                         "(default: the number of samples)")
parser.add_argument("--repeats", type=int, default=10, help="Timed repetitions of every stage")
parser.add_argument("--warmup", type=int, default=2, help="Untimed repetitions after the first call")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--libraries", nargs="+", choices=WORKERS, default=list(WORKERS))
parser.add_argument("--all-devices", action="store_true",
                    help="Run both libraries on all visible devices instead of on a single one "
                         "(the original jVMC needs jax<=0.7.2 for this, see above)")
parser.add_argument("--jvmc-path", default=None,
                    help="Folder with the package jVMC of the original library (default: it is importable)")
parser.add_argument("--jvmc-python", default=sys.executable,
                    help="Python interpreter of the original jVMC, if it lives in another environment")
parser.add_argument("--out-dir", default=None, help="Default: a new results_<date> folder next to this script")
parser.add_argument("--plot", metavar="RESULTS_DIR", default=None, help="Only plot the results in RESULTS_DIR")
parser.add_argument("--worker", choices=WORKERS, default=None, help=argparse.SUPPRESS)


def measure(stages, warmup, repeats):
    """
    ``stages`` is a list of (name, function). The stages are timed in the given order,
    as in a VMC step, so that every repetition works on the samples of the same repetition.
    """
    import jax

    def timed(fn):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        return time.perf_counter() - t0

    first_call = {name: timed(fn) for name, fn in stages}
    for _ in range(warmup):
        for _, fn in stages:
            timed(fn)

    times = {name: [] for name, _ in stages}
    for _ in range(repeats):
        for name, fn in stages:
            times[name].append(timed(fn))

    return first_call, times

def environment_info(**extra_versions):
    import jax
    import jaxlib
    import flax

    device = jax.devices()[0]
    return {
        "device": f"{device.platform.upper()}: {device.device_kind}",
        "n_devices": jax.device_count(),
        "host": platform.node(),
        "versions": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "flax": flax.__version__,
            **extra_versions,
        },
    }

def save(out_dir, name, result):
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)

def import_original(jvmc_path):
    """
    Imports the original jVMC, from ``jvmc_path`` if given, and returns its versions.
    """
    if jvmc_path:
        sys.path.insert(0, os.path.abspath(jvmc_path))
    import mpi4py
    import jVMC

    versions = {"jVMC": jVMC.__version__, "mpi4py": mpi4py.__version__}
    # Written by setup_original.sh
    commit = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(jVMC.__file__))), "ORIGINAL_COMMIT")
    if os.path.exists(commit):
        with open(commit) as f:
            versions["jVMC_commit"] = f.read().split()[0]

    return versions

def exp_hamiltonian(L, g):
    import jVMC_exp.operator.discrete as op

    hamiltonian = 0
    for i in range(L):
        hamiltonian += -1.0 * op.SigmaZ(i) * op.SigmaZ((i + 1) % L) - g * op.SigmaX(i)

    return hamiltonian

def original_hamiltonian(L, g):
    import jVMC.operator as op

    hamiltonian = op.BranchFreeOperator()
    for i in range(L):
        hamiltonian.add(op.scal_opstr(-1.0, (op.Sz(i), op.Sz((i + 1) % L))))
        hamiltonian.add(op.scal_opstr(-g, (op.Sx(i),)))

    return hamiltonian

def original_n_conn(hamiltonian):
    """
    Connected configurations per sample of an operator of the original jVMC, i.e. network
    evaluations per sample of its local estimator: one per non-diagonal operator string, and
    one for all the diagonal ones together (as in ``BranchFreeOperator.compile``).
    """
    diagonal = [all(o["diag"] for o in string if not callable(o)) for string in hamiltonian.ops]
    return diagonal.count(False) + any(diagonal)

def child_env(all_devices=False):
    """
    Environment of every worker, the same for both libraries.
    """
    env = dict(os.environ)
    # One process per library
    env["JVMC_USE_DISTRIBUTED"] = "false"
    # The original jVMC initializes MPI when it is imported. Without the variables srun sets
    # for its own task, every worker is an MPI process of its own (singleton)
    for name in list(env):
        if name.startswith(("PMI_", "PMIX_", "__PMI")):
            del env[name]
    if not all_devices:
        # First of the devices given to this job, or device 0
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    return env


# ----------------------------------------------------------------------------------------
# jVMC_exp
# ----------------------------------------------------------------------------------------

def run_jvmc_exp(cfg, out_dir):
    from flax.traverse_util import flatten_dict
    import jVMC_exp
    from jVMC_exp.nets import CpxRBM
    from jVMC_exp.sampler import MCSampler
    from jVMC_exp.sampler.propose import SpinFlip
    from jVMC_exp.vqs import NQS

    L = cfg["L"]
    hamiltonian = exp_hamiltonian(L, cfg["g"])

    batch_size = cfg["batch_size"]
    net = CpxRBM(numHidden=cfg["num_hidden"], bias=cfg["bias"])
    psi = NQS(net, L, batchSize=batch_size, seed=cfg["seed"])
    sampler = MCSampler(
        psi, SpinFlip(), key=cfg["seed"],
        numChains=cfg["n_chains"], numSamples=cfg["n_samples"],
        thermalizationSweeps=cfg["therm_sweeps"], sweepSteps=L
    )

    def estimate():
        # Reuses the stored samples and their log(psi)
        energy = sampler(hamiltonian)
        return energy.mean, energy.error_of_mean

    first_call, times = measure([
        ("sampling", sampler.sample),
        ("matrix_elements", lambda: hamiltonian.get_conn_elements(sampler.samples, batch_size=batch_size)),
        ("local_estimator", estimate),
    ], cfg["warmup"], cfg["repeats"])

    mean, error = estimate()

    # Parameters and reference values of log(psi), to run the same network in the original jVMC
    configs = np.asarray(sampler.samples[:N_CHECK_CONFIGS])
    np.savez(
        os.path.join(out_dir, "reference.npz"),
        configs=configs, log_psi=np.asarray(psi(configs)),
        **{"params/" + "/".join(k): np.asarray(v) for k, v in flatten_dict(psi.params).items()},
    )

    save(out_dir, "jvmc_exp", {
        "library": "jvmc_exp",
        "label": LABELS["jvmc_exp"],
        **environment_info(jVMC_exp=jVMC_exp.__version__),
        "n_parameters": psi.numParameters if psi.realParams else 2 * psi.numParameters,
        "batch_size": batch_size,
        "sample_devices": len(sampler.samples.sharding.device_set),
        "operator": "generic operator strings (SigmaZ, SigmaX)",
        # The diagonal uses log(psi) of the samples from the sampling
        "evaluations_per_sample": hamiltonian.n_conn,
        "first_call": first_call,
        "times": times,
        "energy": [float(np.real(np.asarray(mean).ravel()[0])), float(np.asarray(error).ravel()[0])],
        "acceptance": float(np.asarray(sampler.acceptance_ratio()).ravel()[0]),
    })


# ----------------------------------------------------------------------------------------
# Original jVMC
# ----------------------------------------------------------------------------------------

def run_jvmc(cfg, out_dir):
    original_versions = import_original(cfg["jvmc_path"])
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import unflatten_dict
    from jVMC.nets import CpxRBM
    from jVMC.sampler import MCSampler, propose_spin_flip
    from jVMC.stats import SampledObs
    from jVMC.vqs import NQS

    L = cfg["L"]
    n_devices = jax.device_count()
    batch_size = cfg["batch_size"] // n_devices
    hamiltonian = original_hamiltonian(L, cfg["g"])
    n_conn = original_n_conn(hamiltonian)
    hamiltonian.ElocBatchSize = max(1, batch_size // n_conn)

    psi = NQS(CpxRBM(numHidden=cfg["num_hidden"], bias=cfg["bias"]), batchSize=batch_size, seed=cfg["seed"])
    sampler = MCSampler(
        psi, (L,), cfg["seed"], updateProposer=propose_spin_flip,
        numChains=cfg["n_chains"] // n_devices, numSamples=cfg["n_samples"],
        thermalizationSweeps=cfg["therm_sweeps"], sweepSteps=L,
    )

    log_psi_error = None
    reference = os.path.join(out_dir, "reference.npz")
    if os.path.exists(reference):
        ref = np.load(reference)
        psi.params = unflatten_dict({
            tuple(k.split("/")[1:]): jnp.asarray(ref[k]) for k in ref.files if k.startswith("params/")
        })
        log_psi = psi(jnp.asarray(ref["configs"]).reshape(n_devices, -1, L))
        log_psi_error = float(np.max(np.abs(np.asarray(log_psi).ravel() - ref["log_psi"])))
        if not log_psi_error < 1e-8:
            raise RuntimeError(f"The networks of the two libraries differ: max |delta log(psi)| = {log_psi_error}")
    else:
        print("WARNING: no jVMC_exp parameters found, the original jVMC runs with its own random parameters")

    state = {}

    def sample():
        state["configs"], state["log_psi"], state["p"] = sampler.sample()
        return state["configs"], state["log_psi"], state["p"]

    def matrix_elements():
        configs = state["configs"]
        if configs.shape[1] <= batch_size:
            return hamiltonian.get_s_primes(configs)
        return [hamiltonian.get_s_primes(configs[:, i:i + batch_size]) for i in range(0, configs.shape[1], batch_size)]

    def estimate():
        # Reuses the stored samples and their log(psi)
        energy = SampledObs(hamiltonian.get_O_loc(state["configs"], psi, state["log_psi"]), state["p"])
        return energy.mean(), energy.var()

    first_call, times = measure([
        ("sampling", sample),
        ("matrix_elements", matrix_elements),
        ("local_estimator", estimate),
    ], cfg["warmup"], cfg["repeats"])

    n_samples = state["configs"].shape[0] * state["configs"].shape[1]
    if n_samples != cfg["n_samples"]:
        raise RuntimeError(f"The original jVMC drew {n_samples} samples instead of {cfg['n_samples']}")
    mean, var = estimate()

    save(out_dir, "jvmc", {
        "library": "jvmc",
        "label": LABELS["jvmc"],
        **environment_info(**original_versions),
        "n_parameters": int(psi.numParameters) if psi.realParams else 2 * int(psi.numParameters),
        "batch_size": cfg["batch_size"],
        "eloc_batch_size": hamiltonian.ElocBatchSize * n_devices,
        "sample_devices": len(state["configs"].sharding.device_set),
        "max_log_psi_difference": log_psi_error,
        "operator": "BranchFreeOperator (Sz, Sx)",
        # All connected configurations, diagonal included
        "evaluations_per_sample": n_conn,
        "first_call": first_call,
        "times": times,
        "energy": [float(np.real(np.asarray(mean).ravel()[0])), float(np.sqrt(np.asarray(var).ravel()[0] / n_samples))],
        "acceptance": float(np.asarray(sampler.acceptance_ratio()).ravel()[0]),
    })


# ----------------------------------------------------------------------------------------
# Figure and summary
# ----------------------------------------------------------------------------------------

STAGE_LABELS = ("Sampling", "Matrix elements", "Evaluation", "Total")

# Validated categorical slots (light mode) and chart ink, see the dataviz reference palette
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

PLOT_STYLE = {
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": BASELINE,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_MUTED,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
}

def load_series(out_dir):
    """
    Per-repetition times of every stage, for every library.
    """
    series = []
    for worker in WORKERS:
        path = os.path.join(out_dir, f"{worker}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            result = json.load(f)
        t = {k: np.asarray(v) for k, v in result["times"].items()}
        series.append({
            **result,
            # All stages are measured within the same repetition
            "stages": dict(zip(STAGE_LABELS, (
                t["sampling"],
                t["matrix_elements"],
                t["local_estimator"] - t["matrix_elements"],
                t["sampling"] + t["local_estimator"],
            ))),
        })

    return series

def print_summary(cfg, series):
    width = max(len(s["label"]) for s in series) + 2
    exp_total = next((np.median(s["stages"]["Total"]) for s in series if s["library"] == "jvmc_exp"), np.nan)
    print(f"\nMedian over {cfg['repeats']} repetitions [s]")
    print(" " * width + "".join(f"{label:>17}" for label in STAGE_LABELS)
          + f"{'Total/jVMC_exp':>17}{'first calls':>17}")
    for s in series:
        total = np.median(s["stages"]["Total"])
        print(f"{s['label']:<{width}}" + "".join(f"{np.median(s['stages'][label]):>17.4f}" for label in STAGE_LABELS)
              + f"{total / exp_total:>17.2f}{sum(s['first_call'].values()):>17.2f}")

    print("\nChecks")
    print("  samples on " + ", ".join(f"{s['sample_devices']} device(s) in {s['label']}" for s in series))
    original = next((s for s in series if s["library"] == "jvmc"), None)
    if original is not None and original["max_log_psi_difference"] is not None:
        print(f"  same network: max |delta log(psi)| on {N_CHECK_CONFIGS} configurations = "
              f"{original['max_log_psi_difference']:.1e}")
    reference = series[0]
    for s in series:
        line = (f"  {s['label']:<{width}} E/N = {s['energy'][0] / cfg['L']:+.6f} +- {s['energy'][1] / cfg['L']:.6f}"
                f"   acceptance = {s['acceptance']:.3f}   network evaluations per sample in <H>: "
                f"{s['evaluations_per_sample']}")
        if s is not reference:
            z = abs(s["energy"][0] - reference["energy"][0]) / np.hypot(s["energy"][1], reference["energy"][1])
            line += f"   |delta E| = {z:.1f} sigma"
        print(line)

def plot(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    series = load_series(out_dir)
    if not series:
        raise SystemExit(f"No results in {out_dir}")
    print_summary(cfg, series)

    plt.rcParams.update(PLOT_STYLE)
    exp = next((s for s in series if s["library"] == "jvmc_exp"), None)
    colors = [SERIES_COLORS[WORKERS.index(s["library"])] for s in series]
    x = np.arange(len(series))

    fig, axes = plt.subplots(1, len(STAGE_LABELS), figsize=(13, 4.6))
    for ax, stage in zip(axes, STAGE_LABELS):
        q25, median, q75 = (np.array([np.percentile(s["stages"][stage], q) for s in series]) for q in (25, 50, 75))
        unit, scale = ("ms", 1e3) if np.max(np.abs(q75)) < 1 else ("s", 1)
        bars = ax.bar(
            x, median * scale, 0.75, color=colors, edgecolor=SURFACE, linewidth=1.5,
            yerr=[(median - q25) * scale, (q75 - median) * scale], error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1},
        )
        for i, s in enumerate(series):
            label = f"{median[i] * scale:.3g}"
            if stage == "Total" and exp is not None and s is not exp:
                label += f"\n({median[i] / np.median(exp['stages'][stage]):.2f}x)"
            ax.annotate(label, (x[i], max(q75[i], 0) * scale), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)

        ax.set_title(stage, loc="left", fontsize=11)
        ax.set_ylabel(f"Time [{unit}]")
        ax.set_xticks([])
        ax.margins(y=0.2)
        ax.axhline(0, color=BASELINE, lw=1)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.suptitle("Monte Carlo estimate of <H>: time per stage", x=0.01, ha="left", fontsize=12)
    fig.legend(bars, [s["label"] for s in series], loc="upper left", bbox_to_anchor=(0.01, 0.94),
               ncol=len(series), frameon=False, fontsize=9, handlelength=1, columnspacing=1.5)

    info = series[0]
    versions = " | ".join(sorted({
        f"{name} {s['versions'][name]}" for s in series for name in ("jVMC_exp", "jVMC", "jax") if name in s["versions"]
    }))
    ratio_note = " (r x): jVMC time / jVMC_exp time." if exp is not None and len(series) > 1 else ""
    fig.text(
        0.01, 0.01,
        f"Transverse-field Ising chain, N = {cfg['L']}, g = {cfg['g']}. Complex RBM, {cfg['num_hidden']} hidden units"
        f"{' with bias' if cfg['bias'] else ''}, {info['n_parameters']} real parameters, complex128. "
        f"{cfg['n_samples']} samples, {cfg['n_chains']} chains, sweeps of {cfg['L']} spin flips, "
        f"{cfg['therm_sweeps']} thermalization sweeps.\n"
        f"At most {cfg['batch_size']} network evaluations per call. Evaluation = full estimate of <H> minus matrix elements. "
        f"Total = sampling + full estimate of <H>.{ratio_note}\n"
        f"Median of {cfg['repeats']} repetitions after compilation and {cfg['warmup']} warm-up runs, error bars: "
        f"interquartile range. {info['n_devices']} x {info['device']}. {versions}",
        fontsize=7.5, color=INK_SECONDARY, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"jvmc_comparison.{ext}"), dpi=200)
    print(f"\nFigure saved to {os.path.join(out_dir, 'jvmc_comparison.png')}")


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------

def main():
    args = parser.parse_args()

    if args.worker is not None:
        with open(os.path.join(args.out_dir, "config.json")) as f:
            cfg = json.load(f)
        {"jvmc_exp": run_jvmc_exp, "jvmc": run_jvmc}[args.worker](cfg, args.out_dir)
        return

    if args.plot is not None:
        plot(args.plot)
        return

    n_samples = -(-args.n_samples // args.n_chains) * args.n_chains
    if n_samples != args.n_samples:
        print(f"Number of samples rounded up to a multiple of the number of chains: {args.n_samples} -> {n_samples}")
    cfg = {
        "L": args.L, "g": args.g, "num_hidden": args.num_hidden or args.L, "bias": args.bias,
        "n_samples": n_samples, "n_chains": args.n_chains, "therm_sweeps": args.therm_sweeps,
        "batch_size": args.batch_size or n_samples,
        "repeats": args.repeats, "warmup": args.warmup, "seed": args.seed,
        "all_devices": args.all_devices, "jvmc_path": args.jvmc_path,
        "date": datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = args.out_dir or os.path.join(HERE, f"results_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    for name in ("jvmc_exp.json", "jvmc.json", "reference.npz"):
        if os.path.exists(os.path.join(out_dir, name)):
            os.remove(os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for worker in [w for w in WORKERS if w in args.libraries]:
        python = args.jvmc_python if worker == "jvmc" else sys.executable
        print(f"=== {LABELS[worker]} ===", flush=True)
        run = subprocess.run(
            [python, "-u", os.path.abspath(__file__), "--worker", worker, "--out-dir", out_dir],
            env=child_env(args.all_devices),
        )
        if run.returncode != 0:
            print(f"    FAILED ({worker}) -- continuing")

    plot(out_dir)

if __name__ == "__main__":
    main()
