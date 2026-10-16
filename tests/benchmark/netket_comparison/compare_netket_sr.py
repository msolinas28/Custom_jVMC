import sys
sys.path.append(sys.path[0] + "/../../..")
import argparse
import json
import os
import subprocess
import warnings
from datetime import datetime

import numpy as np

from compare_netket import (
    measure, environment_info, save, netket_rbm, netket_operator, jvmc_hamiltonian,
    SERIES_COLORS, SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID, BASELINE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES = {
    "jvmc-dense": "jVMC_exp, dense Jacobian",
    "jvmc-batched": "jVMC_exp, batched Jacobian",
    "netket": "NetKet, VMC_SR",
}
# NetKet's VMC_SR solves (S + eps) x = 2 F, jVMC_exp's SR solves (S + eps) x = -F
NETKET_UPDATE_FACTOR = -2

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--L", type=int, default=80, help="Number of spins")
parser.add_argument("--g", type=float, default=1.0, help="Transverse field")
parser.add_argument("--num-hidden", type=int, default=None, help="Hidden units of the RBM (default: L)")
parser.add_argument("--bias", action="store_true", help="Use the bias of the hidden units")
parser.add_argument("--n-samples", type=int, nargs="+", default=[2**k for k in range(10, 21)])
parser.add_argument("--batch-size", type=int, default=2**14,
                    help="Network evaluations per call for the local energies, samples per block of the Jacobian")
parser.add_argument("--diag-shift", type=float, default=1e-3)
parser.add_argument("--pinv-cutoff", type=float, default=1e-14, help="Relative eigenvalue cutoff of the solver")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--series", nargs="+", choices=list(SERIES), default=list(SERIES))
parser.add_argument("--netket-operator", choices=["ising", "local"], default="ising",
                    help="NetKet operator of the local energies: Ising (JAX) or the generic LocalOperator (JAX)")
parser.add_argument("--netket-python", default=sys.executable,
                    help="Python interpreter of the NetKet runs, if NetKet lives in another environment")
parser.add_argument("--all-devices", action="store_true",
                    help="Run on all visible devices instead of the first one (default: one device)")
parser.add_argument("--out-dir", default=None, help="Default: a new results_sr_<date> folder next to this script")
parser.add_argument("--plot", metavar="RESULTS_DIR", default=None, help="Only plot the results in RESULTS_DIR")
parser.add_argument("--worker", choices=list(SERIES), default=None, help=argparse.SUPPRESS)
parser.add_argument("--n", type=int, default=None, help=argparse.SUPPRESS)


# ----------------------------------------------------------------------------------------
# Shared by the workers
# ----------------------------------------------------------------------------------------

def load_inputs(cfg, n_samples, out_dir):
    """
    The parameters written by the driver, and the samples: 0/1 configurations, the same in
    every process.
    """
    from flax.traverse_util import unflatten_dict

    with np.load(os.path.join(out_dir, "params.npz")) as f:
        params = unflatten_dict({tuple(k.split("/")): f[k] for k in f.files})
    samples = np.random.default_rng(cfg["seed"]).integers(0, 2, (n_samples, cfg["L"]), dtype=np.int8)

    return params, samples

def check_single_device(cfg):
    """
    Number of devices, which has to be one unless the comparison runs on all devices.
    """
    import jax

    if not cfg.get("all_devices") and jax.device_count() != 1:
        raise SystemExit(f"{jax.device_count()} devices are visible, but the comparison runs on one (see --all-devices)")

    return jax.device_count()

def peak_memory():
    import resource
    import jax

    stats = [d.memory_stats() for d in jax.local_devices()]
    if stats[0] is not None:
        # The largest peak over the devices, and the peak of every device
        peaks = [s.get("peak_bytes_in_use") for s in stats]
        return {"peak_bytes": max(peaks), "peak_bytes_per_device": peaks, "bytes_limit": stats[0].get("bytes_limit"), "memory": "device"}

    # No device statistics on CPUs: peak resident memory of the process (bytes on macOS, kB on Linux)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_bytes": rss if sys.platform == "darwin" else 1024 * rss, "bytes_limit": None, "memory": "process"}

def is_out_of_memory(error):
    return "RESOURCE_EXHAUSTED" in str(error) or "out of memory" in str(error).lower()

def point_name(series, n_samples):
    return f"{series}_{n_samples}"


# ----------------------------------------------------------------------------------------
# jVMC_exp
# ----------------------------------------------------------------------------------------

def run_jvmc(cfg, n_samples, out_dir, batched):
    import jax
    import jax.numpy as jnp
    import jVMC_exp
    from jVMC_exp.nets import RBM
    from jVMC_exp.objective_function import Observable
    from jVMC_exp.optimizer import SR
    from jVMC_exp.sampler import MCSampler
    from jVMC_exp.sampler.propose import SpinFlip
    from jVMC_exp.solver import Pinv
    from jVMC_exp.vqs import NQS

    check_single_device(cfg)
    L = cfg["L"]
    params, samples = load_inputs(cfg, n_samples, out_dir)
    samples = jnp.asarray(samples, dtype=jVMC_exp.global_defs.DT_SAMPLES)
    weights = jnp.full(n_samples, 1 / n_samples)
    batch_size = min(cfg["batch_size"] or n_samples, n_samples)

    hamiltonian = jvmc_hamiltonian(L, cfg["g"])
    psi = NQS(RBM(numHidden=cfg["num_hidden"], bias=cfg["bias"]), L, batchSize=batch_size, seed=cfg["seed"])
    psi.parameters = jax.tree_util.tree_map(jnp.asarray, params)
    # Never samples: the samples are set below, where SR sets the ones it draws
    sampler = MCSampler(psi, SpinFlip(), key=cfg["seed"], numChains=jax.device_count(), numSamples=n_samples)
    objective = Observable(hamiltonian, batched_jacobian=batched)
    with warnings.catch_warnings():
        # Notice about the naming of diagonalShift and diagonalScale
        warnings.simplefilter("ignore", UserWarning)
        sr = SR(sampler, psi, diagonalShift=cfg["diag_shift"], solver=Pinv(pinv_cutoff=cfg["pinv_cutoff"]))

    def set_samples():
        # jVMC_exp evaluates log(psi) of the samples when sampling, NetKet in the SR step
        sampler._samples, sampler._logPsi, sampler._weights = samples, psi(samples), weights

    def local_energies():
        set_samples()
        return sampler(hamiltonian).mean

    last = {}

    def sr_step():
        set_samples()
        last["update"] = sr.get_update(objective.value_and_grad(sampler, compute_grad=True))
        return last["update"]

    first_call, times = measure([("local_energies", local_energies), ("sr_step", sr_step)], cfg["warmup"], cfg["repeats"])
    np.save(os.path.join(out_dir, f"update_{point_name('jvmc-batched' if batched else 'jvmc-dense', n_samples)}.npy"),
            np.asarray(last["update"]))

    return {
        **environment_info(jVMC_exp=jVMC_exp.__version__),
        "n_parameters": psi.numParameters, "first_call": first_call, "times": times, **peak_memory(),
    }


# ----------------------------------------------------------------------------------------
# NetKet
# ----------------------------------------------------------------------------------------

def run_netket(cfg, n_samples, out_dir):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    import netket as nk
    import optax

    check_single_device(cfg)
    L = cfg["L"]
    params, samples = load_inputs(cfg, n_samples, out_dir)

    batch_size = cfg['batch_size']
    if (batch_size is not None) and (batch_size >= n_samples):
        batch_size = None

    hamiltonian = netket_operator(cfg["netket_operator"], L, cfg["g"])
    n_devices = check_single_device(cfg)
    if n_devices > 1:
        # NetKet shards the samples over the mesh, if one is set
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("S",), axis_types=(jax.sharding.AxisType.Auto,))
        jax.sharding.set_mesh(mesh)
    sampler = nk.sampler.MetropolisLocal(hamiltonian.hilbert, n_chains=n_devices)
    with warnings.catch_warnings():
        # The batch size is the same as in jVMC_exp, power of two or not
        warnings.filterwarnings("ignore", message=".*power-of-two chunk size")
        vstate = nk.vqs.MCState(
            sampler, netket_rbm(cfg["num_hidden"], cfg["bias"], "float64"),
            n_samples=n_samples, chunk_size=batch_size, seed=cfg["seed"],
        )
    vstate.parameters = jax.tree_util.tree_map(jnp.asarray, params)

    def eigh_pinv(A, b):
        # As jVMC_exp.solver.Pinv
        ev, V = jnp.linalg.eigh(A)
        inv_ev = jnp.where(jnp.abs(ev / ev[-1]) > cfg["pinv_cutoff"], 1. / ev, 0.)
        return V @ (inv_ev * (V.conj().T @ b)), None

    driver = nk.driver.VMC_SR(
        hamiltonian, optax.sgd(1e-2), variational_state=vstate,
        diag_shift=cfg["diag_shift"], linear_solver=eigh_pinv,
        mode="real", use_ntk=False, on_the_fly=False,
    )
    # NetKet has no public way to set the samples: these are the ones MCState.sample() would
    # store, with the 0/1 spins of jVMC_exp as +-1
    vstate._samples = jnp.asarray(2 * samples - 1, dtype=sampler.dtype).reshape(n_devices, -1, L)
    if n_devices > 1:
        vstate._samples = jax.device_put(vstate._samples, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("S")))

    last = {}

    def sr_step():
        last["update"] = driver.compute_loss_and_update()[1]
        return last["update"]

    first_call, times = measure(
        [("local_energies", lambda: vstate.local_estimators(hamiltonian)), ("sr_step", sr_step)],
        cfg["warmup"], cfg["repeats"]
    )
    np.save(os.path.join(out_dir, f"update_{point_name('netket', n_samples)}.npy"),
            np.asarray(ravel_pytree(last["update"])[0]))

    return {
        **environment_info(netket=nk.__version__),
        "n_parameters": vstate.n_parameters, "first_call": first_call, "times": times, **peak_memory(),
    }


# ----------------------------------------------------------------------------------------
# Figure and summary
# ----------------------------------------------------------------------------------------

def load_points(out_dir):
    """
    For every series, its points sorted by the number of samples.
    """
    points = {}
    for series in SERIES:
        found = []
        for name in os.listdir(out_dir):
            if name.startswith(series + "_") and name.endswith(".json") and name[len(series) + 1:-5].isdigit():
                with open(os.path.join(out_dir, name)) as f:
                    found.append(json.load(f))
        if found:
            points[series] = sorted(found, key=lambda p: p["n_samples"])

    return points

def update_differences(out_dir, points):
    """
    Relative difference between the update of NetKet and the one of jVMC_exp, for every
    number of samples where both are available.
    """
    differences = {}
    for series in ("jvmc-dense", "jvmc-batched"):
        for p in points.get(series, []):
            files = [os.path.join(out_dir, f"update_{point_name(s, p['n_samples'])}.npy") for s in (series, "netket")]
            if all(os.path.exists(f) for f in files):
                jvmc, netket = (np.load(f) for f in files)
                differences[(series, p["n_samples"])] = (
                    np.linalg.norm(netket - NETKET_UPDATE_FACTOR * jvmc) / np.linalg.norm(netket)
                )

    return differences

def print_summary(cfg, points, differences):
    n_all = sorted({p["n_samples"] for pts in points.values() for p in pts})
    print("\nSR step: median time [s] / peak device memory [GB]")
    print(f"{'samples':>10}" + "".join(f"{SERIES[s]:>32}" for s in points))
    for n in n_all:
        row = f"{n:>10}"
        for series, pts in points.items():
            p = next((p for p in pts if p["n_samples"] == n), None)
            if p is None:
                cell = ""
            elif p.get("oom"):
                cell = "out of memory"
            elif p.get("failed"):
                cell = "failed"
            else:
                peak = "-" if p["peak_bytes"] is None else f"{p['peak_bytes'] / 1e9:.2f}"
                cell = f"{np.median(p['times']['sr_step']):.3f} / {peak}"
            row += f"{cell:>32}"
        print(row)

    if differences:
        worst = max(differences.values())
        print(f"\nSame update in both libraries: max relative difference {worst:.1e} "
              f"over {len(differences)} points" + ("" if worst < 1e-6 else "   <-- CHECK"))

def plot(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    points = load_points(out_dir)
    if not points:
        raise SystemExit(f"No results in {out_dir}")
    differences = update_differences(out_dir, points)
    print_summary(cfg, points, differences)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": BASELINE,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_MUTED,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig, (ax_mem, ax_time) = plt.subplots(1, 2, figsize=(13, 4.8))
    measured = [p for pts in points.values() for p in pts if "times" in p]
    info = measured[0] if measured else {}
    has_memory = any(p.get("peak_bytes") is not None for p in measured)

    for i, (series, pts) in enumerate(points.items()):
        color = SERIES_COLORS[i]
        ok = [p for p in pts if "times" in p]
        failed = next((p for p in pts if "times" not in p), None)
        label = SERIES[series]
        if failed is not None:
            label += f" ({'out of memory' if failed.get('oom') else 'failed'} at {failed['n_samples']})"
        n = [p["n_samples"] for p in ok]

        ax_time.plot(n, [np.median(p["times"]["sr_step"]) for p in ok], "o-", color=color, lw=2, ms=6, label=label)
        if has_memory:
            ax_mem.plot(n, [p["peak_bytes"] / 1e9 for p in ok], "o-", color=color, lw=2, ms=6, label=label)
            if failed is not None and ok and ok[-1].get("bytes_limit"):
                ax_mem.plot(failed["n_samples"], ok[-1]["bytes_limit"] / 1e9, "x", color=color, ms=10, mew=2.5)

    n_all = np.array(sorted({p["n_samples"] for pts in points.values() for p in pts}))
    if has_memory:
        # Memory JAX can use
        limit = next((p["bytes_limit"] for p in measured if p.get("bytes_limit")), None)
        if limit:
            ax_mem.axhline(limit / 1e9, color=INK_MUTED, lw=1)
            ax_mem.annotate("memory available to JAX", (n_all[0], limit / 1e9), xytext=(0, 4),
                            textcoords="offset points", fontsize=8, color=INK_SECONDARY)
        if cfg["batch_size"]:
            # With more samples than this, the batched Jacobian is computed in batches
            ax_mem.axvline(cfg["batch_size"], color=INK_MUTED, lw=1, ls="--")
            ax_mem.annotate(f"batch size of the\nbatched Jacobian", (cfg["batch_size"], 0), xycoords=("data", "axes fraction"),
                            xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
        ax_mem.set_yscale("log")
        ax_mem.set_ylabel("Peak device memory [GB]" if info["memory"] == "device"
                          else "Peak resident memory of the process [GB]")
        ax_mem.legend(frameon=False, fontsize=9, loc="upper left")
    else:
        ax_mem.text(0.5, 0.5, "No memory statistics on this device\n(JAX reports them on GPUs)",
                    ha="center", va="center", color=INK_SECONDARY, transform=ax_mem.transAxes)
    ax_mem.set_title("(a) Peak memory of the process", loc="left", fontsize=11)

    ax_time.set_yscale("log")
    ax_time.set_ylabel("Time [s]")
    ax_time.set_title("(b) Time of one SR step", loc="left", fontsize=11)
    ax_time.legend(frameon=False, fontsize=9, loc="upper left")

    for ax in (ax_mem, ax_time):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of samples")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
        # Plain numbers on the log y axes, which can span less than a decade
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5)))

    versions = sorted({f"{k} {v}" for p in measured for k, v in p["versions"].items() if k in ("jVMC_exp", "netket", "jax")})
    check = (f" Same update in both libraries up to {max(differences.values()):.0e} (relative)." if differences else "")
    fig.text(
        0.01, 0.01,
        f"Transverse-field Ising chain, N = {cfg['L']}, g = {cfg['g']}. Real RBM, {cfg['num_hidden']} hidden units"
        f"{' with bias' if cfg['bias'] else ''}, {info.get('n_parameters')} parameters, float64. Uniformly random samples.\n"
        f"Eigendecomposition pseudo-inverse with diagonal shift {cfg['diag_shift']:g} in both.{check}\n"
        + (f"Only the batched Jacobian uses batches: {cfg['batch_size']} samples (Jacobian) and network evaluations (local energies). "
           if cfg["batch_size"] else "Every series evaluates all samples at once. ") +
        f"Median of {cfg['repeats']} SR steps after compilation and {cfg['warmup']} warm-up steps, sampling excluded.\n"
        f"One process per point, {info.get('n_devices', 1)} device(s): {info.get('device')}. {' | '.join(versions)}",
        fontsize=7.5, color=INK_SECONDARY, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"netket_comparison_sr.{ext}"), dpi=200)
    print(f"\nFigure saved to {os.path.join(out_dir, 'netket_comparison_sr.png')}")


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------

def run_worker(cfg, series, n_samples, out_dir):
    try:
        if series == "netket":
            result = run_netket(cfg, n_samples, out_dir)
        else:
            result = run_jvmc(cfg, n_samples, out_dir, batched=series == "jvmc-batched")
    except Exception as error:
        if not is_out_of_memory(error):
            raise
        result = {"oom": True, "error": str(error).splitlines()[0][:500]}

    save(out_dir, point_name(series, n_samples), {"series": series, "n_samples": n_samples, **result})

def main():
    args = parser.parse_args()

    if args.worker is not None:
        with open(os.path.join(args.out_dir, "config.json")) as f:
            cfg = json.load(f)
        # Only the batched Jacobian uses the batch size: the other series evaluate all samples at once
        if args.worker != "jvmc-batched":
            cfg["batch_size"] = None
        run_worker(cfg, args.worker, args.n, args.out_dir)
        return

    if args.plot is not None:
        plot(args.plot)
        return

    cfg = {
        "L": args.L, "g": args.g, "num_hidden": args.num_hidden or args.L, "bias": args.bias,
        "batch_size": args.batch_size, "diag_shift": args.diag_shift, "pinv_cutoff": args.pinv_cutoff,
        "all_devices": args.all_devices,
        "repeats": args.repeats, "warmup": args.warmup, "seed": args.seed,
        "netket_operator": args.netket_operator,
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    out_dir = args.out_dir or os.path.join(HERE, f"results_sr_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Parameters of the RBM, used by every run
    rng = np.random.default_rng(cfg["seed"])
    params = {"Dense_0/kernel": rng.normal(0, 1 / np.sqrt(cfg["L"]), (cfg["L"], cfg["num_hidden"]))}
    if cfg["bias"]:
        params["Dense_0/bias"] = rng.normal(0, 1e-2, cfg["num_hidden"])
    np.savez(os.path.join(out_dir, "params.npz"), **params)

    env = dict(os.environ)
    env["JVMC_USE_DISTRIBUTED"] = "false"
    # First of the devices given to this job, or device 0
    if not args.all_devices:
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    for series in args.series:
        python = args.netket_python if series == "netket" else sys.executable
        for n_samples in sorted(args.n_samples):
            print(f"=== {SERIES[series]}, {n_samples} samples ===", flush=True)
            path = os.path.join(out_dir, point_name(series, n_samples) + ".json")
            if os.path.exists(path):
                os.remove(path)
            run = subprocess.run(
                [python, "-u", os.path.abspath(__file__), "--worker", series, "--n", str(n_samples), "--out-dir", out_dir],
                env=env,
            )
            if run.returncode != 0 or not os.path.exists(path):
                save(out_dir, point_name(series, n_samples), {"series": series, "n_samples": n_samples, "failed": True})
                print("    FAILED -- skipping the larger numbers of samples")
                break
            with open(path) as f:
                if json.load(f).get("oom"):
                    print("    out of memory -- skipping the larger numbers of samples")
                    break

    plot(out_dir)

if __name__ == "__main__":
    main()
