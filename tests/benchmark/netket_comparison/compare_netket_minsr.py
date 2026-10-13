"""
MinSR step of jVMC_exp and NetKet with a fixed number of samples and a growing number of
parameters: time and peak device memory as the Jacobian grows, and the largest number of
parameters each library can handle on one device.

    python compare_netket_minsr.py                                 # sweep, then plot
    python compare_netket_minsr.py --num-hidden 64 1024 16384      # chosen numbers of hidden units
    python compare_netket_minsr.py --plot results_minsr_<date>     # only re-plots

MinSR (arXiv:2302.01941) solves the natural gradient in the space of the samples instead of
the parameters: with the centered Jacobian O (N_s x N_p) and the centered local energies E,
both normalized by sqrt(N_s), it builds the N_s x N_s tangent kernel T = O O^T, solves
(T + diag_shift) x = E and returns the update O^T x. It is meant for N_p >> N_s, hence the
sweep over the number of parameters (hidden units of the RBM) at a fixed number of samples.

What is timed is one MinSR update from given samples, sampling excluded: log(psi) and local
energies of the samples, the Jacobian, T, the solution x and the update O^T x. The local
energies alone are timed as well.

Series:
    * jVMC_exp, dense Jacobian:   Observable(H) and MinSR(solver=Pinv), the Jacobian is stored.
    * jVMC_exp, batched Jacobian: Observable(H, batched_jacobian=True), T is built from pairs
                                  of batches of samples and the Jacobian is never stored.
    * NetKet, NTK:                VMC_SR(mode="real", use_ntk=True, on_the_fly=False), i.e.
                                  VMC_SRt. It stores the Jacobian.
    * NetKet, NTK on the fly:     VMC_SR(mode="real", use_ntk=True, on_the_fly=True), NetKet's
                                  default for MinSR: T is built from vector-Jacobian products
                                  and the Jacobian is never stored.

To compare the same computation, as in compare_netket_sr.py:
    * Real-parameter RBM: jVMC_exp.nets.RBM, and the same network in NetKet. Both libraries
      then build the same (N_s, N_p) float64 Jacobian and the same N_s x N_s kernel T.
    * Same parameters, same samples (uniformly random configurations: the update is not a
      physical one, but it costs the same) and the same linear solver, the eigendecomposition
      pseudo-inverse of jVMC_exp.solver.Pinv with the same diagonal shift and cutoff. The
      libraries then give the same update, up to the factor -2 of NetKet's convention, and
      this is checked.
    * Same batch size: at most --batch-size network evaluations per call for the local
      energies and --batch-size samples per block of the Jacobian (batchSize of the NQS, and
      chunk_size of the MCState, which VMC_SR also uses for the Jacobian).
    * One device. Every point runs in its own process, since the peak memory of a process
      can't be reset, and a series stops at the first number of parameters that does not fit.
    * The first call (compilation) and --warmup calls are excluded, then the median of
      --repeats calls.
"""
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
from compare_netket_sr import check_single_device, peak_memory, is_out_of_memory

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES = {
    "jvmc-dense": "jVMC_exp, dense Jacobian",
    "jvmc-batched": "jVMC_exp, batched Jacobian",
    "netket": "NetKet, NTK",
    "netket-onthefly": "NetKet, NTK on the fly",
}
NETKET_SERIES = ("netket", "netket-onthefly")
# NetKet's MinSR solves (T + eps) x = 2 Re(E) and returns O^T x, jVMC_exp's solves
# (T + eps) x = E and returns -O^T x
NETKET_UPDATE_FACTOR = -2

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--L", type=int, default=80, help="Number of spins")
parser.add_argument("--g", type=float, default=1.0, help="Transverse field")
parser.add_argument("--num-hidden", type=int, nargs="+", default=[2**k for k in range(4, 15)],
                    help="Hidden units of the RBM, one point each")
parser.add_argument("--bias", action="store_true", help="Use the bias of the hidden units")
parser.add_argument("--n-samples", type=int, default=4096)
parser.add_argument("--batch-size", type=int, default=1024,
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
parser.add_argument("--out-dir", default=None, help="Default: a new results_minsr_<date> folder next to this script")
parser.add_argument("--plot", metavar="RESULTS_DIR", default=None, help="Only plot the results in RESULTS_DIR")
parser.add_argument("--worker", choices=list(SERIES), default=None, help=argparse.SUPPRESS)
parser.add_argument("--n", type=int, default=None, help=argparse.SUPPRESS)


# ----------------------------------------------------------------------------------------
# Shared by the workers
# ----------------------------------------------------------------------------------------

def n_parameters(cfg, num_hidden):
    return cfg["L"] * num_hidden + (num_hidden if cfg["bias"] else 0)

def params_path(out_dir, num_hidden):
    return os.path.join(out_dir, f"params_{num_hidden}.npz")

def load_inputs(cfg, num_hidden, out_dir):
    """
    The parameters written by the driver for this number of hidden units, and the samples:
    0/1 configurations, the same in every process.
    """
    from flax.traverse_util import unflatten_dict

    with np.load(params_path(out_dir, num_hidden)) as f:
        params = unflatten_dict({tuple(k.split("/")): f[k] for k in f.files})
    samples = np.random.default_rng(cfg["seed"]).integers(0, 2, (cfg["n_samples"], cfg["L"]), dtype=np.int8)

    return params, samples

def point_name(series, num_hidden):
    return f"{series}_{num_hidden}"


# ----------------------------------------------------------------------------------------
# jVMC_exp
# ----------------------------------------------------------------------------------------

def run_jvmc(cfg, num_hidden, out_dir, batched):
    import jax
    import jax.numpy as jnp
    import jVMC_exp
    from jVMC_exp.nets import RBM
    from jVMC_exp.objective_function import Observable
    from jVMC_exp.optimizer import MinSR
    from jVMC_exp.sampler import MCSampler
    from jVMC_exp.sampler.propose import SpinFlip
    from jVMC_exp.solver import Pinv
    from jVMC_exp.vqs import NQS

    check_single_device()
    L, n_samples = cfg["L"], cfg["n_samples"]
    params, samples = load_inputs(cfg, num_hidden, out_dir)
    samples = jnp.asarray(samples, dtype=jVMC_exp.global_defs.DT_SAMPLES)
    weights = jnp.full(n_samples, 1 / n_samples)

    hamiltonian = jvmc_hamiltonian(L, cfg["g"])
    psi = NQS(RBM(numHidden=num_hidden, bias=cfg["bias"]), L, batchSize=cfg["batch_size"], seed=cfg["seed"])
    psi.parameters = jax.tree_util.tree_map(jnp.asarray, params)
    # Never samples: the samples are set below, where MinSR sets the ones it draws
    sampler = MCSampler(psi, SpinFlip(), key=cfg["seed"], numChains=1, numSamples=n_samples)
    objective = Observable(hamiltonian, batched_jacobian=batched)
    minsr = MinSR(sampler, psi, diagonalShift=cfg["diag_shift"], solver=Pinv(pinv_cutoff=cfg["pinv_cutoff"]))

    def set_samples():
        # jVMC_exp evaluates log(psi) of the samples when sampling, NetKet in the MinSR step
        sampler._samples, sampler._logPsi, sampler._weights = samples, psi(samples), weights

    def local_energies():
        set_samples()
        return sampler(hamiltonian).mean

    last = {}

    def minsr_step():
        set_samples()
        # MinSR uses the local energies and the Jacobian, not the force
        last["update"] = minsr.get_update(objective.value_and_grad(sampler, compute_grad=False))
        return last["update"]

    first_call, times = measure([("local_energies", local_energies), ("minsr_step", minsr_step)], cfg["warmup"], cfg["repeats"])
    np.save(os.path.join(out_dir, f"update_{point_name('jvmc-batched' if batched else 'jvmc-dense', num_hidden)}.npy"),
            np.asarray(last["update"]))

    return {
        **environment_info(jVMC_exp=jVMC_exp.__version__),
        "n_parameters": psi.numParameters, "first_call": first_call, "times": times, **peak_memory(),
    }


# ----------------------------------------------------------------------------------------
# NetKet
# ----------------------------------------------------------------------------------------

def run_netket(cfg, num_hidden, out_dir, on_the_fly):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    import netket as nk
    import optax

    check_single_device()
    L, n_samples = cfg["L"], cfg["n_samples"]
    params, samples = load_inputs(cfg, num_hidden, out_dir)

    hamiltonian = netket_operator(cfg["netket_operator"], L, cfg["g"])
    sampler = nk.sampler.MetropolisLocal(hamiltonian.hilbert, n_chains=1)
    with warnings.catch_warnings():
        # The batch size is the same as in jVMC_exp, power of two or not
        warnings.filterwarnings("ignore", message=".*power-of-two chunk size")
        vstate = nk.vqs.MCState(
            sampler, netket_rbm(num_hidden, cfg["bias"], "float64"),
            n_samples=n_samples, chunk_size=cfg["batch_size"], seed=cfg["seed"],
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
        mode="real", use_ntk=True, on_the_fly=on_the_fly,
    )
    # NetKet has no public way to set the samples: these are the ones MCState.sample() would
    # store, with the 0/1 spins of jVMC_exp as +-1
    vstate._samples = jnp.asarray(2 * samples - 1, dtype=sampler.dtype).reshape(1, n_samples, L)

    last = {}

    def minsr_step():
        last["update"] = driver.compute_loss_and_update()[1]
        return last["update"]

    first_call, times = measure(
        [("local_energies", lambda: vstate.local_estimators(hamiltonian)), ("minsr_step", minsr_step)],
        cfg["warmup"], cfg["repeats"]
    )
    series = "netket-onthefly" if on_the_fly else "netket"
    np.save(os.path.join(out_dir, f"update_{point_name(series, num_hidden)}.npy"),
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
    For every series, its points sorted by the number of parameters.
    """
    points = {}
    for series in SERIES:
        found = []
        for name in os.listdir(out_dir):
            if name.startswith(series + "_") and name.endswith(".json") and name[len(series) + 1:-5].isdigit():
                with open(os.path.join(out_dir, name)) as f:
                    found.append(json.load(f))
        if found:
            points[series] = sorted(found, key=lambda p: p["n_parameters"])

    return points

def update_differences(out_dir, points):
    """
    Relative difference between the update of every series and the one of jVMC_exp with a
    dense Jacobian, for every number of hidden units where both are available.
    """
    differences = {}
    for p in points.get("jvmc-dense", []):
        reference = os.path.join(out_dir, f"update_{point_name('jvmc-dense', p['num_hidden'])}.npy")
        if not os.path.exists(reference):
            continue
        reference = np.real(np.load(reference))
        for series in SERIES:
            path = os.path.join(out_dir, f"update_{point_name(series, p['num_hidden'])}.npy")
            if series == "jvmc-dense" or not os.path.exists(path):
                continue
            expected = NETKET_UPDATE_FACTOR * reference if series in NETKET_SERIES else reference
            differences[(series, p["num_hidden"])] = (
                np.linalg.norm(np.real(np.load(path)) - expected) / np.linalg.norm(expected)
            )

    return differences

def print_summary(points, differences):
    n_all = sorted({p["n_parameters"] for pts in points.values() for p in pts})
    print("\nMinSR step: median time [s] / peak device memory [GB]")
    print(f"{'parameters':>11}" + "".join(f"{SERIES[s]:>30}" for s in points))
    for n in n_all:
        row = f"{n:>11}"
        for series, pts in points.items():
            p = next((p for p in pts if p["n_parameters"] == n), None)
            if p is None:
                cell = ""
            elif p.get("oom"):
                cell = "out of memory"
            elif p.get("failed"):
                cell = "failed"
            else:
                peak = "-" if p["peak_bytes"] is None else f"{p['peak_bytes'] / 1e9:.2f}"
                cell = f"{np.median(p['times']['minsr_step']):.3f} / {peak}"
            row += f"{cell:>30}"
        print(row)

    if differences:
        worst = max(differences.values())
        print(f"\nSame update in all series: max relative difference {worst:.1e} "
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
    print_summary(points, differences)

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
            label += f" ({'out of memory' if failed.get('oom') else 'failed'} at {failed['n_parameters']})"
        n = [p["n_parameters"] for p in ok]

        ax_time.plot(n, [np.median(p["times"]["minsr_step"]) for p in ok], "o-", color=color, lw=2, ms=6, label=label)
        ax_time.plot(n, [np.median(p["times"]["local_energies"]) for p in ok], ":", color=color, lw=1.5)
        if has_memory:
            ax_mem.plot(n, [p["peak_bytes"] / 1e9 for p in ok], "o-", color=color, lw=2, ms=6, label=label)
            if failed is not None and ok and ok[-1].get("bytes_limit"):
                ax_mem.plot(failed["n_parameters"], ok[-1]["bytes_limit"] / 1e9, "x", color=color, ms=10, mew=2.5)

    n_all = np.array(sorted({p["n_parameters"] for pts in points.values() for p in pts}))
    if has_memory:
        # Size of the Jacobian and of the tangent kernel, and of the memory JAX can use
        ax_mem.plot(n_all, cfg["n_samples"] * n_all * 8 / 1e9, "--", color=INK_MUTED, lw=1, label="Jacobian, float64")
        ax_mem.axhline(cfg["n_samples"] ** 2 * 8 / 1e9, color=INK_MUTED, lw=1, ls=":", label="Tangent kernel T, float64")
        limit = next((p["bytes_limit"] for p in measured if p.get("bytes_limit")), None)
        if limit:
            ax_mem.axhline(limit / 1e9, color=INK_MUTED, lw=1)
            ax_mem.annotate("memory available to JAX", (n_all[0], limit / 1e9), xytext=(0, 4),
                            textcoords="offset points", fontsize=8, color=INK_SECONDARY)
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
    ax_time.set_title("(b) Time of one MinSR step (dotted: local energies alone)", loc="left", fontsize=11)
    ax_time.legend(frameon=False, fontsize=9, loc="upper left")

    for ax in (ax_mem, ax_time):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of parameters")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)

    versions = sorted({f"{k} {v}" for p in measured for k, v in p["versions"].items() if k in ("jVMC_exp", "netket", "jax")})
    check = (f" Same update in all series up to {max(differences.values()):.0e} (relative)." if differences else "")
    hidden = sorted(cfg["num_hidden"])
    fig.text(
        0.01, 0.01,
        f"Transverse-field Ising chain, N = {cfg['L']}, g = {cfg['g']}. Real RBM with {hidden[0]} to {hidden[-1]} hidden units"
        f"{' with bias' if cfg['bias'] else ''}, float64. {cfg['n_samples']} uniformly random samples.\n"
        f"Eigendecomposition pseudo-inverse with diagonal shift {cfg['diag_shift']:g} in all series.{check}\n"
        f"Batches of {cfg['batch_size']} network evaluations (local energies) and samples (Jacobian). "
        f"Median of {cfg['repeats']} MinSR steps after compilation and {cfg['warmup']} warm-up steps, sampling excluded.\n"
        f"One process per point. {info.get('device')}. {' | '.join(versions)}",
        fontsize=7.5, color=INK_SECONDARY, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"netket_comparison_minsr.{ext}"), dpi=200)
    print(f"\nFigure saved to {os.path.join(out_dir, 'netket_comparison_minsr.png')}")


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------

def run_worker(cfg, series, num_hidden, out_dir):
    try:
        if series in NETKET_SERIES:
            result = run_netket(cfg, num_hidden, out_dir, on_the_fly=series == "netket-onthefly")
        else:
            result = run_jvmc(cfg, num_hidden, out_dir, batched=series == "jvmc-batched")
    except Exception as error:
        if not is_out_of_memory(error):
            raise
        result = {"oom": True, "error": str(error).splitlines()[0][:500]}

    save(out_dir, point_name(series, num_hidden), {
        "series": series, "num_hidden": num_hidden, "n_parameters": n_parameters(cfg, num_hidden), **result
    })

def main():
    args = parser.parse_args()

    if args.worker is not None:
        with open(os.path.join(args.out_dir, "config.json")) as f:
            cfg = json.load(f)
        run_worker(cfg, args.worker, args.n, args.out_dir)
        return

    if args.plot is not None:
        plot(args.plot)
        return

    cfg = {
        "L": args.L, "g": args.g, "num_hidden": sorted(args.num_hidden), "bias": args.bias,
        "n_samples": args.n_samples, "batch_size": args.batch_size,
        "diag_shift": args.diag_shift, "pinv_cutoff": args.pinv_cutoff,
        "repeats": args.repeats, "warmup": args.warmup, "seed": args.seed,
        "netket_operator": args.netket_operator,
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    out_dir = args.out_dir or os.path.join(HERE, f"results_minsr_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Parameters of the RBM for every number of hidden units, used by every run. The standard
    # deviation keeps the size of log(psi) independent of the number of hidden units.
    for num_hidden in cfg["num_hidden"]:
        rng = np.random.default_rng([cfg["seed"], num_hidden])
        params = {"Dense_0/kernel": rng.normal(0, 1 / np.sqrt(cfg["L"] * num_hidden), (cfg["L"], num_hidden))}
        if cfg["bias"]:
            params["Dense_0/bias"] = rng.normal(0, 1e-2, num_hidden)
        np.savez(params_path(out_dir, num_hidden), **params)

    env = dict(os.environ)
    env["JVMC_USE_DISTRIBUTED"] = "false"
    # First of the devices given to this job, or device 0
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    for series in args.series:
        python = args.netket_python if series in NETKET_SERIES else sys.executable
        for num_hidden in cfg["num_hidden"]:
            print(f"=== {SERIES[series]}, {num_hidden} hidden units, "
                  f"{n_parameters(cfg, num_hidden)} parameters ===", flush=True)
            path = os.path.join(out_dir, point_name(series, num_hidden) + ".json")
            if os.path.exists(path):
                os.remove(path)
            run = subprocess.run(
                [python, "-u", os.path.abspath(__file__), "--worker", series, "--n", str(num_hidden), "--out-dir", out_dir],
                env=env,
            )
            if run.returncode != 0 or not os.path.exists(path):
                save(out_dir, point_name(series, num_hidden), {
                    "series": series, "num_hidden": num_hidden,
                    "n_parameters": n_parameters(cfg, num_hidden), "failed": True,
                })
                print("    FAILED -- skipping the larger numbers of parameters")
                break
            with open(path) as f:
                if json.load(f).get("oom"):
                    print("    out of memory -- skipping the larger numbers of parameters")
                    break

    plot(out_dir)

if __name__ == "__main__":
    main()
