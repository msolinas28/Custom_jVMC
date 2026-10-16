import sys
sys.path.append(sys.path[0] + "/../../..")
import argparse
import json
import os
import subprocess
import warnings
from datetime import datetime

import numpy as np

from compare_jvmc import (
    measure, environment_info, save, import_original, exp_hamiltonian, original_hamiltonian,
    original_n_conn, child_env, SERIES_COLORS, PLOT_STYLE, INK_SECONDARY, INK_MUTED, GRID,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES = {
    "exp-dense": "jVMC_exp, dense Jacobian",
    "exp-batched": "jVMC_exp, batched Jacobian",
    "jvmc": "jVMC (original)",
    "jvmc-nosnr": "jVMC (original), no SNR estimate",
}
ORIGINAL_SERIES = ("jvmc", "jvmc-nosnr")
# Series that can't apply the SNR regularization
WITHOUT_SNR = ("exp-batched", "jvmc-nosnr")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--L", type=int, default=80, help="Number of spins")
parser.add_argument("--g", type=float, default=1.0, help="Transverse field")
parser.add_argument("--num-hidden", type=int, default=None, help="Hidden units of the RBM (default: L)")
parser.add_argument("--bias", action="store_true", help="Use the bias of the hidden units")
parser.add_argument("--n-samples", type=int, nargs="+", default=[2**k for k in range(10, 20)])
parser.add_argument("--batch-size", type=int, default=None,
                    help="Network evaluations per call for the local energies, samples per block of the "
                         "Jacobian (default: the number of samples)")
parser.add_argument("--diag-scale", type=float, default=1e-3, help="Diagonal regularization S_kk -> (1 + diag_scale) S_kk")
parser.add_argument("--snr-tol", type=float, default=0., help="SNR cutoff of the solver (0: no SNR regularization)")
parser.add_argument("--pinv-tol", type=float, default=1e-14, help="Target residual of the adaptive eigenvalue cutoff")
parser.add_argument("--pinv-cutoff", type=float, default=1e-8, help="Smallest relative eigenvalue cutoff")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--series", nargs="+", choices=list(SERIES), default=list(SERIES))
parser.add_argument("--jvmc-path", default=None,
                    help="Folder with the package jVMC of the original library (default: it is importable)")
parser.add_argument("--jvmc-python", default=sys.executable,
                    help="Python interpreter of the original jVMC, if it lives in another environment")
parser.add_argument("--out-dir", default=None, help="Default: a new results_sr_<date> folder next to this script")
parser.add_argument("--plot", metavar="RESULTS_DIR", default=None, help="Only plot the results in RESULTS_DIR")
parser.add_argument("--worker", choices=list(SERIES), default=None, help=argparse.SUPPRESS)
parser.add_argument("--n", type=int, default=None, help=argparse.SUPPRESS)

def load_params(path):
    from flax.traverse_util import unflatten_dict

    with np.load(path) as f:
        return unflatten_dict({tuple(k.split("/")): f[k] for k in f.files})

def random_samples(cfg, n_samples):
    """
    0/1 configurations, the same in every process.
    """
    return np.random.default_rng(cfg["seed"]).integers(0, 2, (n_samples, cfg["L"]), dtype=np.int32)

def check_single_device():
    import jax

    if jax.device_count() != 1:
        raise SystemExit(f"{jax.device_count()} devices are visible, but the comparison runs on one")

def peak_memory():
    import resource
    import jax

    stats = jax.local_devices()[0].memory_stats()
    if stats is not None:
        return {"peak_bytes": stats.get("peak_bytes_in_use"), "bytes_limit": stats.get("bytes_limit"), "memory": "device"}

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_bytes": rss if sys.platform == "darwin" else 1024 * rss, "bytes_limit": None, "memory": "process"}

def is_out_of_memory(error):
    return "RESOURCE_EXHAUSTED" in str(error) or "out of memory" in str(error).lower()

def point_name(series, n):
    return f"{series}_{n}"

def setup_jvmc_exp(cfg, params, n_samples, batch_size):
    """
    The NQS and a sampler of jVMC_exp that returns the given samples.
    """
    import jax
    import jax.numpy as jnp
    from jVMC_exp.nets import RBM
    from jVMC_exp.sampler import MCSampler
    from jVMC_exp.sampler.propose import SpinFlip
    from jVMC_exp.vqs import NQS

    psi = NQS(RBM(numHidden=params["Dense_0"]["kernel"].shape[1], bias=cfg["bias"]), cfg["L"],
              batchSize=batch_size, seed=cfg["seed"])
    psi.parameters = jax.tree_util.tree_map(jnp.asarray, params)
    sampler = MCSampler(psi, SpinFlip(), key=cfg["seed"], numChains=1, numSamples=n_samples)

    samples = jnp.asarray(random_samples(cfg, n_samples))
    weights = jnp.full(n_samples, 1 / n_samples)

    def set_samples():
        sampler._samples, sampler._logPsi, sampler._weights = samples, psi(samples), weights

    return psi, sampler, set_samples

def setup_jvmc(cfg, params, n_samples, batch_size):
    """
    The NQS, a sampler (which never samples) and the Hamiltonian of the original jVMC, and the
    samples and their weights with the leading device axis of the original.
    """
    import jax
    import jax.numpy as jnp
    import jVMC.mpi_wrapper as mpi
    from jVMC.nets import RBM
    from jVMC.sampler import MCSampler, propose_spin_flip
    from jVMC.vqs import NQS

    L = cfg["L"]
    mpi.distribute_sampling(n_samples, localDevices=1)
    hamiltonian = original_hamiltonian(L, cfg["g"])
    hamiltonian.ElocBatchSize = max(1, batch_size // original_n_conn(hamiltonian))

    psi = NQS(RBM(numHidden=params["Dense_0"]["kernel"].shape[1], bias=cfg["bias"]), batchSize=batch_size, seed=cfg["seed"])
    sampler = MCSampler(psi, (L,), cfg["seed"], updateProposer=propose_spin_flip, numChains=1, numSamples=n_samples)
    psi.params = jax.tree_util.tree_map(jnp.asarray, params)

    samples = jnp.asarray(random_samples(cfg, n_samples))[None]
    weights = jnp.full((1, n_samples), 1 / n_samples)

    return psi, sampler, hamiltonian, samples, weights


# ----------------------------------------------------------------------------------------
# jVMC_exp
# ----------------------------------------------------------------------------------------

def run_jvmc_exp(cfg, n_samples, out_dir, batched):
    import jVMC_exp
    from jVMC_exp.objective_function import Observable
    from jVMC_exp.optimizer import SR
    from jVMC_exp.solver import PinvSNR

    check_single_device()
    batch_size = min(cfg["batch_size"] or n_samples, n_samples)
    psi, sampler, set_samples = setup_jvmc_exp(cfg, load_params(os.path.join(out_dir, "params.npz")), n_samples, batch_size)
    hamiltonian = exp_hamiltonian(cfg["L"], cfg["g"])
    objective = Observable(hamiltonian, batched_jacobian=batched)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sr = SR(
            sampler, psi, diagonalShift=0., diagonalScale=cfg["diag_scale"],
            solver=PinvSNR(snr_tol=cfg["snr_tol"], pinv_tol=cfg["pinv_tol"], pinv_cutoff=cfg["pinv_cutoff"]),
        )

    def local_energies():
        set_samples()
        return sampler(hamiltonian).mean

    last = {}

    def sr_step():
        set_samples()
        last["update"] = sr.get_update(objective.value_and_grad(sampler, compute_grad=True))
        return last["update"]

    first_call, times = measure([("local_energies", local_energies), ("sr_step", sr_step)], cfg["warmup"], cfg["repeats"])
    series = "exp-batched" if batched else "exp-dense"
    np.save(os.path.join(out_dir, f"update_{point_name(series, n_samples)}.npy"), np.asarray(last["update"]))

    return {
        **environment_info(jVMC_exp=jVMC_exp.__version__),
        "n_parameters": psi.numParameters, "first_call": first_call, "times": times, **peak_memory(),
    }


# ----------------------------------------------------------------------------------------
# Original jVMC
# ----------------------------------------------------------------------------------------

def tdvp_without_snr():
    from jVMC.util import TDVP

    class TDVPWithoutSNR(TDVP):
        """
        ``TDVP.solve`` estimates the SNR of every eigenmode of the update in every step. With
        snrTol = 0 the SNR does not change the update, and an infinite SNR gives the same one.
        """
        def _get_snr(self, Eloc, gradients):
            import jax.numpy as jnp

            self.snr = jnp.full(self.ev.shape, jnp.inf)

    return TDVPWithoutSNR

def run_jvmc(cfg, n_samples, out_dir, skip_snr):
    original_versions = import_original(cfg["jvmc_path"])
    from jVMC.stats import SampledObs
    from jVMC.util import TDVP

    check_single_device()
    batch_size = min(cfg["batch_size"] or n_samples, n_samples)
    psi, sampler, hamiltonian, samples, weights = setup_jvmc(
        cfg, load_params(os.path.join(out_dir, "params.npz")), n_samples, batch_size
    )
    tdvp = (tdvp_without_snr() if skip_snr else TDVP)(
        sampler, snrTol=cfg["snr_tol"], pinvTol=cfg["pinv_tol"], pinvCutoff=cfg["pinv_cutoff"],
        makeReal="real", rhsPrefactor=1., diagonalShift=cfg["diag_scale"],
    )

    def local_energies():
        log_psi = psi(samples)
        return SampledObs(hamiltonian.get_O_loc(samples, psi, log_psi), weights).mean()

    last = {}

    def sr_step():
        # TDVP.__call__ without the sampling
        log_psi = psi(samples)
        e_loc = SampledObs(hamiltonian.get_O_loc(samples, psi, log_psi), weights)
        gradients = SampledObs(psi.gradients(samples), weights)
        last["update"] = tdvp.solve(e_loc, gradients)[0]
        return last["update"]

    first_call, times = measure([("local_energies", local_energies), ("sr_step", sr_step)], cfg["warmup"], cfg["repeats"])
    series = "jvmc-nosnr" if skip_snr else "jvmc"
    np.save(os.path.join(out_dir, f"update_{point_name(series, n_samples)}.npy"), np.asarray(last["update"]))

    return {
        **environment_info(**original_versions),
        "n_parameters": int(psi.numParameters), "first_call": first_call, "times": times, **peak_memory(),
    }


# ----------------------------------------------------------------------------------------
# Figure and summary (shared with compare_jvmc_minsr.py)
# ----------------------------------------------------------------------------------------

def load_points(out_dir, series_names, x_key):
    """
    For every series, its points sorted by ``x_key``.
    """
    points = {}
    for series in series_names:
        found = []
        for name in os.listdir(out_dir):
            if name.startswith(series + "_") and name.endswith(".json") and name[len(series) + 1:-5].isdigit():
                with open(os.path.join(out_dir, name)) as f:
                    found.append(json.load(f))
        if found:
            points[series] = sorted(found, key=lambda p: p[x_key])

    return points

def update_differences(out_dir, points):
    """
    Relative difference between the update of every series and the one of jVMC_exp with a
    dense Jacobian, for every point where both are available.
    """
    differences = {}
    for p in points.get("exp-dense", []):
        reference = os.path.join(out_dir, f"update_{point_name('exp-dense', p['n'])}.npy")
        if not os.path.exists(reference):
            continue
        reference = np.real(np.load(reference))
        for series in points:
            path = os.path.join(out_dir, f"update_{point_name(series, p['n'])}.npy")
            if series == "exp-dense" or not os.path.exists(path):
                continue
            differences[(series, p["n"])] = np.linalg.norm(np.real(np.load(path)) - reference) / np.linalg.norm(reference)

    return differences

def print_summary(points, differences, labels, x_key, x_label, step):
    n_all = sorted({p[x_key] for pts in points.values() for p in pts})
    width = max(len(labels[s]) for s in points) + 3
    print(f"\n{step}: median time [s] / peak memory [GB]")
    print(f"{x_label:>11}" + "".join(f"{labels[s]:>{width}}" for s in points))
    for n in n_all:
        row = f"{n:>11}"
        for series, pts in points.items():
            p = next((p for p in pts if p[x_key] == n), None)
            if p is None:
                cell = ""
            elif p.get("oom"):
                cell = "out of memory"
            elif p.get("failed"):
                cell = "failed"
            else:
                peak = "-" if p["peak_bytes"] is None else f"{p['peak_bytes'] / 1e9:.2f}"
                cell = f"{np.median(p['times'][step.lower() + '_step']):.3f} / {peak}"
            row += f"{cell:>{width}}"
        print(row)

    if differences:
        # NaN if any update is NaN
        worst = np.max(list(differences.values()))
        print(f"\nSame update in all series: max relative difference {worst:.1e} "
              f"over {len(differences)} points" + ("" if worst < 1e-6 else "   <-- CHECK"))

def plot_sweep(out_dir, points, differences, labels, x_key, x_label, step, footnote, file_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(PLOT_STYLE)
    fig, (ax_mem, ax_time) = plt.subplots(1, 2, figsize=(13, 4.8))
    measured = [p for pts in points.values() for p in pts if "times" in p]
    info = measured[0] if measured else {}
    has_memory = any(p.get("peak_bytes") is not None for p in measured)

    for series, pts in points.items():
        color = SERIES_COLORS[list(labels).index(series)]
        ok = [p for p in pts if "times" in p]
        failed = next((p for p in pts if "times" not in p), None)
        label = labels[series]
        if failed is not None:
            label += f" ({'out of memory' if failed.get('oom') else 'failed'} at {failed[x_key]})"
        n = [p[x_key] for p in ok]

        ax_time.plot(n, [np.median(p["times"][step.lower() + "_step"]) for p in ok], "o-", color=color, lw=2, ms=6, label=label)
        if has_memory:
            ax_mem.plot(n, [p["peak_bytes"] / 1e9 for p in ok], "o-", color=color, lw=2, ms=6, label=label)
            if failed is not None and ok and ok[-1].get("bytes_limit"):
                ax_mem.plot(failed[x_key], ok[-1]["bytes_limit"] / 1e9, "x", color=color, ms=10, mew=2.5)

    n_all = np.array(sorted({p[x_key] for pts in points.values() for p in pts}))
    if has_memory:
        # Memory JAX can use
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
        ax_mem.text(0.5, 0.5, "No memory statistics", ha="center", va="center", color=INK_SECONDARY,
                    transform=ax_mem.transAxes)
    ax_mem.set_title("(a) Peak memory of the process", loc="left", fontsize=11)

    ax_time.set_yscale("log")
    ax_time.set_ylabel("Time [s]")
    ax_time.set_title(f"(b) Time of one {step} step", loc="left", fontsize=11)
    ax_time.legend(frameon=False, fontsize=9, loc="upper left")

    for ax in (ax_mem, ax_time):
        ax.set_xscale("log", base=2)
        ax.set_xlabel(x_label)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
        # Plain numbers on the log y axes, which can span less than a decade
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5)))

    versions = sorted({f"{k} {v}" for p in measured for k, v in p["versions"].items() if k in ("jVMC_exp", "jVMC", "jax")})
    check = f" Same update in all series up to {np.max(list(differences.values())):.0e} (relative)." if differences else ""
    fig.text(
        0.01, 0.01,
        footnote.format(check=check, n_parameters=info.get("n_parameters"))
        + f"\nOne process per point. {info.get('device')}. {' | '.join(versions)}",
        fontsize=7.5, color=INK_SECONDARY, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{file_name}.{ext}"), dpi=200)
    print(f"\nFigure saved to {os.path.join(out_dir, file_name + '.png')}")

def plot(out_dir):
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    points = load_points(out_dir, SERIES, "n_samples")
    if not points:
        raise SystemExit(f"No results in {out_dir}")
    differences = update_differences(out_dir, points)
    print_summary(points, differences, SERIES, "n_samples", "samples", "SR")

    snr = f", SNR cutoff {cfg['snr_tol']:g}" if cfg["snr_tol"] else ", no SNR cutoff"
    footnote = (
        f"Transverse-field Ising chain, N = {cfg['L']}, g = {cfg['g']}. Real RBM, {cfg['num_hidden']} hidden units"
        f"{' with bias' if cfg['bias'] else ''}, {{n_parameters}} parameters, float64. Uniformly random samples.\n"
        f"Solver of the original TDVP (jVMC_exp.solver.PinvSNR) with S_kk -> (1 + {cfg['diag_scale']:g}) S_kk, "
        f"pinv_tol {cfg['pinv_tol']:g}, pinv_cutoff {cfg['pinv_cutoff']:g}{snr}.{{check}}\n"
        f"Batches of {cfg['batch_size'] or 'all the'} network evaluations (local energies) and samples (Jacobian). "
        f"Median of {cfg['repeats']} SR steps after compilation and {cfg['warmup']} warm-up steps, sampling excluded."
    )
    plot_sweep(out_dir, points, differences, SERIES, "n_samples", "Number of samples", "SR", footnote, "jvmc_comparison_sr")


# ----------------------------------------------------------------------------------------
# Driver (shared with compare_jvmc_minsr.py)
# ----------------------------------------------------------------------------------------

def run_point(run, out_dir, name, info):
    """
    Runs one point of a sweep and saves its result, also when it does not fit in memory.
    """
    try:
        result = run()
    except Exception as error:
        if not is_out_of_memory(error):
            raise
        result = {"oom": True, "error": str(error).splitlines()[0][:500]}

    save(out_dir, name, {**info, **result})

def run_sweep(script, args, series_list, values, info, describe, unit):
    """
    One process per series and value; a series stops at its first value that fails or does
    not fit in memory.
    """
    for series in series_list:
        python = args.jvmc_python if series in ORIGINAL_SERIES else sys.executable
        for n in values:
            print(f"=== {series}, {describe(n)} ===", flush=True)
            path = os.path.join(args.out_dir, point_name(series, n) + ".json")
            if os.path.exists(path):
                os.remove(path)
            run = subprocess.run(
                [python, "-u", script, "--worker", series, "--n", str(n), "--out-dir", args.out_dir],
                env=child_env(),
            )
            if run.returncode != 0 or not os.path.exists(path):
                save(args.out_dir, point_name(series, n), {**info(series, n), "failed": True})
                print(f"    FAILED -- skipping the larger numbers of {unit}")
                break
            with open(path) as f:
                if json.load(f).get("oom"):
                    print(f"    out of memory -- skipping the larger numbers of {unit}")
                    break

def snr_series(series_list, snr_tol):
    """
    The series that can run with this SNR cutoff.
    """
    if not snr_tol:
        return series_list
    skipped = [s for s in series_list if s in WITHOUT_SNR]
    if skipped:
        print(f"--snr-tol {snr_tol:g}: skipping {', '.join(skipped)}, which have no SNR regularization")
    return [s for s in series_list if s not in WITHOUT_SNR]

def main():
    args = parser.parse_args()

    def info(series, n_samples):
        return {"series": series, "n": n_samples, "n_samples": n_samples}

    if args.worker is not None:
        with open(os.path.join(args.out_dir, "config.json")) as f:
            cfg = json.load(f)
        if args.worker in ORIGINAL_SERIES:
            run = lambda: run_jvmc(cfg, args.n, args.out_dir, skip_snr=args.worker == "jvmc-nosnr")
        else:
            run = lambda: run_jvmc_exp(cfg, args.n, args.out_dir, batched=args.worker == "exp-batched")
        run_point(run, args.out_dir, point_name(args.worker, args.n), info(args.worker, args.n))
        return

    if args.plot is not None:
        plot(args.plot)
        return

    cfg = {
        "L": args.L, "g": args.g, "num_hidden": args.num_hidden or args.L, "bias": args.bias,
        "batch_size": args.batch_size, "diag_scale": args.diag_scale, "snr_tol": args.snr_tol,
        "pinv_tol": args.pinv_tol, "pinv_cutoff": args.pinv_cutoff,
        "repeats": args.repeats, "warmup": args.warmup, "seed": args.seed, "jvmc_path": args.jvmc_path,
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    args.out_dir = args.out_dir or os.path.join(HERE, f"results_sr_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    rng = np.random.default_rng(cfg["seed"])
    params = {"Dense_0/kernel": rng.normal(0, 1 / np.sqrt(cfg["L"]), (cfg["L"], cfg["num_hidden"]))}
    if cfg["bias"]:
        params["Dense_0/bias"] = rng.normal(0, 1e-2, cfg["num_hidden"])
    np.savez(os.path.join(args.out_dir, "params.npz"), **params)

    run_sweep(
        os.path.abspath(__file__), args, snr_series(args.series, args.snr_tol), sorted(args.n_samples), info,
        lambda n: f"{n} samples", "samples",
    )
    plot(args.out_dir)

if __name__ == "__main__":
    main()
