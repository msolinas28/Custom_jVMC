import sys
sys.path.append(sys.path[0] + "/../../..")
import argparse
import json
import os
from datetime import datetime

import numpy as np

from compare_jvmc import environment_info, import_original, exp_hamiltonian, measure
from compare_jvmc_sr import (
    load_params, check_single_device, peak_memory, point_name, setup_jvmc_exp, setup_jvmc,
    load_points, update_differences, print_summary, plot_sweep, run_point, run_sweep,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES = {
    "exp-dense": "jVMC_exp, dense Jacobian",
    "exp-batched": "jVMC_exp, batched Jacobian",
    "jvmc": "jVMC (original)",
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--L", type=int, default=80, help="Number of spins")
parser.add_argument("--g", type=float, default=1.0, help="Transverse field")
parser.add_argument("--num-hidden", type=int, nargs="+", default=[2**k for k in range(4, 14)],
                    help="Hidden units of the RBM, one point each")
parser.add_argument("--bias", action="store_true", help="Use the bias of the hidden units")
parser.add_argument("--n-samples", type=int, default=4096)
parser.add_argument("--batch-size", type=int, default=None,
                    help="Network evaluations per call for the local energies, samples per block of the "
                         "Jacobian (default: the number of samples)")
parser.add_argument("--diag-shift", type=float, default=1e-3)
parser.add_argument("--pinv-cutoff", type=float, default=1e-14, help="Relative eigenvalue cutoff of the solver")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--series", nargs="+", choices=list(SERIES), default=list(SERIES))
parser.add_argument("--jvmc-path", default=None,
                    help="Folder with the package jVMC of the original library (default: it is importable)")
parser.add_argument("--jvmc-python", default=sys.executable,
                    help="Python interpreter of the original jVMC, if it lives in another environment")
parser.add_argument("--out-dir", default=None, help="Default: a new results_minsr_<date> folder next to this script")
parser.add_argument("--plot", metavar="RESULTS_DIR", default=None, help="Only plot the results in RESULTS_DIR")
parser.add_argument("--worker", choices=list(SERIES), default=None, help=argparse.SUPPRESS)
parser.add_argument("--n", type=int, default=None, help=argparse.SUPPRESS)


def n_parameters(cfg, num_hidden):
    return cfg["L"] * num_hidden + (num_hidden if cfg["bias"] else 0)

def params_path(out_dir, num_hidden):
    return os.path.join(out_dir, f"params_{num_hidden}.npz")


# ----------------------------------------------------------------------------------------
# jVMC_exp
# ----------------------------------------------------------------------------------------

def run_jvmc_exp(cfg, num_hidden, out_dir, batched):
    import jVMC_exp
    from jVMC_exp.objective_function import Observable
    from jVMC_exp.optimizer import MinSR
    from jVMC_exp.solver import Pinv

    check_single_device()
    n_samples = cfg["n_samples"]
    batch_size = min(cfg["batch_size"] or n_samples, n_samples)
    psi, sampler, set_samples = setup_jvmc_exp(cfg, load_params(params_path(out_dir, num_hidden)), n_samples, batch_size)
    hamiltonian = exp_hamiltonian(cfg["L"], cfg["g"])
    objective = Observable(hamiltonian, batched_jacobian=batched)
    minsr = MinSR(sampler, psi, diagonalShift=cfg["diag_shift"], solver=Pinv(pinv_cutoff=cfg["pinv_cutoff"]))

    def local_energies():
        set_samples()
        return sampler(hamiltonian).mean

    last = {}

    def minsr_step():
        # MinSR.__call__ without the sampling. MinSR uses the local energies and the Jacobian, not the force
        set_samples()
        last["update"] = minsr.get_update(objective.value_and_grad(sampler, compute_grad=False))
        return last["update"]

    first_call, times = measure([("local_energies", local_energies), ("minsr_step", minsr_step)], cfg["warmup"], cfg["repeats"])
    series = "exp-batched" if batched else "exp-dense"
    np.save(os.path.join(out_dir, f"update_{point_name(series, num_hidden)}.npy"), np.asarray(last["update"]))

    return {
        **environment_info(jVMC_exp=jVMC_exp.__version__),
        "n_parameters": psi.numParameters, "first_call": first_call, "times": times, **peak_memory(),
    }


# ----------------------------------------------------------------------------------------
# Original jVMC
# ----------------------------------------------------------------------------------------

def run_jvmc(cfg, num_hidden, out_dir):
    original_versions = import_original(cfg["jvmc_path"])
    from jVMC.stats import SampledObs
    from jVMC.util import MinSR

    check_single_device()
    n_samples = cfg["n_samples"]
    batch_size = min(cfg["batch_size"] or n_samples, n_samples)
    psi, sampler, hamiltonian, samples, weights = setup_jvmc(
        cfg, load_params(params_path(out_dir, num_hidden)), n_samples, batch_size
    )
    minsr = MinSR(sampler, pinvTol=cfg["pinv_cutoff"], diagonalShift=cfg["diag_shift"])

    def local_energies():
        log_psi = psi(samples)
        return SampledObs(hamiltonian.get_O_loc(samples, psi, log_psi), weights).mean()

    last = {}

    def minsr_step():
        # MinSR.__call__ without the sampling
        log_psi = psi(samples)
        e_loc = SampledObs(hamiltonian.get_O_loc(samples, psi, log_psi), weights)
        gradients = SampledObs(psi.gradients(samples), weights)
        last["update"] = minsr.solve(e_loc, gradients, holomorphic=psi.holomorphic)
        return last["update"]

    first_call, times = measure([("local_energies", local_energies), ("minsr_step", minsr_step)], cfg["warmup"], cfg["repeats"])
    np.save(os.path.join(out_dir, f"update_{point_name('jvmc', num_hidden)}.npy"), np.asarray(last["update"]))

    return {
        **environment_info(**original_versions),
        "n_parameters": int(psi.numParameters), "first_call": first_call, "times": times, **peak_memory(),
    }


# ----------------------------------------------------------------------------------------
# Figure and summary
# ----------------------------------------------------------------------------------------

def plot(out_dir):
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    points = load_points(out_dir, SERIES, "n_parameters")
    if not points:
        raise SystemExit(f"No results in {out_dir}")
    differences = update_differences(out_dir, points)
    print_summary(points, differences, SERIES, "n_parameters", "parameters", "MinSR")

    hidden = sorted(cfg["num_hidden"])
    footnote = (
        f"Transverse-field Ising chain, N = {cfg['L']}, g = {cfg['g']}. Real RBM with {hidden[0]} to {hidden[-1]} hidden units"
        f"{' with bias' if cfg['bias'] else ''}, float64. {cfg['n_samples']} uniformly random samples.\n"
        f"Pseudo-inverse with relative cutoff {cfg['pinv_cutoff']:g} and diagonal shift {cfg['diag_shift']:g} in all series.{{check}}\n"
        f"Batches of {cfg['batch_size'] or cfg['n_samples']} network evaluations (local energies) and samples (Jacobian). "
        f"Median of {cfg['repeats']} MinSR steps after compilation and {cfg['warmup']} warm-up steps, sampling excluded."
    )
    plot_sweep(out_dir, points, differences, SERIES, "n_parameters", "Number of parameters", "MinSR", footnote,
               "jvmc_comparison_minsr")


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------

def main():
    args = parser.parse_args()

    def info(series, num_hidden):
        return {"series": series, "n": num_hidden, "num_hidden": num_hidden, "n_parameters": n_parameters(cfg, num_hidden)}

    if args.worker is not None:
        with open(os.path.join(args.out_dir, "config.json")) as f:
            cfg = json.load(f)
        if args.worker == "jvmc":
            run = lambda: run_jvmc(cfg, args.n, args.out_dir)
        else:
            run = lambda: run_jvmc_exp(cfg, args.n, args.out_dir, batched=args.worker == "exp-batched")
        run_point(run, args.out_dir, point_name(args.worker, args.n), info(args.worker, args.n))
        return

    if args.plot is not None:
        plot(args.plot)
        return

    cfg = {
        "L": args.L, "g": args.g, "num_hidden": sorted(args.num_hidden), "bias": args.bias,
        "n_samples": args.n_samples, "batch_size": args.batch_size,
        "diag_shift": args.diag_shift, "pinv_cutoff": args.pinv_cutoff,
        "repeats": args.repeats, "warmup": args.warmup, "seed": args.seed, "jvmc_path": args.jvmc_path,
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    args.out_dir = args.out_dir or os.path.join(HERE, f"results_minsr_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for num_hidden in cfg["num_hidden"]:
        rng = np.random.default_rng([cfg["seed"], num_hidden])
        params = {"Dense_0/kernel": rng.normal(0, 1 / np.sqrt(cfg["L"] * num_hidden), (cfg["L"], num_hidden))}
        if cfg["bias"]:
            params["Dense_0/bias"] = rng.normal(0, 1e-2, num_hidden)
        np.savez(params_path(args.out_dir, num_hidden), **params)

    run_sweep(
        os.path.abspath(__file__), args, args.series, cfg["num_hidden"], info,
        lambda n: f"{n} hidden units, {n_parameters(cfg, n)} parameters", "parameters",
    )
    plot(args.out_dir)

if __name__ == "__main__":
    main()
