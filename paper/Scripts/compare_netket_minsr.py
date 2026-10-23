"""
One MinSR step of jVMC_exp and NetKet at fixed number of samples with a growing number of parameters:
time and peak memory.

Copy of tests/benchmark/netket_comparison/compare_netket_minsr.py. The helpers it imported from
compare_netket.py and compare_netket_sr.py come from the copy of compare_netket_sr.py in this folder. The
figures are made by paper/plot_netket_comparison_minsr_*.py.

Transverse-field Ising chain (PBC), real RBM with a growing number of hidden units, float64, the same
parameters and uniformly random samples in all series. Series: jVMC_exp with the dense Jacobian, jVMC_exp
with the Jacobian in batches of --batch-size samples, NetKet VMC_SR with the NTK (MinSR), and the same
with the NTK built on the fly in batches. Every point runs in its own process, and a series stops at its
first out-of-memory point. Every point writes <series>_<num_hidden>.json (times, peak memory, versions)
and its update as update_<series>_<num_hidden>.npy, which checks that all series give the same update.

The results in Data/ were made on NVIDIA GH200 GPUs (jVMC_exp 1.5.8, jax 0.11.2, netket 3.22.4) through
tests/benchmark/netket_comparison/send_job.sh, with 4 GPUs for the second run. Both runs used these
library settings: tangent kernel summed in chunks of parameters (JVMC_MATMUL_CHUNK_BYTES, default 1 GiB),
compiled batch loop (JVMC_BATCH_LOOP=compiled), batched Jacobian on.

    python compare_netket_minsr.py --out-dir ../Data/compare_netket_minsr
    python compare_netket_minsr.py --all-devices --out-dir ../Data/compare_netket_minsr_4gpus
"""
import sys
sys.path.append(sys.path[0] + "/../..")
import argparse
import json
import os
import subprocess
import warnings
from datetime import datetime

import numpy as np

from compare_netket_sr import (
    measure, environment_info, save, netket_rbm, netket_operator, jvmc_hamiltonian,
    check_single_device, peak_memory, is_out_of_memory,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
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
parser.add_argument("--batch-size", type=int, default=4096 // 4,
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
parser.add_argument("--out-dir", default=None, help="Default: a new results_minsr_<date> folder in paper/Data")
parser.add_argument("--summary", metavar="RESULTS_DIR", default=None, help="Only print the summary of the results in RESULTS_DIR")
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

    check_single_device(cfg)
    L, n_samples = cfg["L"], cfg["n_samples"]
    batch_size = min(cfg["batch_size"] or n_samples, n_samples)
    params, samples = load_inputs(cfg, num_hidden, out_dir)
    samples = jnp.asarray(samples, dtype=jVMC_exp.global_defs.DT_SAMPLES)
    weights = jnp.full(n_samples, 1 / n_samples)

    hamiltonian = jvmc_hamiltonian(L, cfg["g"])
    psi = NQS(RBM(numHidden=num_hidden, bias=cfg["bias"]), L, batchSize=batch_size, seed=cfg["seed"])
    psi.parameters = jax.tree_util.tree_map(jnp.asarray, params)
    # Never samples: the samples are set below, where MinSR sets the ones it draws
    sampler = MCSampler(psi, SpinFlip(), key=cfg["seed"], numChains=jax.device_count(), numSamples=n_samples)
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

    check_single_device(cfg)
    L, n_samples = cfg["L"], cfg["n_samples"]
    params, samples = load_inputs(cfg, num_hidden, out_dir)

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
            sampler, netket_rbm(num_hidden, cfg["bias"], "float64"),
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
        mode="real", use_ntk=True, on_the_fly=on_the_fly,
    )
    # NetKet has no public way to set the samples: these are the ones MCState.sample() would
    # store, with the 0/1 spins of jVMC_exp as +-1
    vstate._samples = jnp.asarray(2 * samples - 1, dtype=sampler.dtype).reshape(n_devices, -1, L)
    if n_devices > 1:
        vstate._samples = jax.device_put(vstate._samples, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("S")))

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
# Summary
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

def print_summary(out_dir):
    points = load_points(out_dir)
    if not points:
        raise SystemExit(f"No results in {out_dir}")
    differences = update_differences(out_dir, points)

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
        # Only the batched Jacobian and the NTK on the fly use the batch size: the other
        # series evaluate all samples at once
        if args.worker not in ("jvmc-batched", "netket-onthefly"):
            cfg["batch_size"] = None
        run_worker(cfg, args.worker, args.n, args.out_dir)
        return

    if args.summary is not None:
        print_summary(args.summary)
        return

    cfg = {
        "L": args.L, "g": args.g, "num_hidden": sorted(args.num_hidden), "bias": args.bias,
        "n_samples": args.n_samples, "batch_size": args.batch_size, "all_devices": args.all_devices,
        "diag_shift": args.diag_shift, "pinv_cutoff": args.pinv_cutoff,
        "repeats": args.repeats, "warmup": args.warmup, "seed": args.seed,
        "netket_operator": args.netket_operator,
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    out_dir = args.out_dir or os.path.join(DATA_DIR, f"results_minsr_{datetime.now():%Y%m%d_%H%M%S}")
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
    if not args.all_devices:
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

    print_summary(out_dir)

if __name__ == "__main__":
    main()
