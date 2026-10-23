"""
Time of the two stages of a Monte Carlo estimate of <H> in jVMC_exp and NetKet: sampling and expect <H>.

Copy of the data part of tests/benchmark/netket_comparison/compare_netket.py, without the matrix-element stage.
The figure is made by paper/plot_netket_comparison.py.

Transverse-field Ising chain (PBC), complex RBM, complex128, the same parameters in both libraries.
Stages, timed in this order in every repetition:
    sampling    MCMC sampling (single-spin-flip Metropolis, thermalization included)
    expect      full estimate of <H> on the samples of the sampling stage: matrix elements,
                network evaluations, mean and error (jVMC_exp: sampler(H), NetKet: vstate.expect(H))
Each library runs in its own process and writes <library>.json (times, energy, versions). jVMC_exp runs first
and writes reference.npz, the parameters NetKet loads and log(psi) on a few configurations to check that the
two networks agree.

Run through tests/benchmark/netket_comparison/send_job.sh, the results go to paper/Data/compare_netket:

    python compare_netket.py
    python compare_netket.py --summary ../Data/compare_netket
"""
import sys
sys.path.append(sys.path[0] + "/../..")
import argparse
import json
import os
os.environ.setdefault("JVMC_BATCH_LOOP", "compiled")
import platform
import subprocess
import time
import warnings
from datetime import datetime

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
WORKERS = ("jvmc", "netket")
STAGES = ("sampling", "expect")
STAGE_LABELS = {"sampling": "Sampling", "expect": "Expect <H>"}
NETKET_OPERATORS = {
    "ising": "NetKet, Ising (JAX)",
    "local": "NetKet, LocalOperator (JAX)",
    "ising-numba": "NetKet, Ising (Numba)",
    "local-numba": "NetKet, LocalOperator (Numba)",
}
N_CHECK_CONFIGS = 64

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--L", type=int, default=80, help="Number of spins")
parser.add_argument("--g", type=float, default=1.0, help="Transverse field")
parser.add_argument("--num-hidden", type=int, default=None, help="Hidden units of the RBM (default: L)")
parser.add_argument("--bias", action="store_true", help="Use the bias of the hidden units")
parser.add_argument("--n-samples", type=int, default=2**12,
                    help="Rounded up to a multiple of the number of chains, as both libraries do")
parser.add_argument("--n-chains", type=int, default=2**10)
parser.add_argument("--therm-sweeps", type=int, default=20,
                    help="Thermalization sweeps at every sampling (n_discard_per_chain in NetKet)")
parser.add_argument("--batch-size", type=int, default=None,
                    help="Maximum number of network evaluations per call in both libraries "
                         "(default: the number of samples)")
parser.add_argument("--repeats", type=int, default=10, help="Timed repetitions of every stage")
parser.add_argument("--warmup", type=int, default=2, help="Untimed repetitions after the first call")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--libraries", nargs="+", choices=WORKERS, default=list(WORKERS))
parser.add_argument("--netket-operators", nargs="+", choices=list(NETKET_OPERATORS), default=["ising", "local"])
parser.add_argument("--all-devices", action="store_true",
                    help="Run both libraries on all visible devices instead of on a single one")
parser.add_argument("--netket-python", default=sys.executable,
                    help="Python interpreter of the NetKet run, if NetKet lives in another environment")
parser.add_argument("--out-dir", default=None, help="Default: paper/Data/compare_netket")
parser.add_argument("--summary", metavar="RESULTS_DIR", default=None, help="Only print the summary of the results in RESULTS_DIR")
parser.add_argument("--worker", choices=WORKERS, default=None, help=argparse.SUPPRESS)


# ----------------------------------------------------------------------------------------
# Shared by the two workers
# ----------------------------------------------------------------------------------------

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

def check_devices(cfg):
    import jax

    n_devices = jax.device_count()
    if not cfg["all_devices"] and n_devices != 1:
        raise SystemExit(f"{n_devices} devices are visible, but the comparison runs on one (see --all-devices)")
    if cfg["n_chains"] % n_devices != 0:
        raise SystemExit(f"The number of chains ({cfg['n_chains']}) has to be divisible by "
                         f"the number of devices ({n_devices})")

    return n_devices

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


# ----------------------------------------------------------------------------------------
# jVMC_exp
# ----------------------------------------------------------------------------------------

def jvmc_hamiltonian(L, g):
    import jVMC_exp.operator.discrete as op

    hamiltonian = 0
    for i in range(L):
        hamiltonian += -1.0 * op.SigmaZ(i) * op.SigmaZ((i + 1) % L) - g * op.SigmaX(i)

    return hamiltonian

def run_jvmc(cfg, out_dir):
    from flax.traverse_util import flatten_dict
    import jVMC_exp
    from jVMC_exp.nets import CpxRBM
    from jVMC_exp.sampler import MCSampler
    from jVMC_exp.sampler.propose import SpinFlip
    from jVMC_exp.vqs import NQS

    L = cfg["L"]
    hamiltonian = jvmc_hamiltonian(L, cfg["g"])

    batch_size = cfg['batch_size'] or cfg["n_samples"]
    net = CpxRBM(numHidden=cfg["num_hidden"], bias=cfg["bias"])
    psi = NQS(net, L, batchSize=batch_size, seed=cfg["seed"])
    sampler = MCSampler(
        psi, SpinFlip(), key=cfg["seed"],
        numChains=cfg["n_chains"], numSamples=cfg["n_samples"],
        thermalizationSweeps=cfg["therm_sweeps"], sweepSteps=L
    )

    def expect():
        # Reuses the stored samples and their log(psi)
        energy = sampler(hamiltonian)
        return energy.mean, energy.error_of_mean

    first_call, times = measure([
        ("sampling", sampler.sample),
        ("expect", expect),
    ], cfg["warmup"], cfg["repeats"])

    mean, error = expect()

    # Parameters and reference values of log(psi), to run the same network in NetKet
    configs = np.asarray(sampler.samples[:N_CHECK_CONFIGS])
    np.savez(
        os.path.join(out_dir, "reference.npz"),
        configs=configs, log_psi=np.asarray(psi(configs)),
        **{"params/" + "/".join(k): np.asarray(v) for k, v in flatten_dict(psi.params).items()},
    )

    n_params = psi.numParameters if psi.realParams else 2 * psi.numParameters
    save(out_dir, "jvmc", {
        "library": "jVMC_exp",
        **environment_info(jVMC_exp=jVMC_exp.__version__),
        "n_parameters": n_params,
        "batch_size": batch_size,
        "sample_devices": len(sampler.samples.sharding.device_set),
        "variants": {
            "jvmc": {
                "label": "jVMC_exp",
                "operator": "generic operator strings (SigmaZ, SigmaX)",
                # The diagonal uses log(psi) of the samples from the sampling
                "evaluations_per_sample": hamiltonian.n_conn,
                "first_call": first_call,
                "times": times,
                "energy": [float(np.real(np.asarray(mean).ravel()[0])), float(np.asarray(error).ravel()[0])],
                "acceptance": float(np.asarray(sampler.acceptance_ratio()).ravel()[0]),
            }
        },
    })


# ----------------------------------------------------------------------------------------
# NetKet
# ----------------------------------------------------------------------------------------

def netket_rbm(num_hidden, bias, dtype="complex128"):
    """
    ``jVMC_exp.nets.CpxRBM`` (or ``jVMC_exp.nets.RBM`` with ``dtype="float64"``) acting on
    a batch of configurations.

    NetKet spins are already +-1, which is ``2 * s - 1`` of the 0/1 jVMC_exp spins.
    The layer has the same name as in jVMC_exp, so the two have the same parameter tree.
    """
    import jax
    import jax.numpy as jnp
    import flax.linen as nn

    def log_cosh(x):
        two = jnp.asarray(2, dtype=x.real.dtype)
        sgn_x = (-two * jnp.signbit(x.real) + 1).astype(x.dtype)
        x = x * sgn_x
        return x + jnp.log1p(jnp.exp(-two * x)) - jnp.log(two)

    class RBM(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(
                num_hidden, use_bias=bias, param_dtype=jnp.dtype(dtype),
                kernel_init=jax.nn.initializers.normal(1e-2), bias_init=jax.nn.initializers.zeros,
            )(x)

            return jnp.sum(log_cosh(x), axis=-1)

    return RBM()

def netket_operator(kind, L, g):
    import netket as nk

    hilbert = nk.hilbert.Spin(0.5, L)
    graph = nk.graph.Chain(L, pbc=True)
    if kind.startswith("ising"):
        # -h sum_i sx_i + J sum_<ij> sz_i sz_j
        operator = nk.operator.IsingJax(hilbert, graph, h=g, J=-1.0)
    else:
        sz, sx = nk.operator.spin.sigmaz, nk.operator.spin.sigmax
        operator = -g * sx(hilbert, 0)
        for i in range(1, L):
            operator -= g * sx(hilbert, i)
        for i, j in graph.edges():
            operator -= sz(hilbert, i) @ sz(hilbert, j)
        if not isinstance(operator, nk.operator.DiscreteJaxOperator):
            operator = operator.to_jax_operator()

    return operator.to_numba_operator() if kind.endswith("numba") else operator

def run_netket(cfg, out_dir):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from flax.traverse_util import unflatten_dict
    import netket as nk

    if check_devices(cfg) > 1:
        # NetKet shards the samples over the mesh, if one is set
        jax.sharding.set_mesh(jax.sharding.Mesh(
            np.asarray(jax.devices()), ("S",), axis_types=(jax.sharding.AxisType.Auto,)
        ))
    L = cfg["L"]
    hilbert = nk.hilbert.Spin(0.5, L)
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=cfg["n_chains"], sweep_size=L)
    with warnings.catch_warnings():
        # The batch size is the same as in jVMC_exp, power of two or not
        warnings.filterwarnings("ignore", message=".*power-of-two chunk size")
        vstate = nk.vqs.MCState(
            sampler, netket_rbm(cfg["num_hidden"], cfg["bias"]),
            n_samples=cfg["n_samples"], n_discard_per_chain=cfg["therm_sweeps"],
            chunk_size=cfg["batch_size"], seed=cfg["seed"], sampler_seed=cfg["seed"],
        )

    log_psi_error = None
    reference = os.path.join(out_dir, "reference.npz")
    if os.path.exists(reference):
        ref = np.load(reference)
        vstate.parameters = unflatten_dict({
            tuple(k.split("/")[1:]): jnp.asarray(ref[k]) for k in ref.files if k.startswith("params/")
        })
        log_psi = vstate.log_value(jnp.asarray(2 * ref["configs"] - 1, dtype=sampler.dtype))
        log_psi_error = float(np.max(np.abs(np.asarray(log_psi) - ref["log_psi"])))
        if not log_psi_error < 1e-8:
            raise RuntimeError(f"The NetKet and jVMC_exp networks differ: max |delta log(psi)| = {log_psi_error}")
    else:
        print("WARNING: no jVMC_exp parameters found, NetKet runs with its own random parameters")

    operators = {kind: netket_operator(kind, L, cfg["g"]) for kind in cfg["netket_operators"]}
    stages = [("sampling", vstate.sample)]
    for kind, operator in operators.items():
        # Reuses the stored samples
        stages.append((f"expect/{kind}", lambda o=operator: vstate.expect(o)))
    first_call, times = measure(stages, cfg["warmup"], cfg["repeats"])

    if vstate.samples.reshape(-1, L).shape[0] != cfg["n_samples"]:
        raise RuntimeError(f"NetKet drew {vstate.samples.reshape(-1, L).shape[0]} samples instead of {cfg['n_samples']}")
    acceptance = float(vstate.sampler_state.acceptance)

    variants = {}
    for kind, operator in operators.items():
        energy = vstate.expect(operator)
        variants[kind] = {
            "label": NETKET_OPERATORS[kind],
            "operator": type(operator).__name__,
            # All connected configurations (diagonal included) and the sample itself
            "evaluations_per_sample": int(operator.max_conn_size) + 1,
            "first_call": {"sampling": first_call["sampling"], "expect": first_call[f"expect/{kind}"]},
            "times": {"sampling": times["sampling"], "expect": times[f"expect/{kind}"]},
            "energy": [float(np.real(energy.mean)), float(energy.error_of_mean)],
            "acceptance": acceptance,
        }

    save(out_dir, "netket", {
        "library": "NetKet",
        **environment_info(netket=nk.__version__),
        "n_parameters": int(sum(2 * p.size if jnp.iscomplexobj(p) else p.size
                                for p in jax.tree_util.tree_leaves(vstate.parameters))),
        "batch_size": cfg["batch_size"],
        "sample_devices": len(vstate.samples.sharding.device_set),
        "max_log_psi_difference": log_psi_error,
        "variants": variants,
    })


# ----------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------

def print_summary(out_dir):
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    results = {}
    for worker in WORKERS:
        path = os.path.join(out_dir, f"{worker}.json")
        if os.path.exists(path):
            with open(path) as f:
                results[worker] = json.load(f)
    series = [{**variant, "library": worker} for worker, r in results.items() for variant in r["variants"].values()]
    if not series:
        raise SystemExit(f"No results in {out_dir}")

    width = max(len(s["label"]) for s in series) + 2
    jvmc = next((s for s in series if s["library"] == "jvmc"), None)
    print(f"\nMedian over {cfg['repeats']} repetitions [s], ratio to jVMC_exp in brackets")
    print(" " * width + "".join(f"{STAGE_LABELS[stage]:>24}" for stage in STAGES) + f"{'first calls':>17}")
    for s in series:
        row = f"{s['label']:<{width}}"
        for stage in STAGES:
            median = np.median(s["times"][stage])
            ratio = f" ({median / np.median(jvmc['times'][stage]):.2f}x)" if jvmc is not None else ""
            row += f"{f'{median:.4f}{ratio}':>24}"
        print(row + f"{sum(s['first_call'].values()):>17.2f}")

    print("\nChecks")
    print("  samples on " + ", ".join(f"{r['sample_devices']} device(s) in {r['library']}" for r in results.values()))
    if "netket" in results and results["netket"]["max_log_psi_difference"] is not None:
        print(f"  same network: max |delta log(psi)| on {N_CHECK_CONFIGS} configurations = "
              f"{results['netket']['max_log_psi_difference']:.1e}")
    reference = series[0]
    for s in series:
        line = (f"  {s['label']:<{width}} E/N = {s['energy'][0] / cfg['L']:+.6f} +- {s['energy'][1] / cfg['L']:.6f}"
                f"   acceptance = {s['acceptance']:.3f}   network evaluations per sample in <H>: "
                f"{s['evaluations_per_sample']}")
        if s is not reference:
            z = abs(s["energy"][0] - reference["energy"][0]) / np.hypot(s["energy"][1], reference["energy"][1])
            line += f"   |delta E| = {z:.1f} sigma"
        print(line)


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------

def child_env(args):
    env = dict(os.environ)
    # The comparison is a single process per library
    env["JVMC_USE_DISTRIBUTED"] = "false"
    if not args.all_devices:
        # First of the devices given to this job, or device 0
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    return env

def main():
    args = parser.parse_args()

    if args.worker is not None:
        with open(os.path.join(args.out_dir, "config.json")) as f:
            cfg = json.load(f)
        {"jvmc": run_jvmc, "netket": run_netket}[args.worker](cfg, args.out_dir)
        return

    if args.summary is not None:
        print_summary(args.summary)
        return

    n_samples = -(-args.n_samples // args.n_chains) * args.n_chains
    if n_samples != args.n_samples:
        print(f"Number of samples rounded up to a multiple of the number of chains: {args.n_samples} -> {n_samples}")
    cfg = {
        "L": args.L, "g": args.g, "num_hidden": args.num_hidden or args.L, "bias": args.bias,
        "n_samples": n_samples, "n_chains": args.n_chains, "therm_sweeps": args.therm_sweeps,
        "batch_size": args.batch_size or n_samples,
        "repeats": args.repeats, "warmup": args.warmup, "seed": args.seed,
        "netket_operators": args.netket_operators, "all_devices": args.all_devices,
        "date": datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = os.path.abspath(args.out_dir or os.path.join(DATA_DIR, "compare_netket"))
    os.makedirs(out_dir, exist_ok=True)
    for name in ("jvmc.json", "netket.json", "reference.npz"):
        if os.path.exists(os.path.join(out_dir, name)):
            os.remove(os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for worker in [w for w in WORKERS if w in args.libraries]:
        python = args.netket_python if worker == "netket" else sys.executable
        print(f"=== {worker} ===", flush=True)
        run = subprocess.run(
            [python, "-u", os.path.abspath(__file__), "--worker", worker, "--out-dir", out_dir],
            env=child_env(args),
        )
        if run.returncode != 0:
            print(f"    FAILED ({worker}) -- continuing")

    print_summary(out_dir)

if __name__ == "__main__":
    main()
