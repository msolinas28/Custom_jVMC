"""
jVMC_exp vs NetKet, in the spirit of Fig. 4 of arXiv:2108.03409: time spent in the
stages of the Monte Carlo estimate of <H> for the transverse-field Ising chain

    H = - sum_i sz_i sz_{i+1} - g sum_i sx_i      (periodic boundaries)

with a complex RBM (``jVMC_exp.nets.CpxRBM``), on the same hardware.

    python compare_netket.py                         # runs both libraries, then plots
    python compare_netket.py --batch-size 65536      # change the memory budget
    python compare_netket.py --plot results_<date>   # only re-plots a finished run

The defaults are those of the paper, apart from the network (a CNN in the paper): N=80,
10^4 samples from 350 Markov chains, sweeps of N single spin flips and 20 thermalization
sweeps. The RBM has N hidden units and no bias by default.

Stages (as in the paper):
    * Sampling:        MCMC, thermalization included.
    * Matrix elements: connected configurations s' and <s|H|s'> of all the samples.
    * Evaluation:      network evaluations on s' and the local estimator, i.e. the time
                       of the full estimate of <H> on the stored samples minus the
                       matrix-element time, since both libraries compute the matrix
                       elements inside the same compiled function as the evaluation.
    * Total:           sampling + full estimate of <H>, both measured.

To make the comparison fair:
    * Each library runs in its own Python process, one after the other, so they do not
      share the JAX runtime, compilation caches or device memory. By default only one
      device is visible to both (``--all-devices`` uses every device, for both).
    * Both use double precision (complex128 parameters) and the same network: NetKet
      runs a batched port of ``jVMC_exp.nets.CpxRBM`` with the parameters of the
      jVMC_exp network, and the run fails if the two give different log(psi) on the
      same configurations. The energies and acceptance rates are compared as well.
    * Same Hamiltonian, number of samples, chains, sweep size, thermalization and update
      (single spin flip Metropolis); the chains are kept between calls in both.
      jVMC_exp evaluates log(psi) of the samples at the end of the sampling and reuses it
      for <H>, NetKet evaluates it in the estimate of <H>: the totals include both.
    * Same memory budget: at most ``--batch-size`` network evaluations per call, which is
      ``batchSize`` of the jVMC_exp NQS and ``chunk_size`` of the NetKet MCState.
      The default is the number of samples: with a larger batch size jVMC_exp pads the
      evaluation of log(psi) of the samples up to the batch size (see ``BatchLayout``),
      which makes its sampling stage slower.
    * The first call of every stage (which includes the jit compilation) and the warm-up
      calls are excluded. The stages are then timed one after the other as in a VMC step,
      ``--repeats`` times, waiting for the device every time (``jax.block_until_ready``).
      The figure shows the median and the interquartile range.
    * NetKet is timed with its default (JAX) ``Ising``, and with the generic
      ``LocalOperator``, which compares to the generic operators of jVMC_exp. The Numba
      operators (matrix elements on the CPU, as in the paper) are available with
      ``--netket-operators``.
"""
import sys
sys.path.append(sys.path[0] + "/../../..")
import argparse
import json
import os
import platform
import subprocess
import time
import warnings
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = ("jvmc", "netket")
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
parser.add_argument("--out-dir", default=None, help="Default: a new results_<date> folder next to this script")
parser.add_argument("--plot", metavar="RESULTS_DIR", default=None, help="Only plot the results in RESULTS_DIR")
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

    batch_size = cfg['batch_size'] or cfg["n_chains"]
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

    @jax.jit
    def conn_jax(operator, samples):
        return operator.get_conn_padded(samples.reshape(-1, L))

    def conn_numba(operator, samples):
        # As in MCState.expect: matrix elements on the CPU, then moved to the device
        return jax.device_put(operator.get_conn_padded(np.asarray(samples).reshape(-1, L)))

    operators = {kind: netket_operator(kind, L, cfg["g"]) for kind in cfg["netket_operators"]}
    stages = [("sampling", vstate.sample)]
    for kind, operator in operators.items():
        conn = conn_numba if kind.endswith("numba") else conn_jax
        stages += [
            (f"matrix_elements/{kind}", lambda o=operator, c=conn: c(o, vstate.samples)),
            # Reuses the stored samples
            (f"local_estimator/{kind}", lambda o=operator: vstate.expect(o)),
        ]
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
            "first_call": {stage: first_call[f"{stage}/{kind}"] for stage in ("matrix_elements", "local_estimator")}
                          | {"sampling": first_call["sampling"]},
            "times": {stage: times[f"{stage}/{kind}"] for stage in ("matrix_elements", "local_estimator")}
                     | {"sampling": times["sampling"]},
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

def load_series(out_dir):
    """
    Per-repetition times of every stage, for every library and operator.
    """
    results, series = {}, []
    for worker in WORKERS:
        path = os.path.join(out_dir, f"{worker}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            results[worker] = json.load(f)
        for variant in results[worker]["variants"].values():
            t = {k: np.asarray(v) for k, v in variant["times"].items()}
            series.append({
                **variant,
                "library": worker,
                # All stages are measured within the same repetition
                "stages": dict(zip(STAGE_LABELS, (
                    t["sampling"],
                    t["matrix_elements"],
                    t["local_estimator"] - t["matrix_elements"],
                    t["sampling"] + t["local_estimator"],
                ))),
            })

    return results, series

def print_summary(cfg, results, series):
    width = max(len(s["label"]) for s in series) + 2
    jvmc_total = next((np.median(s["stages"]["Total"]) for s in series if s["library"] == "jvmc"), np.nan)
    print(f"\nMedian over {cfg['repeats']} repetitions [s]")
    print(" " * width + "".join(f"{label:>17}" for label in STAGE_LABELS)
          + f"{'Total/jVMC_exp':>17}{'first calls':>17}")
    for s in series:
        total = np.median(s["stages"]["Total"])
        print(f"{s['label']:<{width}}" + "".join(f"{np.median(s['stages'][label]):>17.4f}" for label in STAGE_LABELS)
              + f"{total / jvmc_total:>17.2f}{sum(s['first_call'].values()):>17.2f}")

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

def plot(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    results, series = load_series(out_dir)
    if not series:
        raise SystemExit(f"No results in {out_dir}")
    print_summary(cfg, results, series)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": BASELINE,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_MUTED,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    jvmc = next((s for s in series if s["library"] == "jvmc"), None)
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(series))]
    x = np.arange(len(series))

    # One panel per stage, each with its own scale: the stages differ by orders of magnitude
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
            if stage == "Total" and jvmc is not None and s is not jvmc:
                label += f"\n({median[i] / np.median(jvmc['stages'][stage]):.2f}x)"
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

    info = next(iter(results.values()))
    versions = " | ".join(
        f"{name} {r['versions'][name]}" for name, r in
        [("jVMC_exp", results.get("jvmc")), ("netket", results.get("netket"))] if r is not None
    )
    jax_versions = sorted({r["versions"]["jax"] for r in results.values()})
    ratio_note = " (r x): NetKet time / jVMC_exp time." if jvmc is not None and len(series) > 1 else ""
    fig.text(
        0.01, 0.01,
        f"Transverse-field Ising chain, N = {cfg['L']}, g = {cfg['g']}. Complex RBM, {cfg['num_hidden']} hidden units"
        f"{' with bias' if cfg['bias'] else ''}, {info['n_parameters']} real parameters, complex128. "
        f"{cfg['n_samples']} samples, {cfg['n_chains']} chains, sweeps of {cfg['L']} spin flips, "
        f"{cfg['therm_sweeps']} thermalization sweeps.\n"
        f"At most {cfg['batch_size']} network evaluations per call. Evaluation = full estimate of <H> minus matrix elements. "
        f"Total = sampling + full estimate of <H>.{ratio_note}\n"
        f"Median of {cfg['repeats']} repetitions after compilation and {cfg['warmup']} warm-up runs, error bars: "
        f"interquartile range. {info['n_devices']} x {info['device']}. {versions} | jax {', '.join(jax_versions)}",
        fontsize=7.5, color=INK_SECONDARY, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"netket_comparison.{ext}"), dpi=200)
    print(f"\nFigure saved to {os.path.join(out_dir, 'netket_comparison.png')}")


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
        "netket_operators": args.netket_operators, "all_devices": args.all_devices,
        "date": datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = args.out_dir or os.path.join(HERE, f"results_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    # Results of an earlier run in the same folder, so that a failed run can't reuse them
    for name in ("jvmc.json", "netket.json", "reference.npz"):
        if os.path.exists(os.path.join(out_dir, name)):
            os.remove(os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # jVMC_exp first: NetKet takes its parameters
    for worker in [w for w in WORKERS if w in args.libraries]:
        python = args.netket_python if worker == "netket" else sys.executable
        print(f"=== {worker} ===", flush=True)
        run = subprocess.run(
            [python, "-u", os.path.abspath(__file__), "--worker", worker, "--out-dir", out_dir],
            env=child_env(args),
        )
        if run.returncode != 0:
            print(f"    FAILED ({worker}) -- continuing")

    plot(out_dir)

if __name__ == "__main__":
    main()
