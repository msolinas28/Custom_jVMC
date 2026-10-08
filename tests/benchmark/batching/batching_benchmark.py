"""
Benchmark of the sharded batching: wall time and peak device memory of the main batched
code paths, to compare two versions of jVMC_exp (e.g. master and a branch) on the same GPUs.

Every case runs in its own process, so that the peak memory of one case does not leak into
the next and a case that runs out of memory is recorded instead of stopping the benchmark.

Usage (single node, all visible GPUs are used):

    # The other version, e.g. master, as a separate checkout
    git worktree add ../jvmc_master master

    python batching_benchmark.py run --label branch --out branch.json
    python batching_benchmark.py run --label master --lib ../jvmc_master --out master.json
    python batching_benchmark.py compare master.json branch.json

The sizes scale with the number of devices (``--samples-per-device`` etc.), so the same
command stresses every device equally on 1, 2, 4 or 8 GPUs. Pass the same size arguments
to both runs.
"""
import argparse
import json
import os
import subprocess
import sys
import time

CASES = {
    "psi": "psi(s): small output, mainly batching overhead",
    "gradients": "psi.gradients(s): dense Jacobian, assembled from the batches",
    "o_loc": "H.get_O_loc(s, psi): local energies",
    "lazy_force": "LazySampledObs(psi.lazy_gradients(s)).get_covar(o_loc): batched Jacobian",
    "minsr_dense": "MinSR step with the dense Jacobian, holomorphic network",
    "minsr_lazy": "MinSR step with the batched Jacobian, holomorphic network",
    "minsr_dense_nonholo": "MinSR step with the dense Jacobian, non-holomorphic network",
    "minsr_lazy_nonholo": "MinSR step with the batched Jacobian, non-holomorphic network",
}

def _add_size_args(parser):
    parser.add_argument("--lib", default=None, help="Directory containing the jVMC_exp package to benchmark (default: the installed one)")
    parser.add_argument("--samples-per-device", type=int, default=2048)
    parser.add_argument("--batch-per-device", type=int, default=256)
    parser.add_argument("--minsr-samples-per-device", type=int, default=1024, help="MinSR builds an N x N kernel, so it uses fewer samples")
    parser.add_argument("--L", type=int, default=100, help="Number of spins (periodic chain)")
    parser.add_argument("--hidden", type=int, default=400, help="Hidden units of the RBMs")
    parser.add_argument("--repeats", type=int, default=5)

def _size_argv(args):
    return [
        f"--samples-per-device={args.samples_per_device}", f"--batch-per-device={args.batch_per_device}",
        f"--minsr-samples-per-device={args.minsr_samples_per_device}", f"--L={args.L}",
        f"--hidden={args.hidden}", f"--repeats={args.repeats}",
    ] + ([f"--lib={os.path.abspath(args.lib)}"] if args.lib else [])

# ----------------------------------------------------------------------------------------
# One case, in a child process
# ----------------------------------------------------------------------------------------
def _import_jvmc(lib):
    if lib:
        # Bypass the import hook of an editable install, so that `lib` is used
        sys.meta_path[:] = [f for f in sys.meta_path if "__editable__" not in type(f).__module__]
        sys.path.insert(0, os.path.abspath(lib))
    import jVMC_exp
    return jVMC_exp

def _memory_in_use():
    import jax
    stats = [d.memory_stats() for d in jax.local_devices()]
    if any(s is None for s in stats):
        return None, None
    return [s["bytes_in_use"] for s in stats], [s.get("peak_bytes_in_use", 0) for s in stats]

def _run_case(case, args):
    jVMC_exp = _import_jvmc(args.lib)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import flax.linen as nn
    import jVMC_exp.nets as nets
    import jVMC_exp.operator.discrete as op
    from jVMC_exp.vqs import NQS
    from jVMC_exp.stats import SampledObs, LazySampledObs
    from jVMC_exp.sharding_config import DEVICE_SHARDING
    from jVMC_exp.objective_function.base import ObjectiveFunctionOutput

    class NonHolomorphicRBM(nn.Module):
        """Real parameters, complex output"""
        hidden: int

        @nn.compact
        def __call__(self, s):
            x = nn.Dense(self.hidden)(2 * s.ravel() - 1)
            return jnp.sum(jnp.log(jnp.cosh(x))) + 1j * jnp.sum(nn.Dense(1)(jnp.tanh(x)))

    D = jax.device_count()
    is_minsr = case.startswith("minsr")
    N = (args.minsr_samples_per_device if is_minsr else args.samples_per_device) * D
    B = args.batch_per_device * D
    L = args.L

    net = NonHolomorphicRBM(args.hidden) if case.endswith("nonholo") else nets.CpxRBM(numHidden=args.hidden, bias=True)
    psi = NQS(net, L, B, seed=1234)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    s = jax.device_put(jax.random.randint(k1, (N, L), 0, 2).astype(jVMC_exp.global_defs.DT_SAMPLES), DEVICE_SHARDING)
    weights = jax.random.uniform(k2, (N,))
    o_loc_values = jax.random.normal(k3, (N,)) + 1j * jax.random.normal(k4, (N,))

    if case == "psi":
        fn = lambda: psi(s)
        summary = lambda out: out
    elif case == "gradients":
        fn = lambda: psi.gradients(s)
        summary = lambda out: jnp.concatenate([jnp.sum(jnp.abs(out) ** 2, axis=0), out[:8].ravel()])
    elif case == "o_loc":
        H = sum(-1.0 * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) - 0.7 * op.SigmaX(l) for l in range(L))
        fn = lambda: H.get_O_loc(s, psi)
        summary = lambda out: out
    elif case == "lazy_force":
        o_loc = SampledObs(o_loc_values, weights)
        fn = lambda: LazySampledObs(psi.lazy_gradients(s), weights).get_covar(o_loc)
        summary = lambda out: out
    else:
        lazy = "_lazy" in case
        sampler = jVMC_exp.sampler.MCSampler(psi, jVMC_exp.propose.SpinFlip(), 123, D, N)
        opt = jVMC_exp.optimizer.MinSR(sampler, psi, solver=jVMC_exp.solver.Pinv(pinv_cutoff=1e-8), diagonalShift=1e-3)
        o_loc = SampledObs(o_loc_values, weights)

        def fn():
            grad = LazySampledObs(psi.lazy_gradients(s), weights) if lazy else SampledObs(psi.gradients(s), weights)
            return opt.get_update(ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad))
        summary = lambda out: out

    baseline, _ = _memory_in_use()
    times = []
    for rep in range(args.repeats + 1):                  # the first call compiles
        start = time.perf_counter()
        out = jax.block_until_ready(fn())
        times.append(time.perf_counter() - start)
        if rep < args.repeats:
            del out
    _, peak = _memory_in_use()

    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, f"{case}.npy"), np.asarray(summary(out)))
    leaves = jax.tree_util.tree_leaves(out)
    result = {
        "devices": D,
        "device_kind": jax.devices()[0].device_kind,
        "jvmc_path": os.path.dirname(jVMC_exp.__file__),
        "num_samples": N,
        "batch_size": B,
        "num_parameters": int(psi.numParameters),
        "first_call_s": times[0],
        "median_s": float(np.median(times[1:])),
        "min_s": float(np.min(times[1:])),
        "output_mib": sum(x.size * x.dtype.itemsize for x in leaves) / 2 ** 20,
    }
    if baseline is not None:
        extra = [p - b for p, b in zip(peak, baseline)]
        result["peak_extra_mib_max"] = max(extra) / 2 ** 20
        result["peak_extra_mib_mean"] = sum(extra) / len(extra) / 2 ** 20
    print("RESULT " + json.dumps(result), flush=True)

# ----------------------------------------------------------------------------------------
# Driver and comparison
# ----------------------------------------------------------------------------------------
def _run(args):
    save_dir = os.path.splitext(os.path.abspath(args.out))[0] + "_arrays"
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    results = {"label": args.label, "lib": args.lib, "save_dir": save_dir, "cases": {}}
    for case in args.cases:
        print(f"[{args.label}] {case}: {CASES[case]}", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "_case", case, f"--save-dir={save_dir}"] + _size_argv(args)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
        if proc.returncode == 0 and lines:
            results["cases"][case] = json.loads(lines[-1][len("RESULT "):])
            r = results["cases"][case]
            mem = f", peak +{r['peak_extra_mib_max']:.0f} MiB/device" if "peak_extra_mib_max" in r else ""
            print(f"    {r['median_s'] * 1e3:.1f} ms{mem}", flush=True)
        else:
            error = (proc.stderr.strip().splitlines() or ["exit code %d" % proc.returncode])[-1]
            results["cases"][case] = {"error": error[:300]}
            print(f"    FAILED: {error[:300]}", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

def _compare(args):
    import numpy as np
    with open(args.baseline) as f:
        base = json.load(f)
    with open(args.candidate) as f:
        cand = json.load(f)
    a, b = base["label"], cand["label"]
    print(f"{'case':22s} {'time ' + a:>13s} {'time ' + b:>13s} {'speedup':>8s} "
          f"{'mem/dev ' + a:>15s} {'mem/dev ' + b:>15s} {'ratio':>7s}  max rel. diff")
    for case in CASES:
        if case not in base["cases"] or case not in cand["cases"]:
            continue
        rb, rc = base["cases"][case], cand["cases"][case]
        if "error" in rb or "error" in rc:
            status = lambda r: "FAILED" if "error" in r else f"{r['median_s'] * 1e3:.1f} ms"
            print(f"{case:22s} {status(rb):>13s} {status(rc):>13s}")
            continue
        speedup = rb["median_s"] / rc["median_s"]
        mem = lambda r: f"{r['peak_extra_mib_max']:.0f} MiB" if "peak_extra_mib_max" in r else "n/a"
        ratio = (f"{rc['peak_extra_mib_max'] / max(rb['peak_extra_mib_max'], 1e-9):.2f}"
                 if "peak_extra_mib_max" in rb and "peak_extra_mib_max" in rc else "n/a")
        xa = np.load(os.path.join(base["save_dir"], f"{case}.npy"))
        xb = np.load(os.path.join(cand["save_dir"], f"{case}.npy"))
        diff = np.max(np.abs(xa - xb)) / max(np.max(np.abs(xa)), 1e-300) if xa.shape == xb.shape else float("nan")
        print(f"{case:22s} {rb['median_s'] * 1e3:10.1f} ms {rc['median_s'] * 1e3:10.1f} ms {speedup:8.2f} "
              f"{mem(rb):>15s} {mem(rc):>15s} {ratio:>7s}  {diff:.1e}")
    first = next(iter(cand["cases"].values()))
    if "devices" in first:
        print(f"\n{first['devices']} x {first['device_kind']}; memory = peak device memory above the level "
              "before the first call, maximum over devices.")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Benchmark one version of jVMC_exp")
    run.add_argument("--label", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    _add_size_args(run)

    compare = sub.add_parser("compare", help="Compare two result files")
    compare.add_argument("baseline")
    compare.add_argument("candidate")

    case = sub.add_parser("_case")
    case.add_argument("case", choices=list(CASES))
    case.add_argument("--save-dir", required=True)
    _add_size_args(case)

    args = parser.parse_args()
    {"run": _run, "compare": _compare, "_case": lambda a: _run_case(a.case, a)}[args.command](args)

if __name__ == "__main__":
    main()
