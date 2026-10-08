"""
Second probe of the extra GPU memory of the minSR step on `sharded-batched-update`.

The first probe (minsr_memory_probe.py) measured one cold call, synchronised after every phase,
and found the same peak on both branches (4 Jacobians + kernel). The benchmark, which finds one
extra Jacobian on the branch, runs one compiling call plus `repeats` warm calls without any
synchronisation, with XLA_PYTHON_CLIENT_PREALLOCATE=false. This script runs exactly the
benchmark's step and loop, prints the memory after every call and can add synchronisation points:

    --sync none   the benchmark's step, unchanged
    --sync grad   wait for psi.gradients before building SampledObs (dense only)
    --sync obs    wait for SampledObs (observations and mean) before get_update (dense only)
    --sync step   wait for every tangent kernel and for the solver inside get_update
    --sync all    grad + obs + step

The peak memory counter cannot be reset, so every setting needs its own process,
see minsr_memory_probe2.sh.
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("dense", "lazy"), default="dense")
    parser.add_argument("--sync", choices=("none", "grad", "obs", "step", "all"), default="none")
    parser.add_argument("--lib", default=None, help="Directory containing the jVMC_exp package to use (default: the installed one)")
    parser.add_argument("--samples-per-device", type=int, default=1024)
    parser.add_argument("--batch-per-device", type=int, default=256)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.case == "lazy" and args.sync in ("grad", "obs", "all"):
        parser.error(f"--sync {args.sync} needs the dense Jacobian")

    if args.lib:
        # Bypass the import hook of an editable install, so that `lib` is used
        sys.meta_path[:] = [f for f in sys.meta_path if "__editable__" not in type(f).__module__]
        sys.path.insert(0, os.path.abspath(args.lib))
    import jax
    import jax.numpy as jnp
    import jaxlib
    import jVMC_exp
    import jVMC_exp.nets as nets
    from jVMC_exp.vqs import NQS
    from jVMC_exp.stats import SampledObs, LazySampledObs
    from jVMC_exp.sharding_config import DEVICE_SHARDING
    from jVMC_exp.objective_function.base import ObjectiveFunctionOutput

    jvmc_dir = os.path.dirname(jVMC_exp.__file__)
    # Read from .git/HEAD, git is not installed on the compute nodes
    try:
        with open(os.path.join(jvmc_dir, "..", ".git", "HEAD")) as f:
            branch = f.read().strip().removeprefix("ref: refs/heads/")
    except OSError:
        branch = "unknown branch"
    print("jax", jax.__version__, "| jaxlib", jaxlib.__version__, "| jVMC_exp from", jvmc_dir, f"({branch})")
    print("devices:", jax.devices())
    for name in ("XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_MEM_FRACTION"):
        print(f"{name}: {os.environ.get(name)}")
    print(f"case {args.case}, sync {args.sync}")

    # Same setup as the minSR cases of batching_benchmark.py
    D = jax.device_count()
    N, B, L = args.samples_per_device * D, args.batch_per_device * D, args.L
    psi = NQS(nets.CpxRBM(numHidden=args.hidden, bias=True), L, B, seed=1234)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    s = jax.device_put(jax.random.randint(k1, (N, L), 0, 2).astype(jVMC_exp.global_defs.DT_SAMPLES), DEVICE_SHARDING)
    weights = jax.random.uniform(k2, (N,))
    o_loc_values = jax.random.normal(k3, (N,)) + 1j * jax.random.normal(k4, (N,))
    sampler = jVMC_exp.sampler.MCSampler(psi, jVMC_exp.propose.SpinFlip(), 123, D, N)
    opt = jVMC_exp.optimizer.MinSR(sampler, psi, solver=jVMC_exp.solver.Pinv(pinv_cutoff=1e-8), diagonalShift=1e-3)
    o_loc = SampledObs(o_loc_values, weights)

    if args.sync in ("step", "all"):
        kernel, solver = opt._get_tangent_kernel, opt._solver
        opt._get_tangent_kernel = lambda *a: jax.block_until_ready(kernel(*a))
        opt._solver = lambda *a, **kw: jax.block_until_ready(solver(*a, **kw))

    def fn():
        if args.case == "lazy":
            grad = LazySampledObs(psi.lazy_gradients(s), weights)
        else:
            g = psi.gradients(s)
            # Waiting returns the same array, so no extra reference to the Jacobian is kept
            grad = SampledObs(jax.block_until_ready(g) if args.sync in ("grad", "all") else g, weights)
            del g
            if args.sync in ("obs", "all"):
                jax.block_until_ready((grad._observations, grad.mean))
        return opt.get_update(ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad))

    # Sizes per device: the holomorphic Jacobian has one column per real parameter (2 per complex one)
    num_columns = psi.numParameters * (1 if psi.realParams else 2)
    itemsize = jnp.dtype(psi.out_dtype).itemsize
    J, b, T = N // D * num_columns * itemsize, B // D * num_columns * itemsize, N * N // D * 16
    mib = lambda x: x / 2 ** 20
    print(f"N={N}, batch={B}, Jacobian columns={num_columns}; per device: Jacobian J = {mib(J):.1f} MiB, "
          f"batch b = {mib(b):.1f} MiB, kernel T = {mib(T):.1f} MiB")

    def stats():
        all_stats = [d.memory_stats() for d in jax.local_devices()]
        if any(x is None for x in all_stats):
            return None, None
        return [x["bytes_in_use"] for x in all_stats], [x["peak_bytes_in_use"] for x in all_stats]

    # Like the benchmark: memory above the level before the first call, maximum over devices
    baseline, setup_peak = stats()
    if baseline is None:
        print("no memory statistics on this platform: only checking that the step runs")
    else:
        print(f"before the loop: in use {mib(max(baseline)):.1f} MiB, setup peak {mib(max(setup_peak)):.1f} MiB\n")
    record, record_rep = max(setup_peak) if setup_peak else 0, None
    for rep in range(args.repeats + 1):            # the first call compiles
        out = jax.block_until_ready(fn())
        in_use, peak = stats()
        if in_use is not None:
            extra_use = max(u - b0 for u, b0 in zip(in_use, baseline))
            extra_peak = max(p - b0 for p, b0 in zip(peak, baseline))
            is_new = max(peak) > record
            if is_new:
                record, record_rep = max(peak), rep
            print(f"rep {rep}{' (compiles)' if rep == 0 else '           '}  in use {mib(extra_use):8.1f} MiB   "
                  f"peak {mib(extra_peak):8.1f} MiB = {extra_peak / J:5.2f} J{'   <- new peak' if is_new else ''}")
        if rep < args.repeats:
            del out

    if baseline is not None:
        print(f"\nRESULT case={args.case} sync={args.sync} prealloc={os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'default')} "
              f"peak={mib(extra_peak):.1f}MiB ({extra_peak / J:.2f}J, (peak-T)/J={(extra_peak - T) / J:.2f}) set_in_rep={record_rep}")

if __name__ == "__main__":
    main()
