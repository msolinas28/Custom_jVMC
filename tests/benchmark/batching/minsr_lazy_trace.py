"""
Call-by-call memory trace of one lazy minSR step (holomorphic network, same setup as
batching_benchmark.py), to find where `sharded-batched-update` uses ~370 MiB more than master.
Run it once on each branch, on the same GPUs, and compare the two outputs line by line.

After every traced call it waits for the GPU and prints, per device (maximum over devices):
    in use  device memory in use
    peak    peak device memory so far, with its increase since the previous line
    held    memory of the arrays that Python still references
    other   in use - held: memory that no Python array owns, e.g. what the allocator adds
            when it hands out a block larger than the one requested
followed by the arrays of at least --min-mib MiB per device (default 8) that Python references.

If the branch holds more at its peak, "held" and the list of arrays show which array it is.
If it holds the same but "other" is larger, the extra comes from how the allocator places buffers.
"""
import argparse
import collections
import dataclasses
import math
import os
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lib", default=None, help="Directory containing the jVMC_exp package to use (default: the installed one)")
    parser.add_argument("--samples-per-device", type=int, default=1024)
    parser.add_argument("--batch-per-device", type=int, default=256)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=400)
    parser.add_argument("--min-mib", type=float, default=8, help="List the referenced arrays of at least this many MiB per device")
    args = parser.parse_args()

    if args.lib:
        # Bypass the import hook of an editable install, so that `lib` is used
        sys.meta_path[:] = [f for f in sys.meta_path if "__editable__" not in type(f).__module__]
        sys.path.insert(0, os.path.abspath(args.lib))
    import jax
    import jax.numpy as jnp
    import jVMC_exp
    import jVMC_exp.nets as nets
    import jVMC_exp.optimizer.minsr as minsr_module
    import jVMC_exp.sharding_config as sharding_module
    import jVMC_exp.stats as stats_module
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
    print("jax", jax.__version__, "| jVMC_exp from", jvmc_dir, f"({branch})")
    for name in ("XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_ALLOCATOR"):
        print(f"{name}: {os.environ.get(name)}")

    # Same setup as the minsr_lazy case of batching_benchmark.py
    D = jax.device_count()
    N, B, L = args.samples_per_device * D, args.batch_per_device * D, args.L
    n_batches = math.ceil(N / B)
    psi = NQS(nets.CpxRBM(numHidden=args.hidden, bias=True), L, B, seed=1234)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    s = jax.device_put(jax.random.randint(k1, (N, L), 0, 2).astype(jVMC_exp.global_defs.DT_SAMPLES), DEVICE_SHARDING)
    weights = jax.random.uniform(k2, (N,))
    o_loc_values = jax.random.normal(k3, (N,)) + 1j * jax.random.normal(k4, (N,))
    sampler = jVMC_exp.sampler.MCSampler(psi, jVMC_exp.propose.SpinFlip(), 123, D, N)
    opt = jVMC_exp.optimizer.MinSR(sampler, psi, solver=jVMC_exp.solver.Pinv(pinv_cutoff=1e-8), diagonalShift=1e-3)
    o_loc = SampledObs(o_loc_values, weights)

    num_columns = psi.numParameters * (1 if psi.realParams else 2)
    itemsize = jnp.dtype(psi.out_dtype).itemsize
    mib = lambda x: x / 2 ** 20
    print(f"N={N}, batch={B}; per device: one gradient batch = {mib(B // D * num_columns * itemsize):.1f} MiB, "
          f"kernel T = {mib(N * N // D * 16):.1f} MiB\n")

    short_dtype = {"complex128": "c128", "complex64": "c64", "float64": "f64", "float32": "f32"}
    short_spec = {"PartitionSpec('devices',)": "rows", "PartitionSpec()": "repl"}

    def device_bytes(x):
        try:
            return math.prod(x.sharding.shard_shape(x.shape)) * x.dtype.itemsize
        except Exception:
            return x.nbytes

    state = {"line": 0, "kernels": 0, "solved": False, "peak": 0}

    def record(label, out=None):
        # Wait for everything, so that only buffers that are really needed are left
        if out is not None:
            jax.block_until_ready(out)
        live = [x for x in jax.live_arrays() if not x.is_deleted()]
        for x in live:
            x.block_until_ready()
        held = sum(device_bytes(x) for x in live)
        big = collections.Counter(
            f"{x.shape}{short_dtype.get(str(x.dtype), str(x.dtype))}"
            f"{short_spec.get(repr(getattr(x.sharding, 'spec', '')), repr(getattr(x.sharding, 'spec', '')))}"
            for x in live if device_bytes(x) >= args.min_mib * 2 ** 20
        )
        del live
        arrays = " ".join(f"{n}x{key}" for key, n in sorted(big.items()))

        all_stats = [d.memory_stats() for d in jax.local_devices()]
        label = ("update: " if state["solved"] else "") + label
        if any(x is None for x in all_stats):
            print(f"{state['line']:3d} {label:30s} held {mib(held):8.1f} | {arrays}")
        else:
            in_use = max(x["bytes_in_use"] for x in all_stats)
            peak = max(x["peak_bytes_in_use"] for x in all_stats)
            jump = f"(+{mib(peak - state['peak']):6.1f})" if peak > state["peak"] else " " * 9
            state["peak"] = peak
            print(f"{state['line']:3d} {label:30s} in use {mib(in_use):7.1f}  peak {mib(peak):7.1f} {jump}  "
                  f"held {mib(held):7.1f}  other {mib(in_use - held):6.1f} | {arrays}")
        state["line"] += 1

    def traced(fn, label):
        def wrapper(*a, **kw):
            out = fn(*a, **kw)
            record(label, out)
            return out
        return wrapper

    # The gradient batches are produced lazily inside get_update. Through map, no reference
    # to a batch is kept after it is handed over
    lazy = psi.lazy_gradients(s)
    produce = lazy.reusable_iterable
    traced_batch = traced(lambda batch: batch, "gradient batch")
    lazy = dataclasses.replace(lazy, reusable_iterable=lambda: map(traced_batch, produce()))
    grad = LazySampledObs(lazy, weights)

    # Hooks, installed after the setup so that they only see the step
    stats_module._get_mean = traced(stats_module._get_mean, "mean: add batch")
    minsr_module._normalize_batch = traced(minsr_module._normalize_batch, "normalize batch")
    kernel, solver = opt._get_tangent_kernel, opt._solver

    def kernel_hook(*a):
        out = kernel(*a)
        l, r = divmod(state["kernels"], n_batches)
        state["kernels"] += 1
        record(f"kernel l={l} r={r}", out)
        return out

    def solver_hook(A, b, **kw):
        record("solver: start")
        out = solver(A, b, **kw)
        state["solved"] = True
        record("solver: done", out[0])
        return out

    opt._get_tangent_kernel, opt._solver = kernel_hook, solver_hook
    if hasattr(sharding_module, "BatchLayout"):          # branch: T written in place, solution split by the layout
        minsr_module._take_columns = traced(minsr_module._take_columns, "reorder columns of T row")
        sharding_module.BatchLayout.alloc = traced(sharding_module.BatchLayout.alloc, "allocate T")
        sharding_module.BatchLayout.put = traced(sharding_module.BatchLayout.put, "write T row")
        sharding_module.BatchLayout.split = traced(sharding_module.BatchLayout.split, "split solution")
    else:                                                 # master: T concatenated, solution split by _reshape_in_batches
        minsr_module._reshape_in_batches = traced(minsr_module._reshape_in_batches, "split solution")

    record("before get_update")
    update = opt.get_update(ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad))
    record("get_update returned", update)
    del update, grad
    record("after deleting the update")

    if state["peak"]:
        print(f"\nRESULT branch={branch} peak={mib(state['peak']):.1f}MiB (absolute, compare with the probe's peak + ~1 MiB baseline)")

if __name__ == "__main__":
    main()
