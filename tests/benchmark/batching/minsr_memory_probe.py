"""
Where does the extra GPU memory of the dense minSR step on `sharded-batched-update` come from?
Run once per branch on the same GPUs, ideally with XLA_FLAGS=--xla_gpu_autotune_level=0.
Sizes can be lowered with PROBE_SAMPLES_PER_DEVICE / PROBE_BATCH_PER_DEVICE / PROBE_L / PROBE_HIDDEN.
"""
import os
import warnings
import jax
import jax.numpy as jnp
import jaxlib
import jVMC_exp
import jVMC_exp.nets as nets
from jVMC_exp.vqs import NQS
from jVMC_exp.stats import SampledObs
from jVMC_exp.sharding_config import DEVICE_SHARDING
from jVMC_exp.objective_function.base import ObjectiveFunctionOutput

def mem(tag):
    stats = [d.memory_stats() for d in jax.local_devices()]
    if stats[0] is None:
        print(f"  {tag:34s} (no memory statistics on this platform)")
        return
    use = max(s["bytes_in_use"] for s in stats) / 2 ** 20
    peak = max(s["peak_bytes_in_use"] for s in stats) / 2 ** 20
    print(f"  {tag:34s} in use {use:9.1f} MiB   peak so far {peak:9.1f} MiB   (max over devices)")

def describe(name, x):
    y = jax.device_put(x, DEVICE_SHARDING)
    same_buffer = y.addressable_shards[0].data.unsafe_buffer_pointer() == x.addressable_shards[0].data.unsafe_buffer_pointer()
    print(f"  {name:34s} {type(x.sharding).__name__} {x.sharding}\n"
          f"  {'':34s} == DEVICE_SHARDING: {x.sharding == DEVICE_SHARDING} | "
          f"device_put(x, DEVICE_SHARDING): same object {y is x}, same buffer {same_buffer}")

size = lambda name, default: int(os.environ.get(name, default))
D = jax.device_count()
L, H = size("PROBE_L", 100), size("PROBE_HIDDEN", 400)
N, B = size("PROBE_SAMPLES_PER_DEVICE", 1024) * D, size("PROBE_BATCH_PER_DEVICE", 256) * D

print("jax", jax.__version__, "| jaxlib", jaxlib.__version__, "| jVMC_exp from", os.path.dirname(jVMC_exp.__file__))
print("devices:", jax.devices())
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"))

psi = NQS(nets.CpxRBM(numHidden=H, bias=True), L, B, seed=1234)
s = jax.device_put(jax.random.randint(jax.random.PRNGKey(0), (N, L), 0, 2).astype(jVMC_exp.global_defs.DT_SAMPLES), DEVICE_SHARDING)
w = jax.random.uniform(jax.random.PRNGKey(1), (N,))
o_loc = SampledObs(jax.random.normal(jax.random.PRNGKey(2), (N,)) + 1j * jax.random.normal(jax.random.PRNGKey(3), (N,)), w)
opt = jVMC_exp.optimizer.MinSR(
    jVMC_exp.sampler.MCSampler(psi, jVMC_exp.propose.SpinFlip(), 123, D, N), psi,
    solver=jVMC_exp.solver.Pinv(pinv_cutoff=1e-8), diagonalShift=1e-3
)
print(f"N={N}, batch={B}, parameters={psi.numParameters}, Jacobian per device {N * psi.numParameters * 16 / D / 2 ** 20:.1f} MiB")

print("\n1) In-place assembly of the batched output (branch only)")
try:
    from jVMC_exp.sharding_config import BatchLayout
except ImportError:
    print("  no BatchLayout: this is master, skipped")
else:
    layout = BatchLayout(N, B)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        piece = psi._gradients_sh(layout.take(s, 0), parameters=psi.grad_parameters, batch_size=None)
        describe("batch result (piece)", piece)
        out = layout.alloc(piece)
        describe("preallocated output", out)
        before = out.addressable_shards[0].data.unsafe_buffer_pointer()
        out = jax.block_until_ready(layout.put(out, piece, 0))
        print(f"  {'put: output buffer reused':34s} {out.addressable_shards[0].data.unsafe_buffer_pointer() == before}")
        describe("output after put", out)
        del out, piece
    for warning in caught:
        print(f"  WARNING: {warning.message}")

print("\n2) One dense minSR step, phase by phase (the first call also compiles)")
mem("before the step")
g = jax.block_until_ready(psi.gradients(s))
mem("after psi.gradients")
describe("psi.gradients output", g)
grad = SampledObs(g, w)
mem("after SampledObs(...)")
describe("SampledObs._observations", grad._observations)
print(f"  {'SampledObs kept the same array':34s} {grad._observations is g}")
del g
mem("after del of the gradients")
normalized = jax.block_until_ready(grad._normalized_obs)
mem("after _normalized_obs")
describe("normalized Jacobian", normalized)
del normalized
update = jax.block_until_ready(opt.get_update(ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad)))
mem("after get_update")