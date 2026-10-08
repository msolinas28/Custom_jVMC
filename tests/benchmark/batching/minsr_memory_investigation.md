# GPU memory regression of minSR on `sharded-batched-update`

## The problem

On 4× NVIDIA GH200 120GB, the branch `sharded-batched-update` uses **more device memory than
`master` for the minSR step with a holomorphic network**, although it uses much less for the
dense Jacobian itself:

| benchmark case | `master` | branch | difference |
|---|---|---|---|
| `minsr_dense` (holomorphic `CpxRBM`, dense Jacobian) | 5116 MiB | 6378 MiB | **+1262.5 MiB = exactly 2.000 per-device Jacobians** |
| `minsr_lazy` (holomorphic, batched Jacobian) | 2673 MiB | 3044 MiB | +370 MiB ≈ 2.35 per-device batches |
| `gradients` (`psi.gradients`, dense Jacobian only) | 7652 MiB | 3160 MiB | −4492 MiB (branch 2.4× better) |
| `minsr_dense_nonholo` (non-holomorphic) | 4610 MiB | 4485 MiB | −125 MiB |

Values: peak device memory above the level before the first call, maximum over the 4 GPUs.
Per-device sizes for the minSR cases (N = 4096 samples, 40400 parameters, complex128, 4
devices): Jacobian 631.25 MiB, one batch (1024 samples) 157.81 MiB, kernel T 64 MiB.

The `minsr_dense` difference is **exactly** two per-device Jacobians, and it is identical
(to the MiB) with and without GPU autotuning. So it is a deterministic extra buffer: two
Jacobian-sized arrays per GPU are alive at the peak on the branch and not on `master`.
**Find where they come from and fix it without giving up the branch's gains.**

The results are numerically the same on both branches (max. relative difference ≤ 3e-13), and
the problem **does not reproduce on CPU** (4 simulated devices, see below).

## What the branch changes

Compared with `master` (all in `jVMC_exp/`, line numbers on branch commit `92c5eed`):

- `sharding_config.py`: the `sharded` decorator (l. 301) now batches *locally*: batch `i`
  is the `i`-th block of `batch_size / n_devices` rows of every device's shard, so no data
  moves between devices. Helpers are module-level jitted functions on the view
  `(devices, rows per device, ...)`: `_take_batch` (l. 61), `_trim_batch` (l. 72),
  `_put_batch` (l. 76, `donate_argnums=0`), wrapped by `BatchLayout` (l. 84).
- **Non-lazy outputs are preallocated and written in place**: `_call_local` (l. 466)
  allocates the output once with `BatchLayout.alloc` (l. 159:
  `jnp.zeros(shape, dtype, device=piece.sharding)`, i.e. the sharding of the first batch
  result) and writes every batch with the donated `_put_batch`. `master` instead kept all
  batch results in a list and called `jnp.concatenate` (and replicated the inputs).
- Inputs are padded to a multiple of the number of devices (`pad_to_devices`, l. 50); the last
  batch of every device is padded to the batch size inside `_take_batch`.
- The `batch_size=None` path (l. 356) is unchanged: `jax.device_put(a, self.in_sharding[i])`
  then one call. minSR's `_get_single_tangent_kernel` / `_get_double_tangent_kernel`
  (`optimizer/minsr.py` l. 202/221) go through it.
- `stats.py`: `SampledObs.__init__` is unchanged and does
  `self._observations = jax.device_put(observations, DEVICE_SHARDING)` (l. 110).
  `LazySampledObs` (l. 291) splits weights etc. with the iterable's `BatchLayout`.
- `optimizer/minsr.py` `get_update` (l. 117): real/imaginary parts are interleaved instead
  of stacked (non-holomorphic case only, l. 133); the lazy kernel `T` is preallocated and
  filled in place (l. 156–157). The dense holomorphic path (l. 135, 170) is the same code as
  on `master`; it only receives a Jacobian produced by the new assembly.
- `vqs.py`: `psi.gradients` → `_gradients_sh` (l. 492) and `_lazy_gradients_sh` (l. 512)
  use `@sharded(automatic_sharding=True)`, i.e. `jax.jit` without `shard_map`.

## Measurements so far

GPU, 4× GH200, `batching_benchmark.py` defaults (8192 samples, batch 1024, minSR with 4096
samples), source hashes `master` `48822ade93c16a51`, branch `f1bbfde386288625`:

| case | time `master` → branch | memory `master` → branch (with autotuning) | memory without autotuning |
|---|---|---|---|
| `psi` | 26.6 → 15.6 ms | 131 → 131 MiB | – |
| `gradients` | 45.4 → 47.2 ms | 7652 → 3160 MiB | 7652 → 3160 MiB |
| `o_loc` | 5364 → 2722 ms | 132 → 131 MiB | – |
| `lazy_force` | 62.1 → 31.8 ms | 961 → 953 MiB | – |
| `minsr_dense` | 310 → 305 ms | 5116 → 6378 MiB | 5116 → 6378 MiB |
| `minsr_lazy` | 410 → 341 ms | 2685 → 3043 MiB | 2673 → 3044 MiB |
| `minsr_dense_nonholo` | 730 → 719 ms | 4617 → 4485 MiB | 4610 → 4485 MiB |
| `minsr_lazy_nonholo` | 841 → 713 ms | 3664 → 3664 MiB | – |

CPU, 4 simulated devices (`XLA_FLAGS=--xla_force_host_platform_device_count=4`), process
memory (VmRSS sampled every 2 ms) above the level before a warmed-up minSR step, Jacobian
1.02 GiB, kernel 0.25 GiB:

| | `master` | first local-batching commit `681b6eb` | branch |
|---|---|---|---|
| minSR dense | 6.36 GiB | 6.35 GiB | 6.35 GiB |
| minSR lazy | 4.99 GiB | 4.79 GiB | 4.78 GiB |

On CPU (JAX 0.10.1) every array involved is a plain
`NamedSharding(mesh, P('devices'), memory_kind=device)`, `jax.device_put(x, DEVICE_SHARDING)`
returns the same object and buffer, and the donated `_put_batch` reuses the output buffer.

## Ruled out

- **Compilation / GPU autotuning**: identical numbers with `XLA_FLAGS=--xla_gpu_autotune_level=0`.
- **Noise**: the `minsr_dense` gap is exactly 2.000 per-device Jacobians, identical across runs.
- **Output sharding on CPU**: the preallocated outputs are row-sharded (`P('devices')`) in
  `psi.gradients` and in lazy minSR's `T` (not yet checked on GPU).
- **The latest branch commits** (`592296d`, `92c5eed`): cosmetic in the decorator and
  `LazySampledObs.transform`, plus a `diag_scale` change in the SR optimizer (not minSR).

## Hypotheses, most likely first

1. **Extra copies by `jax.device_put`.** On GPU, the Jacobian built by the new assembly
   (`jnp.zeros(..., device=piece.sharding)` + donated `_put_batch`) or the arrays derived from
   it carry a sharding object that is not `== DEVICE_SHARDING` (different type, memory kind,
   or mesh object), so `device_put(x, DEVICE_SHARDING)` copies instead of returning `x`. Two
   such calls hold a Jacobian each while the original is still alive:
   `SampledObs.__init__` (`stats.py` l. 110) and the `batch_size=None` path of the decorator
   when `get_update` calls `_get_single_tangent_kernel(grad)` (`sharding_config.py` l. 358).
   This fits "exactly 2 Jacobians", fits `minsr_lazy` (two batch-sized copies in
   `_get_double_tangent_kernel(batch_l, batch_r)` ≈ 2 × 158 MiB), and fits the non-holomorphic
   case being unaffected (there `get_update` passes the fresh jit output of `_concat_nonholo`
   to the tangent kernel, not the Jacobian itself).
2. **Donation of `out` in `_put_batch` is not honoured on GPU** (JAX would warn "Some donated
   buffers were not usable", which the benchmark discards because each case runs in a
   subprocess with captured stderr). Hint: the `gradients` peak on the branch is 3160 MiB =
   2.5× the per-device output (1262.5 MiB), more than "output + one batch" would suggest.
3. Something else in the dense holomorphic path that keeps a reference to the original
   Jacobian on the branch only.

## Diagnostic to run on a GPU node

Save as `minsr_memory_probe.py` and run it once on each branch, same node and GPUs, from the
repository root (it imports the installed / editable `jVMC_exp`, so the checked-out branch is
what runs):

```bash
git checkout sharded-batched-update
XLA_FLAGS=--xla_gpu_autotune_level=0 python minsr_memory_probe.py > probe_branch.txt 2>&1
git checkout master
XLA_FLAGS=--xla_gpu_autotune_level=0 python minsr_memory_probe.py > probe_master.txt 2>&1
```

```python
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
```

The script has been checked on CPU against both branches (it runs; on CPU every
`device_put` returns the same buffer and `put` reuses the output buffer).

### How to read the output

- **Hypothesis 1** is confirmed if, on the branch but not on `master`, `psi.gradients output`
  (or `SampledObs._observations` / `normalized Jacobian`) shows `same object False` /
  `same buffer False`, or `SampledObs kept the same array False`, or a sharding type / memory
  kind different from `DEVICE_SHARDING`. Compare the printed sharding objects of the two
  branches.
- **Hypothesis 2** is confirmed by `put: output buffer reused False` or a
  "donated buffers were not usable" warning in section 1.
- In section 2, the phase after which "in use" grows by about one per-device Jacobian more
  on the branch than on `master` locates the extra copy.
- Also note the JAX/jaxlib versions: the CPU checks used JAX 0.10.1.

## Constraints for a fix (decisions already made by the user)

- The network is always evaluated on exactly `batch_size` rows, so it is compiled once per
  batch size whatever the number of samples (no recompilation per N).
- Inputs are padded only to a multiple of the number of devices; the last batch of every
  device is padded inside `_take_batch`. Non-lazy outputs are trimmed back to N.
- Outputs follow the sharding of the batch results (`out_specs` is respected: a replicated
  output stays replicated). Do not force every output to row sharding. A fix such as
  "use `DEVICE_SHARDING` in `BatchLayout.alloc` when `piece.sharding.is_equivalent_to(DEVICE_SHARDING, piece.ndim)`"
  keeps this property.
- `in_specs`, `out_specs` and `vmap_in_axes` can only be customised for `batch_size=None`
  calls (batched calls raise).
- Lazy batches are in local order; any full-length array is aligned with
  `iterable.layout.split(x)`.
- Helpers are module-level `jax.jit` functions with static sizes (no `lru_cache`).

## Verifying a fix

- Benchmark (script on both branches, `tests/benchmark/batching/batching_benchmark.py`;
  write results **outside** the repository so that switching branch works):

  ```bash
  python tests/benchmark/batching/batching_benchmark.py run --label branch \
      --cases gradients minsr_dense minsr_lazy minsr_dense_nonholo --out ~/bench/branch_fix.json
  # git checkout master, then the same with --label master --out ~/bench/master.json
  python tests/benchmark/batching/batching_benchmark.py compare ~/bench/master.json ~/bench/branch_fix.json
  ```

  Expected after a fix: branch ≤ `master` in every memory column, results agreeing to
  ~1e-13. `compare` refuses two runs of the same source (it hashes the imported package).
- Tests, on 1 device and on 4 simulated devices (ordering bugs only show with several
  devices):

  ```bash
  python -m pytest tests/ -q
  XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest tests/ -q
  ```

  plus `python -m pytest tests/ -q` on the real GPUs.

## Working conventions with the user

- Ask before changing code when the request is only to answer or investigate; ask before
  committing or pushing.
- Match the surrounding code style and comment density; keep changes minimal.
