# Building S (the QGT) sharded, without a full P×P on any GPU

Findings and implementation options, written 2026-10-08. Everything below refers to branch
`sharded-batched-update` of `Custom_jVMC`, commit `13d6be0`. That commit's `jVMC_exp/` is the
same as at `1b2214c`.

## TL;DR

**Goal.** S (P×P) split by rows across GPUs, with no full P×P on any GPU at any time. This is
needed so the matrix can go to JAXMg (`diagonalization_mode="distributed"`), and so the
library scales across nodes.

**What is wrong today.**
- `Evolution._get_qgt` returns S split by rows (`P('devices')`).
- But inside its `shard_map`, every GPU first computes its local `conj(J).T @ J`, which is
  the **full P×P**. Only then does `psum_scatter` reduce it and keep one slice. Measured on
  GPU, the constant memory term is **identical on 1 and 4 GPUs** (0.907 GB), which proves it.
- MinSR's tangent kernel does the same thing on the sample axis: a full B×B on every GPU.

**Implementation options.**

| | what | status | extra compute | communication of S per build |
|---|---|---|---|---|
| **A** | block-wise reduce-scatter (drop-in for `Evolution`) | code tested on 4 simulated CPU devices | none | ~16·P²·(n−1)/n bytes per GPU |
| **B** | recompute: every GPU builds its own rows from all samples (netket_jaxmg style) | prototype tested on 4 simulated CPU devices | gradients n_dev× | none |
| **C** | hybrid on a 2D mesh (nodes × GPUs): B across nodes, A inside a node | design only | gradients n_nodes× | NVLink only |
| D | XLA "windowed einsum" flag | idea, untested | none | same as A |

**Recommendation.**
1. Implement **A** first. It is a drop-in, costs no extra compute, and is fully tested on CPU.
   Include the `zip` loop fix (§3.A).
2. Benchmark on GPU.
3. Consider **B** or **C** when running across nodes at large P (roughly P ≳ 10k), where the
   P² reduce-scatter over InfiniBand starts to cost more than recomputing gradients (§4).

Also: S is still gathered in full onto every GPU at the **solve** unless
`diagonalization_mode="distributed"` is used (§2.3). Fixing the construction alone does not
give a fully sharded pipeline.

---

## 1. Background: where S is built

`Evolution.get_update` (`jVMC_exp/optimizer/base.py`):

1. For holomorphic nets, it removes the doubled gradient:
   `grad_log_psi.transform(self._remove_double_trans)`.
   - The raw holomorphic Jacobian has **2P** columns, because `flat_gradient_holo` returns
     `[g, i·g]`.
   - So S is **P×P**. (`psi.numParameters` = P.)
   - The force is computed earlier, in `Observable.value_and_grad`, with the 2P-wide
     Jacobian, and its doubling is removed afterwards with `_remove_double_fn`.
2. `_get_lhs_dense(grad_log_psi)`:
   - **Dense (`SampledObs`):** `S = _get_qgt(grad_log_psi._normalized_obs)`.
   - **Batched (`LazySampledObs`):** for each batch,
     `batch, batch_mean = _weight_and_mean(batch, weights)` and then
     `S += _get_qgt(batch)`. At the end, `S - outer(conj(mean), mean)`.
   - Then `_S0 = S`, `_lhs_trans_fn`, `diag_scale`, `diag_shift`.
3. The solver, e.g. `PinvSNR`, calls `solver.util.diagonalize(A, pad_size, mode, T_A)`.

The current `_get_qgt`:

```python
    @sharded(use_vmap=False)
    def _get_qgt(self, grad_log_psi, *, batch_size):
        local = jnp.tensordot(jnp.conj(grad_log_psi), grad_log_psi, axes=(0, 0))   # (P, P) on EVERY device
        return jax.lax.psum_scatter(local, "devices", scatter_dimension=0, tiled=True)
```

**Do not benchmark `stats.(Lazy)SampledObs.get_covar()` as if it were S.**
- That covariance is **replicated** (`P()`) and **2P×2P** for holomorphic nets.
- The optimizer does not use it for S. It is only used for the force,
  `get_covar(o_loc)`, a vector.
- The first version of the benchmark measured `get_covar()` and overstated the S memory per
  GPU by 16×.

**Notation** (benchmark setup: `CpxRBM`, L=10, 512 hidden units, complex128):
- N = samples = 65536; P = 5120.
- n_dev = number of GPUs; n = number of batches.
- J_b = one gradient batch on one GPU, P columns wide: N/(n_dev·n) · P · 16 B.
  That is 1.34/n GB on 4 GPUs and 5.37/n GB on 1 GPU.
- "slice" = one GPU's share of S = P²·16/n_dev B. That is 0.105 GB on 4 GPUs; a full P×P is
  0.42 GB.

---

## 2. Findings

### 2.1 Why a full P×P appears

The sum in S = Σ_s w_s conj(O_s)ᵀ O_s runs over the **samples**, which are split across GPUs.
Each GPU's own term covers *every* entry of S, so the GPUs must add up full-size terms. The
standard collective for that, reduce-scatter (NCCL `ReduceScatter`), takes a full-size buffer
on every device as its *input*. XLA's default plan, one big local product followed by one
reduce-scatter, is the fastest but materializes a P×P on each GPU. A different 1D mesh would
not change this.

There are only two ways around it:
- **Interleave** the computation with the reduction: compute only the block that is about to
  be reduced (option A, and option D via XLA).
- **Don't split the axis you sum over**: every GPU sees all samples, either by recomputing
  their gradients (option B) or with a 2D mesh (option C).

### 2.2 Evidence

**Compiled memory on CPU** (4 simulated devices, `memory_analysis().temp_size_in_bytes`):

| function | extra memory |
|---|---|
| current `_get_qgt`, local batch 1024 × P=1024 | 4.0 slices (= the full P×P) |
| current `_get_qgt`, local batch 4096 × P=1024 | 8.0 slices (= full P×P + a `conj` copy of the local batch) |
| option A `_add_qgt`, local batch 1024 × P=1024 | **1.25** slices |
| option B recompute, P=1024, N=4096, chunk 256 | **2.25** slices, and **no collectives at all** |
| current MinSR `_get_single_tangent_kernel`, B=4096, P=1024 | 4.5 slices (full B×B = 4) |
| MinSR block-wise kernel, n_chunks = n_dev | 1.75 slices |

**GPU sweeps** (`tests/benchmark/batched_jacobian`, peak GB per GPU of the S/force
computation alone):

| | n=dense | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| 4 GPUs, force (`results_20261008_131159`) | 8.07 | 8.07 | 4.10 | 2.05 | 1.04 | 0.52 |
| 4 GPUs, qgt | 5.39 | 6.27 | 3.59 | 2.25 | 1.58 | 1.24 |
| 1 GPU, force (`results_20261008_145244`) | 32.26 | 32.25 | 16.38 | 8.19 | 4.10 | 2.05 |
| 1 GPU, qgt | 21.52 | 22.38 | 13.84 | 7.13 | 3.78 | 2.25 |

Both runs fit one model to about 0.01 GB at every point:
- force = 6·J_b
- dense qgt ≈ 4·J (the peak is the gradient computation itself)
- batched qgt = max(**4·J_b + C**, **5·J_b + S_dev**). The second term exists only for n ≥ 2.
  S_dev = one slice.
- **C = 0.907 GB on both 1 and 4 GPUs.** If S were built fully sharded, C would be 4× smaller
  on 4 GPUs. C ≈ 2 × 0.42 GB (full P×P) + ~70 MB, and it occurs inside `_get_qgt`.
- The **5·J_b + S_dev** term is the next gradient batch (≈4·J_b) being computed while the
  previous batch and S are still alive. That is the `zip` problem in §2.3.
- On 4 GPUs, C dominates from n ≈ 6 onwards. That is why the 4-GPU QGT curve bends away from
  1/n while the force does not. On 1 GPU, the bend only comes at n ≈ 20.

**Diagnostic for later:** plot the 1-GPU curve together with 4 × the 4-GPU curve. With S
built fully sharded, the two overlap. Today the 4-GPU curve lies above at large n.

### 2.3 Other issues found, and their status on the branch

| issue | status |
|---|---|
| `sharded._iter_local_batches` kept `result` alive across `yield` | **fixed** on branch |
| `LazySampledObs.transform` generator kept the raw 2P batch alive → `map(jitted_fn, iterable)` | **fixed** on branch (`stats.py:433`) |
| `sharded._call_local` dispatch overlap (+1 J) → `return jax.block_until_ready(out)` | **fixed** on branch (see memory note on the minSR probe) |
| MinSR batched loop `enumerate(zip(...))` stale tuple | **fixed** on branch (`next(iter(...))`) |
| `_get_lhs_dense` batched loop `for batch, weights in zip(...)`: CPython's `zip` keeps its last result tuple, so the previous batch stays alive **even after `del batch`** while the next one is computed | **OPEN.** Fix in §3.A (verified: 2 → 0 batches alive) |
| same `zip` pattern in `LazySampledObs.mean`, `.var`, `.get_covar` (`stats.py` ~356–409), which affects the force | **OPEN.** Same fix pattern |
| `S += _get_qgt(...)` keeps 3 copies of S's slice (old, new term, sum) | **OPEN.** Option A donates S, which removes it. Not the peak today: the full P×P inside `_get_qgt` is larger |
| `diag_scale`: `S + jnp.diag(scale * jnp.diag(S))` built a **replicated** P×P | **fixed** on branch (`S.at[idx, idx].multiply(1 + scale)`) |
| solve: `diagonalization_mode="device"` runs `jnp.linalg.eigh` on S split by rows; the compiled HLO **all-gathers** S, then every GPU computes the same eigh (full S + full V + workspace ≈ 2–3 P×P per GPU) | by design. Only `"distributed"` avoids it |
| `"distributed"` needs `JVMC_USE_DISTRIBUTED=true` and **one process per GPU** (`srun --ntasks=<n_gpus> --gpus-per-task=1`), otherwise it **silently falls back** to `"device"`. The benchmark runs one process with 4 GPUs, so it always falls back | note |
| JAXMg path does `jax.device_put(A, DEVICE_SHARDING_2D, donate=True)`, a cross-mesh reshard from 1D rows to `MESH_2D`, and `V` back | **unchecked**: does it ever go through a replicated copy? |

**Why JAXMg alone is not enough:** with the full P×P still built in `_get_qgt`, JAXMg lowers
the peak only by a constant factor, from ~2–3 P×P (device eigh) to ~1–2 P×P (construction).
The largest reachable P stays bounded by one GPU's memory. Only with a sharded construction
does memory per GPU scale as P²/n_dev, so that P_max grows like √n_dev. JAXMg also spreads the
O(P³) eigh work, which device mode repeats on every GPU; that helps at large P.

### 2.4 `sharded` decorator and `REPLICATED_SPEC` (tested on the branch)

| case | result |
|---|---|
| `use_vmap=False`, `in_specs=(REPLICATED_SPEC, DEVICE_SPEC)`, `batch_size=None` | correct (option B uses this) |
| `use_vmap=True` with a replicated argument and default `vmap_in_axes` | error on 4 devices (vmap over axis 0 of the full replicated array). **On 1 device it silently maps element by element** |
| `use_vmap=True`, `vmap_in_axes=(None, 0)` | correct |
| custom `in_specs` with a `batch_size` or `yield_iter=True` | `ValueError`, by design |
| `in_specs=(P(None), ...)` | `P(None) == P()` is **False**. The argument is placed **sharded** and XLA adds an all-gather on every call (correct values, wasted communication) |
| output built only from replicated inputs, default `out_specs=DEVICE_SPEC` | **silently wrong shape**: n_dev copies concatenated, (32,) instead of (8,) |

Possible hardening (untested):
- Treat a spec as replicated if `all(p is None for p in spec)`.
- Default the vmap axis of replicated arguments to `None`.

Not tested:
- multi-process `device_put(..., REPLICATED_SHARDING)`;
- the `sharded` on `master`, which is a different implementation.

### 2.5 MinSR (tangent kernel T = J Jᴴ, N×N)

`_get_single_tangent_kernel` / `_get_double_tangent_kernel` (`minsr.py`):
- `all_to_all` turns J from split by samples into split by parameters, giving `(B, P/n_dev)`
  per GPU.
- `local = grad @ conj(grad).T` is then the **full B×B** on every GPU, followed by
  `psum_scatter`.

How bad this is depends on the path:
- **Dense path:** B = N, so a full N×N on every GPU.
- **Batched path:** each (l, r) block is (N/n_batches)². It is already no larger than one
  GPU's share of T once n_batches² ≥ n_dev.

The fix is the same block-wise idea (§3.E), or recompute as in netket_jaxmg (§3.B, §5).

---

## 3. Implementation options

### 3.A Block-wise reduce-scatter (recommended first step)

**Idea.** Split the rows of S into n_dev blocks per GPU. Step j computes the partial sum only
for "block j of every GPU's rows", which is one slice in size, and `psum_scatter`s it so each
GPU receives its own block j. The block count is fixed to n_dev: then each partial sum has
exactly one slice's size, and fewer blocks would be larger. S is donated and updated in place.

**Requirement.** P must be divisible by n_dev², so the padding changes from `% n_dev` to
`% n_dev**2`. `_pad_obs`, the solver's `pad_size`, `_get_tdvp_error` and `util.s_norm_fn` all
read `_params_pad_size`, so nothing else changes.

**Changes to `jVMC_exp/optimizer/base.py`** (tested):

(1) Import and helper, above `class AbstractOptimizer`:

```python
from jVMC_exp.sharding_config import sharded, MESH, DEVICE_SHARDING

def _sharded_zeros_qgt(grad_log_psi):
    """
    Zero matrix of shape (n_parameters, n_parameters), sharded across devices on the first axis.
    """
    n_params = grad_log_psi.shape[1]

    return jnp.zeros((n_params, n_params), grad_log_psi.dtype, device=DEVICE_SHARDING)
```

(2) In `Evolution.__init__`, replace the `self._params_pad_size = ...` line:

```python
        # S is built one block of rows per device at a time (see _add_qgt),
        # which needs the number of parameters to be divisible by the number of devices squared
        self._params_pad_size = (- num_params) % MESH.shape["devices"] ** 2
```

(3) The whole `_get_lhs_dense`. This includes the `zip` fix.

```python
    def _get_lhs_dense(self, grad_log_psi: SampledObs | LazySampledObs):
        '''
        Returns left hand side of the TDVP equation with shape (n_parameters, n_parameters)
        and sharded across devices on the first dimension.
        If n_parameters is not divisible by the number of devices, the output is padded.
        '''
        if self._params_pad_size != 0:
            grad_log_psi.transform(self._pad_obs)
        
        if isinstance(grad_log_psi, SampledObs):
            normalized_obs = grad_log_psi._normalized_obs
            S = self._add_qgt(_sharded_zeros_qgt(normalized_obs), normalized_obs, batch_size=None)
        else:
            mean = 0
            S = None
            # zip would keep the last batch alive while the next one is computed,
            # since it holds on to the tuple it last returned
            batches = iter(grad_log_psi._observations)
            for weights in grad_log_psi._weights:
                batch, batch_mean = self._weight_and_mean(next(batches), weights, batch_size=None)
                mean += batch_mean
                if S is None:
                    S = _sharded_zeros_qgt(batch)
                # S is donated and updated in place
                S = self._add_qgt(S, batch, batch_size=None)
                # Free the batch before the next one is computed
                del batch

            S = self._subtract_mean_outer(S, mean, batch_size=None)

        if self._params_pad_size != 0:
            grad_log_psi.transform(self._unpad_obs)

        self._S0 = S
        S = self._lhs_trans_fn(S)

        if self.diag_scale > 1e-15:
            idx = jnp.arange(S.shape[0] - self._params_pad_size)
            S = S.at[idx, idx].multiply(1 + self.diag_scale)
        if self.diag_shift > 1e-15:
            idx = jnp.arange(S.shape[0] - self._params_pad_size)
            S = S.at[idx, idx].add(self.diag_shift)

        return S
```

(4) Replace `_get_qgt` with these two methods. `_weight_and_mean` is unchanged.

```python
    @sharded(use_vmap=False, donate_argnums=1)
    def _add_qgt(self, S, grad_log_psi, *, batch_size):
        """
        Returns ``S + conj(grad_log_psi).T @ grad_log_psi``, with ``S`` sharded across devices
        on the first axis. ``S`` is donated, so it is updated in place.

        Each device sums over its own samples, and the partial sums are reduced across devices
        one block of rows at a time: the partial sum of a block has as many rows as the share
        of ``S`` of one device, so no device ever holds a matrix of the size of the full ``S``.
        The number of parameters has to be divisible by the number of devices squared.
        """
        n_devices = MESH.size
        n_samples, n_params = grad_log_psi.shape
        block = n_params // n_devices ** 2
        # Column p = d * n_params // n_devices + j * block + r is row r of block j of device d
        grad_blocks = grad_log_psi.reshape(n_samples, n_devices, n_devices, block)

        def add_block(j, S):
            # Block j of the rows of every device, in device order
            rows = jax.lax.dynamic_index_in_dim(grad_blocks, j, axis=2, keepdims=False)
            local = jnp.tensordot(
                jnp.conj(rows.reshape(n_samples, n_devices * block)), grad_log_psi, axes=(0, 0)
            )
            # Every device receives the sum over all devices of its own block j
            local = jax.lax.psum_scatter(local, "devices", scatter_dimension=0, tiled=True)
            S_block = jax.lax.dynamic_slice_in_dim(S, j * block, block, axis=0)

            return jax.lax.dynamic_update_slice_in_dim(S, S_block + local, j * block, axis=0)

        return jax.lax.fori_loop(0, n_devices, add_block, S)

    @sharded(use_vmap=False, donate_argnums=1)
    def _subtract_mean_outer(self, S, mean, *, batch_size):
        """
        Returns ``S - outer(conj(mean), mean)``, with ``S`` and ``mean`` sharded across devices
        on the first axis. ``S`` is donated, so it is updated in place.
        """
        full_mean = jax.lax.all_gather(mean, "devices", tiled=True)

        return S - jnp.outer(jnp.conj(mean), full_mean)
```

Notes:
- `donate_argnums=1` refers to `S`, because the function the decorator compiles takes the
  kwargs dict as argument 0. Donation through `sharded` was verified: the old `S` is deleted.
- In the dense path, `normalized_obs` belongs to the `SampledObs`, which the solver still
  uses, so it must **not** be donated. It isn't.
- `_subtract_mean_outer`: `mean` from `_weight_and_mean` is split across devices, and each
  device's rows of S need its own slice of `mean` (rows) and the full `mean` (columns, via a
  tiny all-gather).
- With 1 device, `_add_qgt` is a single block, the full S, which can't be avoided anyway.

**Tested** (4 simulated CPU devices, `TDVP.get_update(loss.value_and_grad(sampler))`,
current code vs option A):
- Networks: holomorphic `CpxRBM` with P=80 (no padding) and P=50 (padding 14 instead of 2),
  and `RBM` with real parameters and `make_real=True`; dense and batched; `diagonalShift=1e-3`,
  `diagonalScale=1e-2`.
- S matches to ≤1.5e-16 relative and updates to ≤3.6e-15. For the real RBM the update is zero
  in both versions, which is expected, so that comparison doesn't test much.
- S is split by rows, and the padded rows and columns are exactly zero.
- Extra memory of `_add_qgt`: 1.25 slices, against 4.0 for `_get_qgt`.
- Batches still alive when the next gradient batch is requested: 2 → 0.

**Expected on GPU** (extrapolated from the fit, not measured): batched QGT peak ≈ 4·J_b + ~1
slice, i.e. 4 GPUs n=1…16 ≈ 5.4, 2.8, 1.5, 0.8, 0.45 GB. Batched n=1 ≈ dense, 1/n down to
n≈8, and 4 GPUs = 1 GPU / 4.

Time cost: n_dev reduce-scatters of 1/n_dev size per batch instead of one, so the same total
bytes. Expect a small slowdown per batch, unmeasured.

### 3.B Recompute (netket_jaxmg style)

**Idea.** Don't split the axis you sum over:
- Copy the **samples** (tiny: N × sample shape) and the weights to every GPU.
- Each GPU loops over chunks of **all** samples, recomputes their gradients, and adds
  `conj(w·J_chunk[:, my rows]).T @ J_chunk` to its own rows of S.
- The mean is accumulated locally too: every GPU sees all samples, so no psum is needed.

Result: no collective on anything larger than the samples, and per GPU only one Jacobian chunk
plus its own rows of S. The price is that every GPU computes the gradients of all N samples.

**Prototype** (tested), as a `TDVP` subclass:

```python
import jax
import jax.numpy as jnp

from jVMC_exp.optimizer.tdvp import TDVP
from jVMC_exp.sharding_config import sharded, MESH, REPLICATED_SPEC

class TDVPRecompute(TDVP):
    def get_lhs_recompute(self, samples, weights, chunk_size):
        '''
        Same output as `_get_lhs_dense`, but built from the samples instead of the gradients.
        `chunk_size` is the number of samples whose gradients each device computes at once.
        '''
        num_samples = samples.shape[0]
        weights = weights / jnp.sum(weights)

        # Pad to a multiple of the chunk size with copies of the first sample, which get zero weight
        num_pad = (-num_samples) % chunk_size
        if num_pad:
            samples = jnp.concatenate([samples, jnp.repeat(samples[:1], num_pad, axis=0)])
            weights = jnp.pad(weights, (0, num_pad))

        S = self._qgt_recompute(
            samples, weights,
            parameters=self.psi.grad_parameters, chunk_size=chunk_size, batch_size=None
        )

        self._S0 = S
        S = self._lhs_trans_fn(S)

        if self.diag_scale > 1e-15:
            idx = jnp.arange(S.shape[0] - self._params_pad_size)
            S = S.at[idx, idx].multiply(1 + self.diag_scale)
        if self.diag_shift > 1e-15:
            idx = jnp.arange(S.shape[0] - self._params_pad_size)
            S = S.at[idx, idx].add(self.diag_shift)

        return S

    @sharded(use_vmap=False, in_specs=(REPLICATED_SPEC, REPLICATED_SPEC), static_kwarg_names=("chunk_size",))
    def _qgt_recompute(self, samples, weights, *, parameters, chunk_size, batch_size):
        """
        Rows of ``S = sum_s w_s conj(O_s - mean).T (O_s - mean)`` owned by this device, where
        ``samples`` and ``weights`` are replicated on all devices.
        """
        n_devices = MESH.size
        device = jax.lax.axis_index("devices")
        n_chunks = samples.shape[0] // chunk_size

        single_gradient = lambda s: self.psi.flat_gradient_function(self.psi.apply_fun, parameters, s)

        def jacobian(chunk):
            J = jax.vmap(single_gradient)(chunk)
            if self.psi.holomorphic:
                J = self._remove_double_trans(J)
            return self._pad_obs(J)

        J_shape = jax.eval_shape(jacobian, samples[:chunk_size])
        n_params = J_shape.shape[1]
        rows = n_params // n_devices

        def add_chunk(i, carry):
            S_local, mean = carry
            chunk = jax.lax.dynamic_slice_in_dim(samples, i * chunk_size, chunk_size)
            w = jax.lax.dynamic_slice_in_dim(weights, i * chunk_size, chunk_size)
            J = jacobian(chunk)
            # Columns of the Jacobian that correspond to the rows of S owned by this device
            J_rows = jax.lax.dynamic_slice_in_dim(J, device * rows, rows, axis=1)
            S_local = S_local + jnp.conj(w[:, None] * J_rows).T @ J
            mean = mean + w @ J

            return S_local, mean

        S_local = jnp.zeros((rows, n_params), J_shape.dtype)
        # The zeros are the same on every device, the loop makes them device dependent
        S_local = jax.lax.pcast(S_local, ("devices",), to="varying")
        mean = jnp.zeros(n_params, J_shape.dtype)
        S_local, mean = jax.lax.fori_loop(0, n_chunks, add_chunk, (S_local, mean))

        mean_rows = jax.lax.dynamic_slice_in_dim(mean, device * rows, rows)

        return S_local - jnp.outer(jnp.conj(mean_rows), mean)
```

Notes:
- Padding of the parameters stays at `% n_dev`; n_dev² is not needed here.
- `pcast(..., to="varying")` is required: the `fori_loop` carry must have the same
  per-device type on input and output.
- The output is per-device (rows chosen with `axis_index`), so the default
  `out_specs=DEVICE_SPEC` is right. See the trap in §2.4, G.
- `chunk_size`: each GPU holds a `chunk_size × P` Jacobian. To match the memory of today's
  batched path, use `chunk_size = psi.batchSize // n_dev`.
- Like the current batched path, it accumulates S uncentred and subtracts the mean at the
  end, so it has the same cancellation risk when |mean| is much larger than the spread.

**Tested** (4 simulated CPU devices):
- vs the current `_get_lhs_dense`, for `CpxRBM` P=80, `CpxRBM` P=50 (padded), and `RBM` with
  `make_real=True`; `chunk_size` 64 and 100 (100 doesn't divide N=512, which checks the
  sample padding).
- S and `_S0` match to ≤6e-16 and are split by rows.
- P=1024, N=4096, chunk 256: extra memory 2.25 slices, output 1 slice, and **no collective
  ops in the compiled HLO**.

**Wiring it in (not done; sketch, untested).**
1. `get_update` needs the samples and weights the gradients came from. **Don't** read
   `self.sampler.samples`: `Evolution.cross_validation` swaps in a subset, then restores the
   full set (in `finally`) *before* calling `get_update(objective_fn_out_1)`. Options:
   - add `samples` and `weights` fields to `ObjectiveFunctionOutput`, filled in
     `Observable.value_and_grad`, or
   - add a new observable type that stores `(psi, samples, weights)`.
2. In `get_update`, call `get_lhs_recompute` when a flag is set (e.g. a constructor argument
   `qgt_mode="recompute"`).
3. **Compute the force in the same pass**, which saves one whole gradient pass (§4):
   - every GPU holds the full `J` chunk, so add a carry `F += conj(J).T @ (w * (E_chunk - E_mean))`;
   - copy `E_loc` to every GPU and compute `E_mean = Σ w E` before the loop;
   - return `(S_local, F)` with `out_specs=(DEVICE_SPEC, REPLICATED_SPEC)`. F is the same on
     every device, so `P()` is correct.
   
   This equals today's `get_covar(o_loc)` after `_remove_double_fn`, because today's force is
   Σ w conj(O)(E − Ē). The objective function then no longer needs its own gradient pass.

### 3.C Hybrid on a 2D mesh (design only)

Mesh `(nodes, gpus)` with n_nodes × g devices, built per process like netket_jaxmg's `mesh.py`
(`get_device_grid`). Each GPU (a, k) owns global row block `a·g + k` of S, i.e.
`out_specs=P(("nodes", "gpus"))`.

- Samples: split over `gpus`, copied across `nodes` (`P("gpus")`), so each **node** holds all
  N samples, N/g per GPU.
- Each GPU computes J for its N/g samples, all P columns.
- Node a needs rows R_a (P/n_nodes rows). Apply option A **inside the node**: the block-wise
  partial sums for the rows of node a, with `psum_scatter` over the `gpus` axis only (NVLink).
  Use `axis_index("nodes")` to choose R_a.
- Mean: `psum` over `gpus` of `w @ J` (a P-vector).
- Cost: gradients n_nodes× redundant (not n_dev×); communication of S intra-node only,
  ≈16·(P/n_nodes)·P·(g−1)/g bytes per GPU; no full P×P anywhere.
- jVMC's `MESH` is 1D. This needs a second mesh, and probably a 2D path in `sharded`.

### 3.D XLA "windowed einsum" (idea, untested)

XLA can rewrite "matmul + reduce-scatter" into a loop of block products with
collective-permutes. It acts only under `jit` with sharding annotations, not inside
`shard_map`. On GPU it is off by default. To try it:
- `XLA_FLAGS=--xla_gpu_threshold_for_windowed_einsum_mib=0`, possibly with
  `--xla_gpu_multi_streamed_windowed_einsum=true`;
- write S as a plain `jnp.einsum` under `jit` with `out_shardings` split by rows;
- check that the compiled HLO has a `while` loop with `collective-permute` instead of a P×P
  `reduce-scatter`.

This is the same algorithm as A, hidden behind a compiler flag. It can't be tested on CPU.

### 3.E MinSR: block-wise tangent kernel (single kernel tested)

Same idea on the sample axis, applied after the existing `all_to_all`. Rows of `grad_l` must
be divisible by `n_dev × n_chunks` (B, or 2B when `self._concat`); otherwise fall back to
`n_chunks=1` or pad. In the CPU test with B=4096 and P=1024, extra memory went from 4.5 to 1.75
slices, and the result equalled `g @ conj(g).T`. `grad_l ≠ grad_r` (the double kernel) is
untested.

```python
def _tangent_kernel_chunked(grad_l, grad_r, *, n_chunks):
    """
    Rows of grad_l @ conj(grad_r).T owned by this device, for grad_l and grad_r already
    redistributed by all_to_all (all samples, own parameter shard). The local partial sum
    is built one block of rows at a time, so it never has the size of the full kernel.
    """
    n_dev = MESH.size
    b = grad_l.shape[0]
    sub = b // n_dev // n_chunks
    # Row s = d * (b // n_dev) + j * sub + r, i.e. sub-block j of the rows owned by device d
    grad_l4 = grad_l.reshape(n_dev, n_chunks, sub, grad_l.shape[1])
    grad_r_h = jnp.conj(grad_r).T

    def body(j, T_local):
        # Sub-block j of the rows of every device, in device order
        rows = jax.lax.dynamic_index_in_dim(grad_l4, j, axis=1, keepdims=False).reshape(n_dev * sub, -1)
        # Device d receives the sum over parameter shards of its own sub-block j
        piece = jax.lax.psum_scatter(rows @ grad_r_h, 'devices', scatter_dimension=0, tiled=True)
        return jax.lax.dynamic_update_slice_in_dim(T_local, piece, j * sub, axis=0)

    T_local = jnp.zeros((b // n_dev, grad_r.shape[0]), jnp.result_type(grad_l, grad_r))
    # The zeros are the same on every device, the loop makes them device dependent
    T_local = jax.lax.pcast(T_local, ('devices',), to='varying')

    return jax.lax.fori_loop(0, n_chunks, body, T_local)

# in MinSR:
    @sharded(use_vmap=False)
    def _get_single_tangent_kernel(self, grad, *, batch_size):
        grad = jax.lax.all_to_all(grad, 'devices', split_axis=1, concat_axis=0, tiled=True)
        return _tangent_kernel_chunked(grad, grad, n_chunks=MESH.size)

    @sharded(use_vmap=False)
    def _get_double_tangent_kernel(self, grad_l, grad_r, *, batch_size):
        grad_l = jax.lax.all_to_all(grad_l, 'devices', split_axis=1, concat_axis=0, tiled=True)
        grad_r = jax.lax.all_to_all(grad_r, 'devices', split_axis=1, concat_axis=0, tiled=True)
        return _tangent_kernel_chunked(grad_l, grad_r, n_chunks=MESH.size)
```

The recompute version for MinSR is exactly what netket_jaxmg does (§5).

---

## 4. Trade-offs

**Gradient evaluations** (N samples, n devices):

| | total on the machine | per GPU (wall time) |
|---|---|---|
| dense, Jacobian kept in memory | 1·N | N/n |
| current batched path (force pass + S pass) | 2·N | 2N/n |
| B with a separate force pass | (n+1)·N | N(1 + 1/n) |
| B with the force in the same pass | n·N | N |
| C (hybrid), force in the same pass | n_nodes·N | N/g |

**When does recompute (B) beat reduce-scatter (A)?** When

  16·P² / BW  >  N · t_grad,  with t_grad = gradient time per sample.

So recompute favours large P, slow links and moderate N; reduce-scatter favours many samples,
fast links and expensive networks.

Rough numbers from the benchmark (`CpxRBM`, P=5120, N=65536):
- All N gradients on 1 GPU ≈ 27 ms (1-GPU force time).
- The 4-GPU S build ≈ 84 ms, gradients included.
- **One node (NVLink):** the reduce-scatter (~0.3 GB per GPU) takes milliseconds, so B costs
  about +15–30% in time and its only gain is memory.
- **Across nodes**, assuming ~25 GB/s per GPU over InfiniBand (not checked for JUPITER): the
  reduce-scatter takes ~13 ms at P=5120 (a draw), ~0.2 s at P=20k and ~6 s at P=100k. The
  RBM's gradient time grows about linearly in P, so B wins above P ≈ 10k.

**Scaling caveat for B:** per-GPU gradient work stays N whatever n is, while the matrix
product shrinks as 1/n. With many nodes the gradients become the bottleneck. That is the
reason for C.

---

## 5. Reference: how netket_jaxmg does it

Source: <https://github.com/therooler/netket_jaxmg>, `src/srt.py` (`srt_onthefly`), called
from `src/vmc_sr.py`. JAXMg paper: Wiersema, arXiv:2601.14466.

- They never build S. They build the **NTK** T = J Jᵀ (N×N; 2N×2N in complex mode), so the
  sum runs over the **parameters**.
- Inside `shard_map`, `in_specs=(P("S", None), P(), P())`:
  - own samples split across GPUs;
  - `all_samples` copied to every GPU (`with_sharding_constraint(samples, P())`);
  - parameters copied to every GPU (`jax.lax.pvary(parameters_real, "S")`).
- `nt.empirical_ntk_by_jacobian` computes T[own samples, all samples]. Every GPU
  **recomputes the Jacobian of all samples**, in chunks of `chunk_size` with `lax.map`.
  `out_specs=P("S", None)`; no reduction, no full-size temporary.
- Centering, weights and the diagonal shift are elementwise on the split T under `jit`.
  `jnp.eye(N)` and `jnp.outer(w_sqrt, w_sqrt)` rely on XLA's sharding propagation, which I
  didn't verify.
- Solve: `jaxmg.potrs` (Cholesky) directly on the 1D row-split T,
  `partial(potrs, T_A=2**12, mesh=..., in_specs=(P("S", None),))`. No 2D reshard.
- They set `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, which changes how memory is allocated and
  reported. Keep that in mind when comparing memory numbers.

---

## 6. How to test

### CPU (login node, 4 simulated devices)

```bash
ulimit -c 0
export OPENBLAS_NUM_THREADS=32 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4
/e/scratch/neuquass/solinas1/env/jVMC/bin/python -u my_check.py
```

Useful checks:

```python
# 1. Correctness: compare against the current code (run the old package and the patched one,
#    save S/_S0/update with np.savez, compare the unpadded block S0[:n, :n]).
psi = NQS(nets.CpxRBM(numHidden=5), 10, N, seed=123)            # P=50 -> exercises the padding
mc = sampler.MCSampler(psi, jVMC_exp.propose.SpinFlip(), key=123, numChains=64, numSamples=N)
mc.sample()
opt = TDVP(mc, psi, make_real=False, diagonalShift=1e-3, diagonalScale=1e-2)
update = opt.get_update(loss.value_and_grad(mc))                # loss = Observable(H, batched_jacobian=...)
S0 = opt._S0; print(S0.sharding.spec)                           # expect P('devices',)

# 2. Extra memory of a decorated method (one slice = P*P*16 // n_dev bytes)
jsh = next(v for k, v in opt._sharded_cache.items() if k[0] == "_add_qgt")
ma = jsh.lower({}, S, W).compile().memory_analysis()            # first arg = kwargs dict
print(ma.temp_size_in_bytes / slice_bytes)

# 3. No communication on large arrays
hlo = jsh.lower(...).compile().as_text()
print(sorted(set(re.findall(r"(all-reduce|reduce-scatter|all-gather|all-to-all|collective-permute)", hlo))))

# 4. Is a batch still alive when the next one is requested?
#    Wrap obs.observations in a generator that counts jax.live_arrays() of batch shape
#    just before calling next() on the inner iterator. Don't keep a reference to the
#    batch in the wrapper, and filter by shape[0] == local batch rows, so that S is not counted.
```

Pitfalls met along the way:
- `jax.live_arrays()` also lists S. Filter by shape.
- A reference held by the test code itself also counts.
- Shard_map carries need `pcast(..., to="varying")` when they start as zeros.

### GPU

- Benchmark: `tests/benchmark/batched_jacobian/` (committed on the branch). Run
  `jacobian_sweep.sh` on a 4-GPU node: one process sees all 4 GPUs.
  - It builds S exactly as the optimizer does: `transform(_remove_double_trans)` +
    `optimizer._get_lhs_dense`.
  - Sampling and E_loc use a fixed batch (`--sample_batch 4096`), and only the Jacobian is
    batched (`psi._batchSize = n_samples // n_batches`).
  - It records peak − in-use before, maxed over GPUs; the median of 5 timed runs; the shape,
    sharding and size per GPU of the output; and `result_norm`.
  - Each sweep goes to a new `results_<timestamp>/`, and `plot_jacobian.py` makes the plot.
- For 1 GPU: `CUDA_VISIBLE_DEVICES=0`.
- To compare implementations, add a `--qgt_impl {current,blockwise,recompute}` switch to
  `batched_jacobian.py`.
- Success criterion for A: batched n=1 ≈ dense, the QGT curve follows 1/n to n≈8, and
  4-GPU × 4 ≈ the 1-GPU curve.
- Memory tests must run warm, unsynced repetitions; a cold or synced probe hides some
  effects (see the minSR probe).
- JAXMg (`"distributed"`) needs `JVMC_USE_DISTRIBUTED=true` and
  `srun --ntasks=<n_gpus> --gpus-per-task=1`. The benchmark is not set up for that.

---

## 7. Open questions

- GPU memory and time of A and B; nothing beyond CPU yet.
- Time cost of A's n_dev smaller reduce-scatters, and real inter-node bandwidth on JUPITER
  (the break-even P in §4).
- Does the JAXMg reshard (1D rows → `MESH_2D` → back) ever replicate? Measure the full
  `get_update` memory with `"distributed"`.
- Multi-process `jax.device_put(..., REPLICATED_SHARDING)` inside `sharded`, which B relies on
  for the samples.
- The `zip` stale-tuple pattern in `LazySampledObs.mean/var/get_covar` (force path) is still
  open.
- XLA windowed einsum (D) as a zero-code alternative to A.
