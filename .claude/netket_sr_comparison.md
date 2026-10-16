# jVMC_exp vs NetKet, SR step: findings and next steps

State on 2026-10-14. These notes, together with the folder `netket_sr_comparison/` next to them, are all you need to continue from scratch: the scripts and patches are in that folder, not in a scratch directory.

The benchmark is `tests/benchmark/netket_comparison/compare_netket_sr.py`. It measures one SR step from given samples:
- **Model:** real RBM with L = 80 spins and 80 hidden units, so P = 6400 parameters, in float64, on the TFIM.
- **Samples:** uniformly random configurations.
- **Solver:** the eigendecomposition pseudo-inverse, the same in both libraries.
- **Hardware:** one GH200, with 76.5 GB available to JAX.
- **Units:** J = N·P·8 bytes is the size of the Jacobian (13.4 GB at 2^18, 26.8 GB at 2^19). P×P = 0.33 GB.

## Environment (JUPITER)

- **Python environment:** `$SCRATCH/env/nk_jvmc` has jax 0.11.2, flax 0.12.10, netket 3.22.4 and jVMC_exp installed in editable mode. All results below come from it.
  - `tests/benchmark/netket_comparison/send_job.sh` activates `$SCRATCH/env/jVMC` instead, which has jax 0.10.1 and no NetKet. Change that line before using it.
- **Modules:** `module load Stages/2026 GCCcore/.14.3.0 CUDA/13 Python/3.13.5 cuDNN/9.19.0.56-CUDA-13 NCCL/default-CUDA-13`.
- **CPU checks on the login node:**
  - Set `JAX_PLATFORMS=cpu`, `JVMC_USE_DISTRIBUTED=false` and `OPENBLAS_NUM_THREADS=32`. Above 64 threads, numpy linalg crashes and writes a core dump of about 25 GB, so also run `ulimit -c 0`.
  - `XLA_FLAGS=--xla_force_host_platform_device_count=4` simulates 4 devices.
- **One GPU per point:** every point runs on one GPU, even when the allocation has more. The driver sets `CUDA_VISIBLE_DEVICES` to the first GPU, `check_single_device()` aborts otherwise, and every JSON records `n_devices: 1`. NetKet does run on several GPUs, as `compare_netket.py --all-devices` shows; the SR script was kept on one GPU on purpose.

## Uncommitted changes on branch `nk-bench`: commit them before moving

1. **`jVMC_exp/sharding_config.py`**
   - The loop over the batches of `_call_local` runs either in Python (`sharded_python`, the original) or inside one jitted function (`sharded_compiled` + `_jitted_call_local`; called `sharded_test` in the results below). Results are bit-identical on CPU, on 1 and 4 devices.
   - `sharded` picks one at every call from `global_defs.BATCH_LOOP`: set `JVMC_BATCH_LOOP=python|compiled` (default `python`) or call `jVMC_exp.global_defs.update(batch_loop=...)` at any time.
   - Lazy calls (`yield_iter`, which the batched Jacobian uses) still loop in Python. They have to, because their batches are consumed outside.
2. **`jVMC_exp/optimizer/base.py`, `_add_qgt`:** J^T·J is now summed over chunks of at most 1 GiB of samples, inside the same function (8 lines).
   - The chunk size is `global_defs.MATMUL_CHUNK_BYTES` (default 1 GiB; `JVMC_MATMUL_CHUNK_BYTES` or `update(matmul_chunk_bytes=...)`; 0 = no chunking, i.e. the code before the patch). The MinSR kernel uses the same constant. It's read when an optimizer compiles these products, so a change applies to optimizers created afterwards.
   - The updates are bit-identical when one chunk covers the samples, and differ by at most 3e-13 with forced small chunks (CPU, 1 and 4 devices, Pinv and PinvSNR).
   - The sharding is preserved: on 4 CPU devices the compiled program has only per-device shapes, and its only collective is the existing reduce-scatter of one block of S rows.
3. **`tests/benchmark/netket_comparison/compare_netket_sr.py`:**
   - `--batch-size` now defaults to `None`, meaning all samples in one batch for jVMC. `--n-samples` goes from 2^10 to 2^19.
   - **Bug in `run_netket`:** `if batch_size <= n_samples: batch_size = None` is always true, so NetKet is never chunked. With `--batch-size None` this happens to match jVMC. With a batch size it doesn't, and the fix is `>=`.
4. `compare_netket.py` and `compare_netket_minsr.py` are also modified, by other work.

Results folders (untracked):
- **`results_sr_20261014_121140`:** batch size 16384, original library.
- **`results_sr_20261014_172839`:** batch size N, with `sharded_test` active and the `_add_qgt` change.

## Findings

### Time: the Python batch loop caused all of it

| N | jVMC E_loc, original | jVMC E_loc, `sharded_test` | NetKet E_loc | jVMC SR, original | jVMC SR, `sharded_test` | NetKet SR |
|---|---|---|---|---|---|---|
| 2^10 | 0.067 s | 0.004 s | 0.000 s | 0.401 s | 0.337 s | 0.327 s |
| 2^14 | 0.054 s | 0.006 s | 0.002 s | 0.425 s | 0.380 s | 0.369 s |
| 2^18 | 0.941 s | 0.040 s | 0.035 s | 1.795 s | 0.906 s | 0.859 s |

- **Cause:** `sharded._call_local` runs one Python iteration per batch with 4 jit dispatches (2 `_take_batch`, the kernel, `_put_batch`). That costs 0.36–0.6 ms of host time per batch, against about 30 µs of GPU work for 16384 RBM evaluations. In the original runs, SR time minus E_loc time was the same as NetKet's within 3–7%. NetKet runs its chunks as a loop inside one jitted function.
  - Up to 2^14 both runs had the same number of E_loc batches (81–86), so the gain at those sizes is purely `sharded_test`.
  - At 2^18 the batch size also changed between the runs (16384 vs N).
- **Drawback of `sharded_test`:** it compiles the network once per number of samples. On CPU the network program compiled again at N = 37 and 128; the original compiles it once, and only recompiles small helpers per N.
  - NetKet also recompiles for every new N.
  - CPU count of compiles at a new N (L = 20, chunk 4096): jVMC makes 13 for E_loc and 7–8 for SR; NetKet makes 1 and 3.
- **Alternative that keeps compile-once:** `netket_sr_comparison/grouped_kernel.patch`, which is CPU-checked but not measured on GPU.
  - It's opt-in through `@sharded(group_batches=True)` on `_apply_fun_sh`, `_apply_ratio_sh`, `_get_O_loc` and the continuous `_evaluate_flat`.
  - Each dispatch evaluates up to 64 batches in a loop inside the compiled function. The compiled function always gets a full group of rows (the last group is zero-padded), so it compiles once per batch size. The number of real batches is passed at run time, so padding batches are not evaluated.
  - On CPU, E_loc and log ψ are identical to `sharded` on 1 and 4 devices, and the network compiles once.
  - Risk: the loop has a run-time trip count, so the GPU may sync the loop condition with the host at every iteration.
- **Why batched and dense take the same time:** the batched path computes the Jacobian twice. For the RBM one Jacobian pass costs O(N·P) operations (about 5 GFLOP at 2^18), against O(N·P²) for S (about 2·10^13 FLOP) and a fixed cost for `eigh`. The extra pass is under 1% of the step.
- **The RBM on TFIM is the worst case for jVMC's loop**, because the GPU work per batch is much smaller than the host cost of a dispatch. Heavier networks shrink the gap; they won't make jVMC faster, since both libraries run the same XLA computations.

### Memory of the dense path

**The mechanism.** On the first call for each shape, XLA autotunes every matrix multiply and reduction:
- **What it allocates:** its own random-filled copy of each operand, with 8 MiB of guard memory on each side.
- **What it doesn't do:** check how much memory is free.
- **On failure:** if one of these allocations fails, the compile fails.
- **Signature in the error:** "Out of memory while trying to allocate 25.02GiB" at 2^19, which is J plus 16 MiB.
- **Dedup:** the two operands of J^T·J get two separate copies.
- **Why NetKet escapes it:** it pays the same copies, but its whole SR step is one jitted function compiled before J exists.

**Peak per stage**, at 2^17 with batch 16384 and the original library (job 2322378, `probe_sr_memory.py stages`):

| stage | default | autotuning off |
|---|---|---|
| Jacobian (J + 2 batch results) | 1.26 J | 1.26 J |
| `SampledObs.__init__` → `_get_mean(J)`, autotuned on a J copy | 2.02 J | 1.26 J |
| force `_get_weighted_covar` | at most 2 J (hidden by the previous stage) | 1.26 J |
| `_normalize`: raw and normalized J both alive | 2.01 J | 2.01 J |
| `_add_qgt`: J + 2 autotuning copies | **3.17 J** | 2.01 J |
| solve | J + about 7 P×P | |

**Other measurements:**

| measurement | result |
|---|---|
| current dense path, autotuning off, 2^19 | fits: 53.9 GB = 2.01 J |
| NetKet, 2^18 default | 3.05 J |
| NetKet, 2^18 autotuning off | 2.03 J |
| NetKet, 2^19 default | 2.03 J |
| cost of autotuning off | jVMC 2^17 SR 0.84 → 1.08 s; NetKet 2^18 0.85 → 1.41 s |
| old block prototype (S and force built from row blocks of the stored J, Python-level, batch 16384) | 2.02 J from 2^17 to 2^19; 2^20 ran out of memory in the `_get_mean` autotuning (50.02 GiB = J + 16 MiB) |

Notes on these:
- In steady state both libraries peak at 2 J. The copy made by `_normalize` is therefore not a difference from NetKet, but it can be removed.
- Turning autotuning off is not a fix, because of its cost.
- The block prototype was **rejected as a design.** The dense path should stay plain matrix multiplies, changed only by local patches.

**Small N** (job 2322952, at 2^10):
- **Measured:** jVMC peaks at 7.17 P×P (2.35 GB), NetKet at 6.0 P×P (1.97 GB).
- **Where:** jVMC's peak is reached inside `Pinv._diagonalize`, during the eager `jnp.linalg.eigh(A)`, both with and without autotuning. At that point `A` is still referenced, by `get_update` and by the solver call, and the normalized J is still alive.
- **Not the cause:** the diagonal-shift copy (`S.at[].add`) and the Vᵀ copy.
- **NetKet:** its `eigh` runs inside its single jitted function.
- **Scaling:** the extra grows as P².

**Ideas examined and dropped:**
- **Compiling ahead of time at initialization:** `jax.jit(f).lower(ShapeDtypeStruct).compile()`.
  - In JAX 0.11.2 the normal `f(J)` call then compiles again (tested on CPU). Only calling the returned `Compiled` object reuses it.
  - The persistent compilation cache didn't bridge the two in a quick test.
  - The shape must be known in advance, and the result is capped at 2 J.
- **Fixing the root cause:** it's in XLA's autotuner, so it would need an XLA change. Options:
  - fall back to the default kernel when the copies don't fit;
  - benchmark a smaller version of the problem;
  - choose kernels from cost models;
  - benchmark at the first run on the real buffers.
- **Existing XLA flags close to a root fix (untested):** `--xla_gpu_dump_autotune_results_to` / `--xla_gpu_load_autotune_results_from` and `--xla_gpu_per_fusion_autotune_cache_dir`. They skip autotuning for shapes already tuned.
- **No flag caps the size of the autotuning copies** in this jaxlib.

### Result of the stage probe (jobs 2325612/15/18): the chunked `_add_qgt` does not work on GPU

Stage probe at 2^17 with the current library (`sharded_test` active, `_add_qgt` in chunks), batch = N and batch 16384:

| stage | peak, batch = N | peak, batch 16384 |
|---|---|---|
| Jacobian | 1.03 J | 1.02 J |
| `_get_mean(J)` (autotuned on a J copy) | 2.02 J | 2.02 J |
| `_normalize` | 2.02 J | 2.02 J |
| `_add_qgt` in chunks | **3.49 J** (+1.46 J) | **3.49 J** (+1.46 J) |

- **The batch size is not the cause:** both profiles are identical, and the Jacobian program at batch = N adds nothing.
- **The 2^19 error, in full:** the instructions that failed are `gemm_fusion_dot_general` with `kind: __triton_gemm`. XLA fused each chunk's `slice` into the matrix multiply, so the operands of the fusion are the whole J (`f64[524288,6400]` × 2), and the autotuner allocated two full copies again (25.02 GiB each). Chunking inside the jitted function is therefore defeated on GPU. The CPU backend doesn't build these fusions or autotune, which is why the CPU checks didn't show it.
- **Candidate fix, CPU-checked, not run on GPU:** `netket_sr_comparison/add_qgt_barrier.patch` adds `jax.lax.optimization_barrier((r, g))` after the slices. The chunks are then materialized (2 × 1 GiB of temporary memory), so the multiply's operands are chunk-sized. On CPU the updates are identical with one chunk and differ by at most 3e-13 with forced chunks, on 1 and 4 devices.
- **The same applies to `mean_force_normalize.patch`:** its chunked `_get_mean` / `_get_weighted_covar` would very likely be fused the same way. Add the same barrier before relying on it.

## Next steps, in order

1. **Test the barrier on GPU:** apply `add_qgt_barrier.patch` (or put a patched copy first in `PYTHONPATH`), then run `probe_sr_memory.py stages --n 131072` and `dense --n 524288`.
   - Expected: `_add_qgt` no longer raises the peak, so the peak is the 2.02 J of `_get_mean`, and 2^19 fits.
2. **If it works, do the same for `_get_mean`, the force and `_normalize`:** use `mean_force_normalize.patch` with the barrier added to `_sum_over_rows`. Expected peak: about 1 J, so 2^20 should fit.
   - `_get_mean` and `_get_weighted_covar` sum over chunks of at most 1 GiB of samples (helper `_sum_over_rows` in `stats.py`); on 4 CPU devices this compiles with no all-gather.
   - `_normalize` works in place when the `SampledObs` owns J. `SampledObs(..., owned=False)` is new, and `Observable` and `Infidelity` pass `owned=True` for the Jacobian they create.
   - Ownership is opt-in because `jax.device_put` returns a new object that shares the caller's buffer, so donating it would delete the caller's array (verified).
   - The patch was CPU-checked as part of the earlier full dense patch: updates identical with the default chunk, at most 4e-13 apart with forced chunks, on 1 and 4 devices, with no donation warnings.
3. **If the barrier doesn't work:** fall back to the saved autotuning results (`--xla_gpu_dump_autotune_results_to` / `--xla_gpu_load_autotune_results_from`), or accept 3 J.
4. **Small N:** run `eigh` in a jitted function that donates `A`. Not written yet.
5. **E_loc:** choose between `sharded_test` (recompiles per N, measured) and the grouped kernel (compiles once, needs a GPU measurement).
6. **Benchmark:**
   - fix the NetKet chunking condition if a batch size is used, then rerun;
   - compare on a heavier architecture: wrap a jVMC per-sample module with `nn.vmap` for NetKet, so both run the same code and parameters. Candidates: a CNN or ViT from `jVMC_exp.nets` with P around 10^4–10^5, plus a holomorphic case, where jVMC solves P complex unknowns against NetKet's 2P real ones;
   - optionally, multi-GPU with an `--all-devices` option as in `compare_netket.py`.
7. **Optional:** report the autotuner issue upstream to XLA with a minimal reproducer: `jax.jit(lambda J: J.T @ J)` with J larger than a third of GPU memory runs out of memory on the first call.

## MinSR (compare_netket_minsr.py), 2026-10-15

Run `results_minsr_20261014_171328`: N = 4096, H = 16 to 8192, batch size = N. At H = 4096 the peaks were jVMC dense 3.04 J, jVMC batched 4.04 J, NetKet 2.05 J in both modes. jVMC ran out of memory at H = 8192 and NetKet did not.

**Causes**, read from the code. They add up to the measured 3 J and 4 J, but are not yet probed on GPU (`probe_minsr_memory.py` is ready):
- **Dense:**
  - `_blockwise_tangent_kernel`: T = O·Oᵀ on the full O is autotuned on two copies of O at the first call (+2 J). This sets the 3 J.
  - The SR causes also apply: `_get_mean(J)` +1 J at the first call, `_normalize` +1 J, and probably `_adjoint_dot` +1 J at the first call.
- **Batched:**
  - With batch = N it holds the whole Jacobian.
  - The diagonal block (l == r) went through `_get_double_tangent_kernel(batch_l, batch_r)` with a second, recomputed copy of the same batch, plus the autotuner's 2 copies: 4 J.

**Why NetKet's "on the fly" uses the same memory:** `srt_onthefly` builds T with `nt.empirical_ntk_by_jacobian` ("directly instantiating Jacobians and contracting"). Chunking only splits the columns of T, while the left operand stays the Jacobian of all samples.
- CPU, compiled temporary memory, N = 1024, P = 10240: NTK 2.00 J with and without chunking; on the fly 2.11 J unchunked and 2.24 J with `chunk_size=128`.
- `compare_netket_minsr.py` has the same NetKet chunking bug as the SR script (`batch_size <= n_samples`).

**Applied, uncommitted, in `jVMC_exp/optimizer/minsr.py`:**
1. **`_blockwise_tangent_kernel`:** T is summed over chunks of at most 1 GiB of parameters (columns), with `optimization_barrier`, as in `_add_qgt`.
2. **Batched path:** the diagonal block uses `_get_tangent_kernel(batch_l)` (single operand). Batch r is still taken from the iterator, to keep it in step, and freed at once.
   - For the benchmark, run with `--batch-size 1024`; with batch = N the batched path cannot save anything.
- **CPU checks** (`netket_sr_comparison/check_minsr.py`: real RBM and CpxRBM, dense and batched, 1 and 4 devices, padding):
  - With 1 GiB chunks the updates are identical to the previous code.
  - With forced small chunks they differ by at most 3e-13.
  - On 4 devices the kernels have the same single reduce-scatter.
  - With several chunks the CPU build needs about two more blocks of T as temporary memory. GPU unmeasured; XLA usually folds `local + dot` into the multiply.

**Not done yet:** `_adjoint_dot` (Oᵀx) on the full O, `_get_mean` and `_normalize`, which are the same as SR steps 2–3, and the NetKet chunking bug.

**To confirm on GPU:**
```bash
python -u probe_minsr_memory.py stages --hidden 4096 --out OUT              # OUT: config.json + params_<H>.npz of a minsr run
python -u probe_minsr_memory.py stages --hidden 4096 --out OUT --batched
```

### Sweeps with per-series batch sizes (2026-10-15)

`results_sr_20261015_175402` and `results_minsr_20261015_175402`. Only the batched series (SR), or the batched and "on the fly" series (MinSR), use the batch size: 2^14 for SR, N/4 = 1024 for MinSR. Library state: `sharded_test`, chunked `_add_qgt`, MinSR fixes 1 and 4.

- **SR:**
  - jVMC dense and NetKet are both at about 2.0 J and both fit up to 2^19; both run out of memory at 2^20.
  - jVMC batched stays flat at 3.6–4.0 GB and runs 2^20 in 2.57 s.
  - Time at 2^19: dense 1.423 s, batched 1.447 s, NetKet 1.378 s.
- **MinSR** (N = 4096):
  - jVMC dense is at 2.01 J (fix 1 works); it runs out of memory at H = 16384.
  - jVMC batched is at 1.25 J, about 5 batches of N/4 alive, not the about 0.5 J I predicted. This is not probed yet.
  - NetKet NTK is at 2.04 J and runs out of memory at H = 16384.
  - NetKet "on the fly" is at 2.02 J, but **1.27 J at H = 16384, where it fits.**
  - Time at H = 8192: jVMC dense 0.690 s, NetKet 0.746 s.
- **Both dense jVMC failures** (SR 2^20, MinSR H = 16384) are the `_get_mean` reduction autotuned on a copy of J (J + 16 MiB). That's the next limit, and step 2 of the next steps addresses it.
- **Correction, autotuning failure modes:** XLA's cuBLAS autotuner *skips* when its copies don't fit (only warnings, and the run continues). XLA's fusion autotuner (`config_assigner`: reduction fusions, Triton GEMM fusions) *fails* instead.
  - That's why NetKet SR drops from 3.05 J at 2^18 to 2.03 J at 2^19, and why NetKet "on the fly" fits at H = 16384 at 1.27 J.
  - The earlier statement that "on the fly" gains nothing from chunking (based on CPU compiled memory) was wrong for the run-time peak. Its 2 J at smaller sizes is first-call autotuning.

## How to run

### Stage probe (GPU)

`netket_sr_comparison/probe_sr_memory.py` reuses `run_jvmc` / `run_netket` from `compare_netket_sr.py`, and writes into the `--out` folder.

| mode | what it runs |
|---|---|
| `stages` | dense path; records memory after every stage |
| `dense` | dense path, as the benchmark runs it |
| `blocked` | the old block prototype, applied as a monkeypatch |
| `netket` | NetKet, as the benchmark runs it |

- **The `--out` folder:** it must contain `config.json` and `params.npz` from a `compare_netket_sr.py` run. The batch size used is the one in that `config.json`. Optional `ref/update_<series>_<n>.npy` files are compared with the new update.
- **Each point needs its own process**, because the peak memory of a process can't be reset.

The probe job creates both inputs from one cheap benchmark point, then runs the probes:

```bash
cd Custom_jVMC/.claude/netket_sr_comparison
sbatch probe_job.sh            # or run its lines in an interactive allocation with one GPU
```

By hand:
```bash
python -u ../../tests/benchmark/netket_comparison/compare_netket_sr.py --series jvmc-dense \
    --n-samples 1024 --batch-size 0 --out-dir $PWD/out_batch0       # 0 = all samples in one batch
python -u probe_sr_memory.py stages --n 131072 --out out_batch0
XLA_FLAGS=--xla_gpu_autotune_level=0 python -u probe_sr_memory.py stages --n 131072 --out out_batch0 --tag _noautotune
python -u probe_sr_memory.py dense --n 524288 --out out_batch0    # prints the full error if it runs out of memory
```

**Reading the output:**
- **`call k`:** call 0 is the first local-energy-only stage, call 1 is the first SR step (which includes compilation and autotuning), and later calls are the steady state.
- **`in use`:** device memory in use after the stage.
- **`peak`:** the process peak so far.
- **`<- peak +x J`:** this stage raised the peak.
- **Units:** all numbers are in units of J for the probed N.
- **Memory only:** the probe synchronizes after every stage, so its timings are not representative.

### Testing a patch without changing the library

Copy `jVMC_exp` to another folder, apply the patch there, and put that folder first in `PYTHONPATH`:
```bash
PYTHONPATH=/path/to/copy python -u probe_sr_memory.py ...
```
The copy then shadows the editable install. Every probe in these notes used this.

To apply a patch to the library instead:
```bash
git apply .claude/netket_sr_comparison/<name>.patch      # undo with: git apply -R <name>.patch
```
Both patches apply cleanly on top of the uncommitted changes above. The grouped patch still applies after the split of `sharding_config.py` into `sharded_python` / `sharded_compiled` (checked 2026-10-16); it changes the shared base class, so it applies to both batch loops.

### CPU correctness check (before and after a change)

```bash
export JAX_PLATFORMS=cpu JVMC_USE_DISTRIBUTED=false OPENBLAS_NUM_THREADS=32 XLA_FLAGS=--xla_force_host_platform_device_count=4
cd Custom_jVMC/.claude/netket_sr_comparison
python check_dense.py before.npz                     # current code
# ... apply the change ...
python check_dense.py after.npz --ref before.npz     # prints relative differences of the SR updates
```
The check covers Pinv and PinvSNR with SNR on, and padding of samples (2002) and of parameters (420) on 4 devices. It also reports donation warnings.

## Files in `netket_sr_comparison/`

| file | what it is |
|---|---|
| `probe_sr_memory.py` | GPU memory probe per stage (see above) |
| `probe_job.sh` | SLURM job for next step 1 |
| `check_dense.py` | CPU correctness check of the dense SR updates |
| `mean_force_normalize.patch` | next step 2: `_get_mean` / `_get_weighted_covar` in chunks + `_normalize` in place (`owned=True`) |
| `add_qgt_barrier.patch` | next step 1: `optimization_barrier` on the chunks in `_add_qgt` |
| `probe_minsr_memory.py` | GPU memory probe per stage for MinSR |
| `check_minsr.py` | CPU correctness check of the MinSR updates (dense and batched, real and holomorphic) |
| `grouped_kernel.patch` | E_loc in groups of up to 64 batches per dispatch, compiled once per batch size |
