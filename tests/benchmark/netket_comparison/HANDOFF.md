# jVMC_exp vs NetKet benchmarks: handoff notes

Notes for continuing this work in a new session. State as of 2026-10-12.

## Goal

Compare `jVMC_exp` (this repo) with the latest NetKet (3.22.4), **as fairly as possible**
("apples to apples" is the user's main requirement). The starting point was a figure like
Fig. 4 of arXiv:2108.03409 (time per stage of the estimate of <H>).

## Files in this folder

| File | What it does |
|---|---|
| `compare_netket.py` | Time per stage of the Monte Carlo estimate of <H> (sampling, matrix elements, evaluation, total), Fig. 4 style. TFIM chain, `CpxRBM` (switched from the paper's CNN at the user's request). |
| `compare_netket_sr.py` | One SR step at fixed N_p with a growing number of samples: time and peak memory, and the largest Jacobian each library handles. Series: jVMC_exp dense Jacobian, jVMC_exp batched Jacobian, NetKet `VMC_SR`. |
| `send_job.sh` | SLURM job (JUWELS booster, 1 GPU): `sbatch send_job.sh [compare_netket.py \| compare_netket_sr.py] [options]`. Time limit 1 h, may be too short for the full SR sweep. |

Both scripts write a new `results_<date>/` (or `results_sr_<date>/`) folder with JSON per run,
a summary table on stdout and a PNG/PDF figure. `--plot <folder>` only re-plots.

Git: `compare_netket.py` and `send_job.sh` were committed by the user in `e37c520`. Not committed
at the time of writing: the refactor of `compare_netket.py` (module-level `jvmc_hamiltonian`,
`netket_operator`, `netket_rbm(..., dtype)`), the new `send_job.sh` usage line,
`compare_netket_sr.py` and this file.

The user edited `run_jvmc` in `compare_netket.py` (batch size `cfg['batch_size'] or cfg["n_chains"]`,
`batch_size` passed to `get_conn_elements`, no device check or sample-count check there): keep
these edits.

## Environment

- NetKet is **not** in the user's local env (`/Users/som30090/env/jVMC_env_exp`, jax 0.10.2,
  flax 0.12.5). `pip install netket==3.22.4` works with jax 0.10.2 without changing jax (checked
  with a dry run). Locally it was tested by installing into a separate folder
  (`pip install --target <dir> netket==3.22.4 beartype einops equinox igraph orjson plum-dispatch sparse texttable --no-deps`)
  and running with `PYTHONPATH=<dir>`, so the user's env was never modified.
- On the cluster NetKet has to be installed in the same env as jVMC_exp (same JAX for both).
- Nothing has been run on a GPU yet. All numbers below are from the user's Mac CPU.

## Fairness rules used in both scripts (agreed with the user)

- **Separate processes**, one per library (and, for the SR sweep, one per point), never both
  libraries in one process: importing NetKet sets a global JAX mesh (axis `"S"`) that breaks
  jVMC_exp's `shard_map` in the same process.
- **One device** by default (`CUDA_VISIBLE_DEVICES` = first visible GPU); `compare_netket.py --all-devices`
  runs both on all devices (NetKet gets a mesh via `jax.sharding.set_mesh`).
- **Same network and parameters**: NetKet runs a batched port of the jVMC_exp net (`netket_rbm`, same
  `log_cosh`, same parameter names `Dense_0/kernel|bias`) loaded with the same parameters; log(psi)
  is checked equal (max diff 0 / 1e-23).
- **Same physics and settings**: TFIM `H = -sum sz sz - g sum sx`, PBC; same chains, samples, sweep
  size (L), thermalization (`n_discard_per_chain` = `thermalizationSweeps`), single-spin-flip
  Metropolis, complex128 / float64.
- **Same memory budget**: `--batch-size` = jVMC_exp `batchSize` = NetKet `chunk_size` (network
  evaluations per call for <H>; samples per Jacobian block in the SR step).
- **Timing**: first call (compile) and warm-ups excluded, `jax.block_until_ready`, median of repeats.
- **SR benchmark specifically** (user's choices):
  - **Real-parameter RBM** (`jVMC_exp.nets.RBM`, NetKet `mode="real"`) so both build the same
    (N_s, N_p) float64 Jacobian and N_p x N_p S. With `CpxRBM`, jVMC_exp solves an N_p complex system
    (holomorphic) while NetKet's `VMC_SR` has no holomorphic mode and solves a 2N_p real one.
  - **Same solver**: jVMC_exp `Pinv` (eigh pseudo-inverse) and the same formula passed to NetKet as
    `linear_solver`; same `diag_shift`.
  - Same uniformly random samples in both (cost is the same as with MCMC samples); sampling excluded.
  - Check: NetKet's update = -2 x jVMC_exp's update; measured relative difference <= 7e-13.
  - NetKet samples are set through the private `vstate._samples` (no public API).
  - Peak memory: GPU `memory_stats()["peak_bytes_in_use"]`; on CPU the process peak RSS is used.
    A series stops at its first out-of-memory point (OOM detection is untested, GPU only).

## Results so far (CPU, not representative of GPU)

**<H> stages** (N=80, CpxRBM alpha=1, 10150 samples, 350 chains): totals 2.26 s (jVMC_exp) vs
2.16 s (NetKet Ising, 0.96x) and 2.18 s (NetKet LocalOperator). Earlier with the paper's CNN:
129 s vs 135 s. The matrix-element stage is 6-14x slower in jVMC_exp, but negligible in the total
for the TFIM.

**SR step** (N_p = 3200, real RBM, 40 hidden units, batch size 4096; time [s] / peak RSS [GB]):

| samples | jVMC dense | jVMC batched | NetKet VMC_SR |
|---|---|---|---|
| 4096 | 2.41 / 1.81 | 3.24 / 2.11 | 2.36 / 1.03 |
| 16384 | 3.42 / 3.49 | 7.11 / 2.32 | 3.56 / 2.06 |
| 65536 | 9.81 / 8.37 | 24.41 / 2.51 | 8.54 / 6.11 |

- Dense jVMC_exp and NetKet: same time, same memory growth per sample (~3.2-3.4 Jacobian-sized
  copies); NetKet lower mainly by a smaller baseline. Expectation was that NetKet fits 1.5-2x more
  samples in the dense mode: not seen on CPU, the GPU run has to decide.
- Batched jVMC_exp: memory flat (no Jacobian-size limit) but 2.5x slower at 2^16.

## Findings about jVMC_exp (explained to the user, not fixed)

1. **Matrix elements are memory-bound**: XLA cost analysis, L=80, 10150 samples: jVMC_exp moves
   3.09 GB (54 ms) vs NetKet IsingJax 0.09 GB (3.8 ms) and LocalOperatorJax 0.63 GB (9.9 ms); the
   output is only 0.28 GB. `branch_free.Operator._get_conn_elements` carries a full copy of the
   configuration through a `lax.scan` for every operator string, diagonal ZZ strings included
   ((N_s, 2L, L) int32 per step), then a boolean-mask gather and a concatenate with `s`. Removing the
   fermion-sign arithmetic or `lax.switch` does not change the runtime (XLA folds them).
2. **Compile time**: `jax.lax.switch(id, self.prefactorsC, kwargs)` in `_get_conn_elements` (one branch
   per operator string, 160 here) is traced in every function that uses the operator. First call of the
   matrix elements 2.17 s (1.55 s tracing) -> 0.24 s with the prefactors as a constant array; local
   estimator 2.30 s -> 0.76 s (of which ~0.6 s is the run). Grows with the number of terms in H.
   Measured with `jax.monitoring.register_event_duration_secs_listener` on
   `/jax/core/compile/{jaxpr_trace_duration,jaxpr_to_mlir_module_duration,backend_compile_duration}`.
3. **Padding**: with `batchSize` larger than the number of configurations, `BatchLayout.take` pads the
   batch up to `batchSize` (1000 configs: `batchSize=8000` is 5.5x slower than `batchSize=1000`).
   That's why `compare_netket.py` defaults the batch size to the number of samples.
4. **Dense Jacobian copies**: `flat_gradient_holo` returns `[g, 1j*g]` (2N_p wide, removed later in
   `get_update`); for real-parameter nets `flat_gradient` computes a complex Jacobian (two backward
   passes, imaginary part zero) that `NQS.gradients` casts to real; `get_covar` and `transform` /
   `_normalized_obs` create full-size copies.
5. **Batched Jacobian** computes the Jacobian twice (once for the force in `Observable.value_and_grad`,
   once for S in `_get_lhs_dense`).

## Things the user decided

- Wanted: the SR-step comparison (apples to apples), and the "largest Jacobian" sweep
  (fixed N_p, growing N_s).
- Not wanted for now: scaling curves of <H> vs size, multi-GPU scaling, TDVP, autoregressive sampling.

## Next steps

- Run both scripts on a GPU (`sbatch send_job.sh compare_netket_sr.py`, then `compare_netket.py`) and
  interpret: does NetKet fit more samples in the dense mode on GPU, how far the batched mode goes,
  how the times compare.
- Possibly: refine the sample grid near the out-of-memory point; raise the job time limit.
- If the user wants: fix findings 1-5 in the library (e.g. prefactors as an array instead of
  `lax.switch`, no configuration copies for diagonal strings, no padding beyond the number of samples,
  no doubled holomorphic Jacobian, reuse of the Jacobian batches for the force and S).
