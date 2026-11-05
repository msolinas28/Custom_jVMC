# Final bug fix + documentation

Handover for continuing the work in a new chat. Last updated 2026-11-05.

## Goal

Before the release, go through the library one folder (or module) at a time and, for each one:

1. Update the documentation: docstrings in the code and the matching Sphinx page in
   `documentation/source/`.
2. Look for bugs and report them **in the chat**, never in the documentation.

The user then answers point by point: "fix it", "leave it", "I fixed it" (verify it), or a
question. Only change code when asked; answer questions with snippets that were tested on a
scratch copy. The user commits after each folder.

## Procedure for one folder

1. Read every file of the folder, its Sphinx page, its tests, and the call sites in the rest of
   the library (`grep`). Check `git status` / `git log` first: the user edits between rounds.
2. Write a reproduction script in the scratchpad for every suspected bug and run it on CPU with
   1 and 3 devices. Only report what was run, or say clearly that something was read, not run.
   Compare against exact results (dense matrices, `ExactSampler`, analytic values) wherever
   possible.
3. Update the documentation (see conventions below), then check that the code is unchanged apart
   from docstrings with the AST script below.
4. Rewrite or create the Sphinx page and build it.
5. Run the related tests.
6. Report: documentation summary, then bugs ranked by severity with file and line, then what was
   checked and found correct.

When the user asks to fix something: fix it, add a test when the fix changes behaviour, rerun the
tests, and say what changed. When the user says "I fixed it": verify it by running, since several
user fixes needed a correction (e.g. `jnp.prod` on a tuple, a MALA wrap at the wrong point,
`to_array` keeping the wrong branch, `.astype` on a `SizedIterable`).

## Documentation conventions

- Sphinx has **no napoleon** (`documentation/source/conf.py`), so no NumPy-style `Parameters` /
  `Returns` sections. Use the style of `vqs.py`:

  ```
  Args:
      * ``name``: Description, continued on the next line with a trailing \
        backslash and aligned indentation.

  Returns:
      Description.
  ```

- Docstrings are normal (non-raw) strings: math needs double backslashes, e.g.
  ``:math:`\\psi(s)` ``.
- Module constants: document with `#:` comments directly above the assignment (autodoc picks
  them up). Module docstrings for modules with configuration.
- **No bugs in the documentation**: no `.. warning::` blocks or "known issue" notes. Docstrings
  describe the intended contract. Where a convention is itself in question, describe what the
  object does in neutral terms (e.g. "maps the local state 0 to 1"). This is also saved in the
  auto-memory (`no-bugs-in-docs`).
- Sphinx pages use full paths, e.g. `.. autoclass:: jVMC_exp.operator.discrete.branch_free.SigmaX`.
  `automodule:: jVMC_exp.sampler` alone shows nothing, because the classes are imported into the
  package. Many old pages still point at `jVMC.*` and fail to import.
- Keep the user's wording where it is still correct; match the surrounding style.

## Tools and commands

- Python with JAX: `~/venv/jVMC_new/bin/python` (jax 0.10.1) on the old machine;
  `~/env/jVMC_env_exp/bin/python` (jax 0.10.2) on the Mac used since 2026-11-05, where pytest is not
  installed: use `python -m unittest tests.<name>_test`. `-m` puts the working directory first on
  `sys.path`, so `cd` into a scratch copy to test it. The system `python3` has no JAX and is only
  good for AST checks and text edits.
- Always set `JAX_PLATFORMS=cpu`; simulate devices with
  `XLA_FLAGS=--xla_force_host_platform_device_count=3`, and run scripts with
  `PYTHONPATH=<repo>`.
- Run test suites **one at a time**: two suites in parallel were killed for lack of memory
  (15 GB machine).
- Scratch copies for prototypes: `cp -r jVMC_exp tests pytest.ini <scratch>/trial/`, patch,
  then run with `PYTHONPATH=<scratch>/trial`. The scratchpad has been wiped between sessions:
  anything worth keeping goes in `.claude/`.
- Sphinx is not installed in the venvs. Install it into the scratchpad only:
  `~/venv/jVMC_new/bin/pip install -q --target <scratch>/sphinxdeps sphinx sphinx_rtd_theme mock`,
  then build one page from `documentation/source`:
  `PYTHONPATH=<scratch>/sphinxdeps JAX_PLATFORMS=cpu ~/venv/jVMC_new/bin/python -m sphinx -b html -E -d <scratch>/doctrees . <scratch>/html <page>.rst`.
  Check the log for lines mentioning the page and count `system-message` in the HTML.
- Code unchanged apart from docstrings (`#:` comments don't appear in the AST anyway):

  ```python
  import ast, subprocess
  def strip(src):
      tree = ast.parse(src)
      for n in ast.walk(tree):
          if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module)) and n.body \
                  and isinstance(n.body[0], ast.Expr) \
                  and isinstance(getattr(n.body[0], "value", None), ast.Constant) \
                  and isinstance(n.body[0].value.value, str):
              n.body = n.body[1:] or [ast.Pass()]
      return ast.dump(tree)
  old = subprocess.run(["git", "show", f"HEAD:{p}"], capture_output=True, text=True).stdout
  print(strip(old) == strip(open(p).read()))
  ```

## Progress

| Folder / module | Docs | Bugs | State |
|---|---|---|---|
| `sampler/` | done, `sampler.rst` rewritten | reported, most fixed | committed (`f61e78f`) |
| `vqs.py` | done, `vqs.rst` rewritten | reported, most fixed | committed (`f2f67ce`, `76a82fc`) |
| `global_defs.py` | done, new `global_defs.rst` in the toctree | reported, fixed | committed (`d52b4e9`) |
| `operator/` | done, `operator.rst` rewritten | reported; transpose bug deferred | committed (`73c2f1b`) |
| `geometry.py` | done, new `geometry.rst` in the toctree | reported, 1/2/5/6 fixed, new `tests/geometry_test.py` | committed (`da8264f`) |
| `sharding_config.py` | done, new `sharding_config.rst` in the toctree, `parallelism.rst` rewritten | reported, 3/6 fixed, 1 deferred (patch ready) | committed (`323ebd1`) |
| `stats.py` | done, `stats.rst` rewritten | reported, 1-4 fixed (2-4 by the user), new AR(1) test | **uncommitted** |

Still to do: `stepper.py`, `nets/`,
`objective_function/`, `optimizer/`, `solver/`, `symmetry/`, `util/`, and the pages `nets.rst`,
`util.rst`, `mpi.rst`, `index.rst`, `examples.rst`.

### Stats, changes of 2026-11-05 (uncommitted)

- Class docstrings moved from `__init__` to the class (autodoc shows only the class docstring);
  docstrings for every public method/property; `stats.rst` rewritten (pointed at `jVMC.stats`,
  used the removed `covar()`), example tested on 1 and 3 devices.
- Bugs reported (numbers as in the chat): 1 `get_autocorrelation_time` FFT length
  `n = ceil(log2(L))` instead of `2 * 2**ceil(log2(L))` (τ from the first ~12 samples);
  2 R-hat/ACT reshape the padded observations (the 8 `stats_test` failures on 3 devices);
  3 `linear_map` multiplies along the sample axis; 4 `weights=None` takes the dtype of the
  observations (complex weights / errors, float16 weights); 5 `transform` keeps `_num_obs`
  (`str()` raises after a shape change), returns `None` against its annotation;
  6 `LazySampledObs.var` / one-pass covariances lose precision for large means; 7 small items.
- Fixed: 1 (FFT length `2 * 2**ceil(log2(L))`, new `test_act_matches_exact_ar1_value`, fails on
  the old code); 2, 3, 4 by the user, verified (all 30 `stats_test` pass on 1 and 3 devices).
  Left: 5 (user: transformations are not meant to change the number of components), 6, 7.
  `.claude/stats_fixes.patch` is the prototype, now applied: can be deleted.
- Checked and not a bug: the ACT window fallback is unreachable (τ at the last lag is 0 for
  mean-subtracted data); denormal weights are flushed to 0 by XLA on CPU (no `1/w` overflow); MC
  samples are stored chain-major on 1 and 3 devices, as R-hat/ACT assume.

### Operator folder, changes of 2026-11-03

- Docstrings for every public class/method/property in `operator/`; `operator.rst` rewritten
  (the old page described `BranchFreeOperator`, `Sx`, `scal_opstr`, POVM, none of which exist).
- `SigmaPlus` and `SigmaMinus` swapped: they were σ⁻ and σ⁺ with respect to `SigmaZ` (state 0 =
  up). `test_repeated_sites_dense_matrix` updated, `test_pauli_and_ladder_operators_dense_matrix`
  added.
- `raise NotImplemented` → `return NotImplemented` (12 places, discrete and continuous `base.py`).
- By the user: `to_dense` passes `zero_tolerance`; continuous `get_O_loc` uses
  `psi.eval_parameters`.

### Geometry, changes of 2026-11-03

- Docstrings for all of `geometry.py`; new `geometry.rst`, added to the toctree after `operator`.
- `uniform_populate`: `jnp.repeat` → `jnp.tile` (start positions were wrong when extents differ).
- `__eq__` also compares the class; `__hash__` added.
- `PBC`: broadcast when `np.ndim(PBC) == 0`, stored as `tuple(bool(p) for p in PBC)`, so lists and
  NumPy / JAX arrays work (a list used to become a tuple of lists).
- Abstract `domain` is now an abstract property; abstract `uniform_populate(key, shape, dtype)`.
- New `tests/geometry_test.py` (4 tests, all fail on the old code).

### Sharding config, changes of 2026-11-03 (uncommitted)

- Docstrings and `#:` comments for all of `sharding_config.py`; new `sharding_config.rst` in the
  toctree; `parallelism.rst` rewritten (old one described pmap and MPI).
- Batched `sharded` calls raise `ValueError` if the positional arguments have different lengths
  (were silently zero padded).
- `ParamSpec` renamed `Params`: `sharding_config.P` is `PartitionSpec` again.
- Two tests added to `tests/batch_layout_test.py`.

## Open items

### Deferred: important

- **`sharded` caches keep objects alive** (user: later release). Static kwargs (`psi`, `operator`)
  in `instance._sharded_cache` keep every NQS alive while the decorated instance lives; with
  `BATCH_LOOP="compiled"` and more than one batch, nothing is ever freed, because the module-level
  `_jitted_call_local` takes `jsh_fn` as a static argument (JAX's global cache keeps it). Tested
  patch: `.claude/sharded_cache_weakref.patch` (weak references to static objects with
  `weakref.finalize` dropping the entry, compiled loop stored next to `jsh_fn` on the instance as
  a closure; a `functools.partial` would leak through JAX's `_infer_params_cached`, whose
  signature keeps `jsh_fn` as a default).

- **Discrete operators: `get_O_loc` and `Infidelity` measure the transpose of the operator.**
  Full analysis, tested patch and timings in `.claude/operator_transpose_bug.md`. Until it is
  fixed, the sampler measures ⟨σ⁻⟩ for `SigmaPlus`.

### Waiting for a decision

- `sharding_config.py`: user did not answer yet whether to delete the `scaled()` TODO comment
  (answered in its docstring) and whether to drop `mpi.rst` from the toctree.


- `global_defs`: `DT_OUT_REAL = f16 / bf16` fails in `solver/util.py` `diagonalize`
  (`jnp.linalg.eigh` has no LAPACK routine for them; the `except` only catches `ValueError`).
  `CG` works. Suggested: promote to at least `f32` in `diagonalize` (not tested).

### Still open in folders already done

- `sharding_config.py` (left by the user): `batch_size=None` fails when the number of samples is
  not a multiple of the devices; `broadcast_split_key` only supports raw threefry keys (intended);
  `static_argnums` unusable; `yield_iter` + custom specs raises with `batch_size=None`. Cosmetic:
  KeyError for a missing `batch_size`, import message says "not set" when set to `false`.
  Multi-process can be tested on CPU: 2 processes with
  `jax.config.update("jax_cpu_collectives_implementation", "gloo")` and
  `jax.distributed.initialize("127.0.0.1:<port>", 2, pid)` before importing the library.

- `geometry.py` (left by the user): lengths of `PBC` / `extent` not checked against `n_dim`;
  `count_particles_in_cell` counts particles outside an open domain in the edge cells (documented).

User chose to leave these, or will patch them in a later release:

- `sampler/propose.py`: `POVMOutcome` crashes (update has shape `(1,)`) and its `% 4` binds to the
  random number (outcomes 4, 5 reachable). The user plans to remove POVM.
- `sampler/mc_sampler.py`: `CutoffSampler` with an `eval_ratio` net raises `UnboundLocalError`
  (`get_ratio` only defined in the `else` branch). Its docstring still has a `.. warning::` from
  the first round.
- `sampler/exact_sampler.py`: docstring still has a `.. warning::` about device divisibility from
  the first round, although `get_probabilities` no longer uses `shard_map`. Check and remove.
- `sampler/base.py`: `sampler_state` setter restores weights but not samples / `logPsi`
  (mismatched lengths possible); MC weights overflow to NaN for `mu < 2` with large `|log ψ|`
  (`CutoffSampler` already subtracts the max); `__call__` with `num_samples` resamples every
  call; `reset(key)` doesn't restart the chains.
- `sampler/propose.py`: `MALA(appy_fun=...)` typo in the keyword; one RWM/MALA proposer can't be
  shared by samplers with different `numChains`.
- `vqs.py`: variables outside the `params` collection unsupported (documented as a requirement);
  `frozen_parameters` path semantics and frozen parameters still in the gradients (user will
  patch later); parameters committed to a single device fail on several devices (left).
- `global_defs`: `MATMUL_CHUNK_BYTES` is fixed at compile time (documented); its env variable
  rejects `2**30` / `1e9`.
- `operator/`: Jordan-Wigner sign counts non-fermionic sites (documented); `CoulombInteraction`
  rejects a list of charges, and a scalar charge `q` scales by `q` while an array scales by
  `q_i q_j`; `ParticleDensity` default `linear_partition=(10, 10)` only fits 2D;
  `laplacian_mode="forward"` accepted but raises at evaluation (`_frwrd_lap_fix.py` needs
  `fwdlap`, not a dependency); `0 - op` raises `NotImplementedError` while `0 + op` works;
  `examples/ex5`, `ex6` use the removed POVM classes (user will remove POVM).

### Noticed while working on other folders, to check when their turn comes

- `optimizer/minsr.py` and `solver/pinv.py` call `grad_log_psi.transform(lambda x: ...)` with a new
  lambda at every step: `transform` wraps it in `jax.jit`, so it compiles again at every step
  (~10 ms each on a (300, 10) array, measured). `optimizer/base.py` stores its functions
  (`self._pad_obs` etc.) and does not recompile. User (2026-11-05): fix when doing `optimizer/` and
  `solver/`.
- Tests: on 3 devices, 32 tests in operator/vqs/minsr/objective_function fail only because their
  batch sizes are powers of 2 ("has to be divisible by the number of devices").
- `objective_function/infidelity.py`: `_get_f_loc_sh` calls `operator._get_conn_elements(s, {})`,
  so operators with callable (time-dependent) prefactors get no keyword arguments.
- `optimizer/base.py`: the reweighting between substeps uses a hard-coded factor
  `2 * jnp.real(logPsi - logPsi0)`, i.e. assumes `logProbFactor = 0.5`; also check the weights are
  normalized.
- `util/util.py`: `measure()` passes `num_samples` to the sampler; with `ExactSampler` it ends up
  in `get_O_loc` kwargs (ignored by real operators).
- `solver/util.py` imports `USE_DISTRIBUTED` by name; fine now that it can only change at import
  (`sharding_config` sets it from `jax.distributed.is_initialized()`).
- Docs: `parallelism.rst` describes `global_defs.set_pmap_devices()` and MPI, which no longer
  exist; `mpi.rst` points at `jVMC.mpi_wrapper`; `nets.rst`, `stats.rst`, `util.rst` use `jVMC.*`.
