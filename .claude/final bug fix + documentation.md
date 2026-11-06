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
| `stats.py` | done, `stats.rst` rewritten | reported, 1-4 fixed (2-4 by the user), new AR(1) test | committed (`aa6544d`) |
| `symmetry/` | done, new `symmetry.rst` in the toctree after `nets` | reported, 2 fixed (+2 tests), rest left | committed (`aaf56e1`) |
| `objective_function/` | done, new `objective_function.rst` in the toctree after `sampler` | reported, 1 fixed by the user (corrected), new test, rest left | committed (`2a05a0a`) |
| `optimizer/` | done, new `optimizer.rst` in the toctree after `objective_function` | MinSR recompilation, D, E fixed (+2 tests); A by the user; B left; C, F explained, waiting | committed (`890a558`) |
| `stepper.py` | done, `util/stepper.rst` moved to `stepper.rst`, in the toctree after `optimizer` | 1 (ex2) and 3 fixed (+2 tests), 2, 4-7 left | committed (`85ce561`) |
| `util/` | done, `util.rst` rewritten, the four `util/*.rst` removed (`git rm`) | reported (1-10); 3 removed by the user (`init_net`), 5 moved to the transpose bug, 4, 6-10 left; 1, 2 explained, waiting | **uncommitted** |

Still to do: `nets/`,
`solver/`, and the pages `nets.rst`,
`mpi.rst`, `index.rst`, `examples.rst`.

### Util, changes of 2026-11-06 (uncommitted)

- Docstrings for every public function/class/method of `util.py`, `grads.py`, `key_gen.py`,
  `input_manager.py`, `output_manager.py`, module docstrings, `#:` comments for `ObservableEntry`
  and `OperatorWithKwargs`. `OutputManager`/`InputManager` docstrings were NumPy style (no napoleon):
  rewritten. `measure` docstring: the "list of operators" form is gone (the code never accepted it;
  `ObservableEntry` and the optimizer docs don't have it). The docstring of
  `matrix_to_jvmc_operator` was after the local import, so not a docstring: moved above it (AST
  check that drops all bare strings: code unchanged in the six files).
- `util.rst` rewritten (sections: measurements, output/input with an SR + `InputManager` example,
  operators from matrices, S-norm, internal helpers; the `init_net` section was dropped when the
  user removed the function); `util/minsr.rst`, `util/tdvp.rst`,
  `util/symmetries.rst`, `util/output_manager.rst` removed (pointed at `jVMC.util.*`; covered by
  `optimizer.rst`, `symmetry.rst`, `util.rst`). Snippets run on 1 and 3 devices; no Sphinx warning
  for the page.
- Bugs reported (numbers as in the chat): 1 `OutputManager.save_to_h5()` raises TypeError for a
  `None` in the metadata, and `SR`/`TDVP` with the default `PinvSNR` (`snr_tol=0`, or any ExactSampler)
  always store `snr=None` with `save_meta_data=True`; the default `append=False` truncates the file
  first, so the incremental file loses everything after the failing key (parameters gone);
  static `write_metadata(x=None)` fails too; 2 `NQS(seed=None)` on several processes: AssertionError
  in `jax.device_put` (each process draws its own `generate_seed()`; seed=123 works);
  3 `init_net`: `"net2"` calls `TwoNets(net1=, net2=)` (field is `nets`), mutates `descr` (2nd call
  KeyError), tuple `actFun` KeyError, int `dims` TypeError, only used by POVM examples;
  4 `OutputManager(path)` default `append=True`: rerunning a script appends to the old datasets
  (`times` = [0 1 2 0 1 2]); 5 `matrix_to_jvmc_operator` builds Mᵀ (`to_dense` = Mᵀ), which the
  deferred get_O_loc transpose bug turns back into M: fix both together; 6 `check_network_parameter_shape`
  is always True (dict accepted in memory, fails in the file); 7 `format_key(jnp.int32(5))` returns
  the scalar (invalid key), NumPy copy of a key TypeError, float seed truncated; 8 `measure` drops
  the imaginary part (non-Hermitian observables); 9 `pick_gradient` decides holomorphy at one input
  (a non-holomorphic net whose gradient vanishes there counts as holomorphic; `NQS` uses a random
  normal input, so unlikely); 10 small: `InputManager.get_network_checkpoint` KeyError instead of
  RuntimeError without checkpoints, `write_dataset(group="")` fails with a path, static metadata with
  the name of a series breaks the series, observable named `times` collides, ragged observables fail
  in `save_to_h5`, `matrix_to_jvmc_operator` accepts `sites=(1, 1)`, `get_iterable` unused.
- Answers (2026-11-06): 3 `init_net` removed by the user (unused now): its section in `util.rst`
  and the mention in the module docstring removed. Left behind: `import ... activation_functions as
  act_funs` in `util.py` (only `init_net` used it); `examples/ex5`, `ex6` still call `init_net` (POVM
  examples, to be removed by the user); `util/__init__.py` is back to HEAD, so its package docstring
  is gone (not rendered by `util.rst`). 4 leave. 5 fix together with the transpose bug: recorded in
  `.claude/operator_transpose_bug.md` (new paragraph at the top; the patch refreshed, the
  `Infidelity` hunk now uses `op_kwargs`, `git apply --check` passes). 6, 7, 9, 10 leave.
  8 intended (real parts), leave.
  1 explained: `_as_h5_array(None)` returns `None` (`jnp.asarray(None)` raises, the fallback
  `device_get` passes it through); the in-memory series is `[None, None, ...]` and
  `parent.create_dataset(data=...)` at `_write_dict_to_h5` (`output_manager.py:384`) raises
  `TypeError: Object dtype dtype('O') has no native HDF5 equivalent`; static case line 224
  (`One of data, shape or dtype must be specified`). Streaming already skips `None`
  (`_append_h5`, line 125), which is why only `save_to_h5()` fails. Proposed: skip `None` in
  `_write_dict_to_h5` and static `write_metadata` like streaming does, and convert the whole tree
  before opening the file (and before `del parent[key]`), so a failure leaves the file untouched;
  optionally don't report `snr` when the SNR cutoff is off (solver pass).
  2 explained: seeds are per process, not per device. Proposed: `generate_seed()` draws on process 0
  and `multihost_utils.broadcast_one_to_all`; all processes must call it (they do: `NQS`, samplers).
  Prototyped in the scratchpad on 2 processes: same seed, `NQS(seed=None)` works, MCSampler fine;
  single process returns the value. The sampler already uses only process 0's key
  (`broadcast_split_key`), so nothing changes there.
- Checked: `s_norm_fn` = dense reference (variance of G·v) to 1e-16 for holomorphic, non-holomorphic
  and real-parameter nets, Pinv and CG (callable qgt), padding on 3 devices; works as `normFunction`
  of `time_evolution`. All four flat gradients match finite differences; holomorphy detection right
  in complex64 and for large gradients. `measure` with time-dependent operators and
  `num_samples` + ExactSampler is harmless (callable prefactors must take `**kwargs`). Layout
  helpers consistent with `parameters_flat` and the holomorphic gradients. OutputManager/InputManager
  roundtrip of parameters and observables, groups, checkpoints by time; `_as_h5_array` of global
  arrays works on process 0 alone (2 processes). MCSampler(key=None) on 2 processes: fine.
- Tests: `output_manager_test` OK on 1 and 3 devices (run from the scratchpad: it writes
  `test_output.h5` in the cwd), `infidelity_test`, `vqs_test` OK.

### Stepper, changes of 2026-11-05

- Module docstring (ODE, list of steppers, right-hand side contract `f(y, t, **rhsArgs, intStep=k)`,
  usage); docstrings for every class/method/property, `Initializer arguments:` / `Arguments:` ->
  `Args:`; the step-size controller formula in `AbstractAdaptiveStepper`; the `intStep` values and
  stage times of every scheme in its `step()`. The RK23/RK45 "FSAL not exploited yet, see TODO"
  sentence reworded to what the code does. AST check: code unchanged.
- `git mv documentation/source/util/stepper.rst documentation/source/stepper.rst` (pointed at
  `jVMC.util.stepper`), rewritten, added to `index.rst` after `optimizer`; `optimizer.rst` links to
  it. Page snippets (module usage, TDVP + RK23, SR + `Euler(optax.exponential_decay(...))`) run on
  1 and 3 devices; Sphinx build has no warnings for the page.
- Bugs reported (numbers as in the chat): 1 `examples/ex2_unitary_time_evolution.ipynb` passes
  `tol=` to `AdaptiveHeun` (TypeError; runs to t = 2 with `rtol=0., atol=integratorTol`);
  2 `atol=0` with `y = 0`, or `rtol = atol = 0`: every attempt rejected, dt -> 0 -> NaN, then the
  misleading "right-hand side is NaN" RuntimeError after 40-91 evaluations; 3 Heun/AdaptiveHeun/
  RK23/RK45 promote float32/complex64 `y` to float64/complex128 (float64 `jnp.array` tableaux; Euler
  keeps the dtype; harmless in the optimizers, the parameter setter casts back); 4 validation:
  `max_step=0` accepted (dt = 0, `time_evolution` never ends), negative `rtol`/`atol` accepted,
  `Euler.update_dt` accepts dt <= 0, `min_step`/`max_step` setters don't recheck min < max;
  5 custom `normFunction` returning a Python float: ZeroDivisionError when the error is exactly 0;
  6 question: only `Euler` has `update_dt`, so `ground_state_search` rejects the other steppers;
  7 cosmetic: `RK23/RK45._k0` stored but never read.
- User: fix 1 and 3, leave 2, 4, 5, 7; 6 is the intended usage (only `Euler` in ground state
  searches). Fixed 1: ex2 uses `AdaptiveHeun(timeStep=dt, rtol=0., atol=integratorTol)` (stored
  outputs not re-run). Fixed 3 as the user suggested, with two adjustments: `A` and `b` of the four
  tableaux are cast to `global_defs.DT_OUT_REAL` at construction, `c` is not (a bfloat16 `c` rounds
  `1.5 + 0.75e-3` to `1.5`: the stage times would be lost); `_adaptive_step_control` returns
  `float(dt)`, because with the cast `self.dt` became a float32 array and `t + self.dt` float32
  stage times. New `TestDtype` in `tests/stepper_test.py` (both tests fail on the committed code;
  the time test also fails with the cast but without `float()`). Module docstring says the
  coefficients follow `DT_OUT_REAL` and the times stay 64-bit.
- Checked: local error slopes on an inhomogeneous linear ODE (Euler 2, Heun 3, AdaptiveHeun 3,
  RK23 4, RK45 6; error estimates 3, 3, 5, so the exponent -1/(p+1) is right); intStep sequences;
  all five steppers in `TDVP.time_evolution` (ExactSampler, `resample_stepper` True/False, 1 and 3
  devices, identical results, energy conserved to 1e-3 ... 5e-6); `normFunction` reaches the
  stepper through `time_evolution` kwargs. Tests: `stepper_test` OK on 1 and 3 devices,
  `tdvp_test` OK (1 device). Scripts were in the scratchpad (`order.py`, `edge.py`, `tdvp_all.py`).

### Optimizer, changes of 2026-11-05

- User answers: A made `stepper` a required argument of `time_evolution` (verified; also changed
  the TDVP.ground_state_search message, which had it reversed: corrected); B leave (docstring now
  says the final `"energy"` entry repeats the last step); C, F asked for explanations (given in the
  chat, waiting for a decision); D, E fixed:
  - D: `TDVP.cross_validation` computes S2 first and then `_get_tdvp_error(update_1, F2, var2)`.
    New `TestCrossValidation` (dense reference; fails on the committed code, 4.4e-15 vs 1.43). It
    needs biases in the RBM: without them, spin flip maps the even basis states onto the odd ones.
  - E: reweighting uses `/ sampler.logProbFactor` instead of `2 *`. New `TestSubstepReweighting`
    (logProbFactor 0.5 and 1; fails on the committed code).

- Docstrings for every public class/method/property of the five files; new `optimizer.rst`
  (SR ground state search + TDVP with AdaptiveHeun and an observable, tested on 1 and 3 devices).
- Fixed (pre-approved by the user): MinSR passed a new lambda to `transform` at every step,
  1-2 compilations per update on 3 devices (26-40 ms -> 5-6 ms per update); now
  `self._pad_obs` / `self._unpad_obs` jitted once in `__init__`. The same fix is still due in
  `solver/pinv.py` (`_snr_step`).
- Bugs reported: A `time_evolution` default `stepper=Euler` is the class: TypeError without a
  stepper; B last `"energy"` entry of `time_evolution` repeats the previous step (self.o_loc not
  recomputed after the final `sampler.sample()`), and the last time overshoots `t_max`;
  C MinSR for holomorphic nets: update complex in the real layout, effective step = SR with
  diagonal shift lambda/2 (matches to 7e-14); for real nets the update is complex with zero
  imaginary part; D TDVP cross validation: the reported validation TDVP error is the error of
  the first-half update on the first half (in-sample, identical digits), not on the second;
  E reweighting between substeps (`resample_stepper=False`) hard-codes |psi|^2 (logProbFactor
  0.5): weights off by 0.13 for logProbFactor=1; F matrix-free lhs with make_real=False and
  diag_scale differs from the dense one (2e-2); small: TDVP.ground_state_search error message
  talks about SR, tqdm total/updates in time_evolution, Evolution warns at every construction.
- Checked: both make_real conventions reproduce exact real-time evolution of fully expressive
  nets (real-parameter and holomorphic, 1 - F ~ 1e-11); QGT (dense, batched, padded on 3 devices)
  equals get_covar; MinSR = SR for real and non-holomorphic nets; default solver instances hold
  only configuration (sharing them is harmless).

### Objective function, changes of 2026-11-05

- Docstrings for every public class/method/property of `base.py` and `infidelity.py`; new
  `objective_function.rst` (example with SR + Observable and Infidelity, tested on 1 and 3 devices).
- Bugs reported: 1 `Infidelity` drops the keyword arguments (the optimizer's `t`) and calls
  `operator._get_conn_elements(s, {})`: a time-dependent operator raises `KeyError: 't'`;
  2 control variates are biased for a non-unitary operator (1 - i dt H, dt = 0.2: exact 0.754,
  simple 1.139, adaptive 0.792), silently; 3 adaptive coefficient is 0/0 = NaN when R^psi is
  constant (psi = chi with the identity), and the NaN stays in the moving average; 4 small items
  (Estimator.__call__ evaluates estimator_fn on all samples at once, no batching; ParametricObservable
  has no operator/estimator_fn properties; `control_variate` setter calls `.lower()` on non-strings;
  `moving_average_width=0` gives NaN; Estimator caches the gradient kind of the first psi).
- Checked: force formulas (derivation of R_k), all three gradients vs finite differences (tests),
  Infidelity estimator algebra and CV algebra (simple/adaptive), default identity works for
  spinful states regardless of `l_dim`, Estimator gradient-kind detection on a vanishing first
  sample still matches finite differences. Infidelity uses the transposed operator like
  `get_O_loc` (deferred operator bug); `matrix_to_jvmc_operator` builds sigma_y transposed, which
  compensates for it, so the tests with complex unitaries pass.
- 3 devices: the test suites only fail with the known batch size 16 not divisible by 3.
- User: fixed 1 (keyword arguments `**op_kwargs` passed through `__call__`, `_get_f_loc`,
  `_get_f_loc_sh`), leave 2, 3, 4. The fix called `_get_conn_elements(s, **op_kwargs)`, which
  takes a dict: every Infidelity call raised TypeError (15 of 16 tests); corrected to
  `_get_conn_elements(s, op_kwargs)`. New `test_time_dependent_operator` (fails on the committed
  code with `KeyError: 't'`).

### Symmetry, changes of 2026-11-05

- Docstrings for every public class/function/property; module docstring of `lattice_symetries`
  rewritten (site enumerations); new `symmetry.rst`, example tested on 1 and 3 devices.
- Dense checks (scripts were in the scratchpad: build P from `transformed_states`/`weights`,
  check P^2 = P, P = P^+, [P, H] = 0 with real symmetric library Hamiltonians, sum over q = 1):
  translations (all q), D4/D2, spin flip, compositions all exact for spins, spinless and spinful
  fermions. Composed elements are exactly A_l A_r. Translation element: A c_j A^+ = c_{perm[j]};
  psi_P(s shifted by +a) = e^{+2 pi i q a / L} psi_P(s) for spins.
- Bugs reported: 1 fermionic particle-hole C^2 = (-1)^(M(M-1)/2) (M modes), so (1 + C)/2 is not a
  projector for spinless L = 2, 3, 6, 7, ... and spinful odd L; 2 `ProjectedOrbitNet` flattens the
  configurations (2D CNN with sampleShape (L, L) crashes; `sample()` returns flat configs);
  3 `compose_projectors` re-indexes `perm_sign` by output site, single elements use input sites
  (antiperiodic translation x reflection breaks); 4 `local_phase` ignored for spins/bosons;
  5 `sep` average overflows for |Re log psi| > ~355; 6 `exp` average turns NaN/+inf into zero
  amplitudes; 7 small items. Prototype fixes for 1, 2, 3, 5: `.claude/symmetry_fixes.patch`
  (against `HEAD` before the docstrings), all dense checks pass, tests unchanged. Check scripts:
  `.claude/symmetry_checks/` (run as `python symm_ph.py <dir of the scripts>`, with `PYTHONPATH`).
- User: fix 2 only, leave 1, 3, 4, 5, 6, 7. Fixed 2: `ProjectedOrbitNet.__call__` reshapes the
  transformed configurations to `s.shape` (also for the coefficient nets), `sample()` returns the
  shape of `base_net.sample()`. New tests `test_projected_orbit_net_keeps_the_shape_...` and
  `..._sample_keeps_the_shape_...` (fail on the old code). Docstrings of 1, 3, 4 reworded to what
  the code does (particle-hole sign formula, compose `perm_sign` formula, `local_phase` only for
  fermions). The patch still holds the left prototypes 1, 3, 5 for a later release.
- Not a bug: translation(q=0) x particle-hole for spinless even L is not a projector because T and
  C anticommute there (T|full> = -|full>); composition = P_l P_r is a projector only if they commute.

### Stats, changes of 2026-11-05

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

- `nets/rnn1d_general.py`: `NQS(RNN1DGeneral(...))` fails at evaluation on CPU (1 device too):
  "scan body function carry input and carry output must have equal types ... {V:devices}" (the
  initial carry `jnp.zeros(inputDim)` is not marked varying inside `shard_map`). Found 2026-11-05.
- `solver/pinv.py` (`_snr_step`) calls `grad_log_psi.transform(lambda x: ...)` with a new lambda
  at every step: `transform` wraps it in `jax.jit`, so it compiles again at every step. User
  (2026-11-05): fix when doing `solver/`. (Fixed in `optimizer/minsr.py` on 2026-11-05.)
- Tests: on 3 devices, 32 tests in operator/vqs/minsr/objective_function fail only because their
  batch sizes are powers of 2 ("has to be divisible by the number of devices").
- `objective_function/infidelity.py`: `_get_f_loc_sh` calls `operator._get_conn_elements(s, {})`,
  so operators with callable (time-dependent) prefactors get no keyword arguments.
- `optimizer/base.py`: the reweighting between substeps uses a hard-coded factor
  `2 * jnp.real(logPsi - logPsi0)`, i.e. assumes `logProbFactor = 0.5`; also check the weights are
  normalized.
- `util/util.py`: `measure()` passes `num_samples` to the sampler; with `ExactSampler` it ends up
  in `get_O_loc` kwargs (ignored by real operators). Checked 2026-11-06: harmless, callable
  prefactors must take `**kwargs`.
- Optimizer on 2 CPU processes (gloo): `SR.ground_state_search` with `MCSampler` aborts (SIGABRT, no
  traceback) in the first step; sampling and measuring work on 2 processes. Probably the solver's
  distributed path (`jaxmg` is Linux/GPU only). Check in the `solver/` pass. Found 2026-11-06.
- `solver/util.py` imports `USE_DISTRIBUTED` by name; fine now that it can only change at import
  (`sharding_config` sets it from `jax.distributed.is_initialized()`).
- Docs: `parallelism.rst` describes `global_defs.set_pmap_devices()` and MPI, which no longer
  exist; `mpi.rst` points at `jVMC.mpi_wrapper`; `nets.rst`, `stats.rst`, `util.rst` use `jVMC.*`.
