# Discrete operators: `get_O_loc` measures the transpose of the operator

Findings from 2026-11-03, against commit `d52b4e9` plus the uncommitted operator-folder changes of
that day (documentation, `SigmaPlus`/`SigmaMinus` swap, `return NotImplemented`). Status: **not
fixed yet**, kept for later because it changes results. A tested patch is at the end.

## The bug in one sentence

For every discrete (branch-free) operator, `get_O_loc` computes the local estimator of `Mᵀ`,
where `M` is the matrix returned by `to_sparse` / `to_dense`, and the tests compare against `M`
as the true operator.

## Evidence

- On a 3-site chain with a random complex `CpxRBM`, for 8 operators, `get_O_loc(s, psi)`
  matches `(Mᵀψ)(s)/ψ(s)` to 1e-16 everywhere, on 1 and 3 devices. It matches `(Mψ)(s)/ψ(s)`
  only when `M` is real symmetric.

  | Operator | `max|O_loc − (Mψ)/ψ|` today |
  |---|---|
  | `SigmaX`, `SigmaZ`, real Hubbard, diagonal operators | < 1e-15 (correct) |
  | `SigmaY(0)` | 4.0 |
  | `SigmaX(0)*SigmaY(1)` | 3.7 |
  | Ising + σy field | 1.9 |
  | `SigmaPlus(0)`, `Creation(0)` | 2.0 |
  | `c†_0 c_2` | 1.1 |

- One spin in `(|0⟩ + i|1⟩)/√2`, an eigenstate of the standard σy with eigenvalue +1:
  `to_array` + `to_sparse` give ⟨σy⟩ = +1, `ExactSampler(psi)(op.SigmaY(0))` gives −1.
- A Hermitian hopping with complex amplitude, `(0.3+0.4i) c†_0 c_2 + h.c.`, has its expectation
  value with the wrong sign: sampler +0.0508, exact −0.0508.
- `to_sparse` itself is right: `tests/to_sparse_to_array_test.py` checks it against a Hamiltonian
  built from the standard Pauli matrices, and the fermion tests check it against hand-built
  references.

## Cause

The elementary operators define a ket action: `O|s⟩ = mat_els[s] · |map[s]⟩`, i.e.
`M[map(s), s] = mat_els[s]`. `to_sparse` uses them that way (row = connected configuration,
column = input).

`get_O_loc` needs rows: `O_loc(s) = Σ_s' ⟨s|O|s'⟩ ψ(s')/ψ(s)`. It instead computes
`Σ mat_el(s→s') ψ(s')/ψ(s) = Σ_s' M[s', s] ψ(s')/ψ(s)`, which is the local estimator of `Mᵀ`
(`jVMC_exp/operator/discrete/base.py`, `_get_O_loc`, last line). The order of the factors in a
product is consistent with `M` (the engine applies the last factor first), so the whole operator
comes out transposed, not just each factor.

`Infidelity._get_f_loc_sh` (`jVMC_exp/objective_function/infidelity.py`) uses
`operator._get_conn_elements(s, {})` in the same way, so it also uses `Mᵀ`.

## Consequences

Worked out from the algebra; the per-sample errors above were measured.

- Real symmetric operators (Ising, Heisenberg, real Hubbard): no effect.
- Ground-state searches that only use the sampler come out right even for complex Hermitian `H`:
  minimizing ⟨Hᵀ⟩ = ⟨H̄⟩ gives the complex conjugate of the true ground state, and measuring
  any observable with `Oᵀ` on that state returns ⟨O⟩ of the true state.
- Real-time evolution with a complex `H` is wrong (it evolves with `H̄`). Example: a spin in
  `|0⟩` with `H = σy` gets ⟨σx(t)⟩ = −sin 2t instead of +sin 2t.
- Anything that mixes the sampler with `to_sparse` / `to_array`, an exact state, or a prescribed
  complex initial state disagrees.
- Non-Hermitian correlators come out transposed: ⟨c†_i c_j⟩ → ⟨c†_j c_i⟩, ⟨σ⁺_i σ⁻_j⟩ → ⟨σ⁻_i σ⁺_j⟩.

## The original jVMC had the same engine

Its `_get_s_primes` + `get_O_loc` compute `Σ matEl·ψ(map(s))/ψ(s)` exactly like this library.
Running the original definitions (`Sz = [-1, 1]`, so state 0 = down; `Sy` matEls `[1j, -1j]`;
`Sp` map `[1, 0]` matEls `[1, 0]`; `Sm` map `[0, 0]` matEls `[0, 1]`; `creation` /
`annihilation` as in the original) through the current engine:

| Original operator | What its sampler measured |
|---|---|
| `Sx`, `Sz` | correct (symmetric) |
| `Sy` | correct: it was written in the transposed form |
| `Sp`, `Sm` | the transpose: ⟨Sp⟩ came out as ⟨σ⁻⟩ and vice versa |
| `creation`, `annihilation` | the transpose: ⟨c†⟩ came out as ⟨c⟩ |

Symmetric combinations (`Sp Sm + Sm Sp`, `c†c + h.c.`) were unaffected.

In this library `SigmaZ` uses state 0 = up and `SigmaY` was changed to match `to_sparse`. After
the `SigmaPlus` / `SigmaMinus` swap of 2026-11-03, **every** elementary operator gives the
standard matrix in `to_sparse` and the sampler measures its transpose. The convention is now
consistent, so fixing the engine once makes everything right.

**Side effect of that swap, until the engine is fixed:** the sampler measures ⟨σ⁻⟩ for
`SigmaPlus` (before the swap the two errors cancelled for this operator).

## Why `Infidelity` works today

`util.util.matrix_to_jvmc_operator` expands a dense matrix in a local basis I, σx, σy, σz and maps
the basis elements to `IdentityOperator`, `SigmaX`, `SigmaY`, `SigmaZ`. Its σy element is
`[[0, 1j], [-1j, 0]]`, which is σyᵀ, i.e. exactly what the engine measures for `SigmaY`. I, σx, σz
are symmetric and tensor products transpose factor by factor, so for any `U` built by the helper
the sampler measures `U` (checked: `to_sparse(U_op) == Uᵀ`, sampler measures `U`). The two
transposes cancel.

The infidelity tests and `examples/ex8` build `U` and `U†` through this helper and compare with
dense `U`, so they test the pair helper + engine and pass. No test builds a complex or
non-symmetric operator directly from `op.*` and compares `get_O_loc` with `to_sparse`.

The σy entry was added in commit `54e2988` (2026-04-16), together with the helper, and never
changed. `Infidelity` with an operator built by hand from `op.*` (e.g. `1 − iδt·H` with a σy field
or ladder operators) uses `Uᵀ` today.

## Options compared (prototyped on scratch copies)

| | A: conjugate matrix elements in `get_O_loc` | B: transpose the operator strings for `get_O_loc` |
|---|---|---|
| Hermitian operators (incl. complex hopping, σy) | correct | correct |
| Non-Hermitian operators | estimator of `M†`: means become ⟨O⟩* (⟨c†_0 c_2⟩ = −0.0636i instead of +0.0636i; `X0*Y0 = iZ0` becomes −iZ0) | correct |
| `Infidelity` with helper-built `U`, `U†` | breaks: sampler measures `conj(U)`; 10 infidelity tests fail | works; helper's `to_sparse` becomes `U` |
| operator, infidelity, to_sparse, tdvp, objective_function tests | 12 failed | 61 passed |

Option A also needs the docs to say that the operator passed to `get_O_loc` must be Hermitian
(the elementary `SigmaPlus`, `Creation`, `Annihilation` are not Hermitian themselves).
Applying A only to `get_O_loc` and leaving `Infidelity` alone would keep `Infidelity` working but
leave two conventions in the library.

**Recommendation: option B.** It is exact for every operator and needs no Hermitian restriction.

### How option B works

- `_compile` builds a second set of arrays for the transposed strings: each string's factors in
  reverse order, each elementary operator replaced by its transpose.
- The transpose of an elementary operator is computed generically from `map` / `mat_els`
  (inverse map on the non-zero elements). It is branch-free for every predefined operator; a
  clear `ValueError` is raised otherwise.
- Fermion signs carry over unchanged: they only depend on the occupations of other sites (and,
  for spin up, of the down mode on the same site), which are the same in `s` and `s'`.
- `get_O_loc` and `Infidelity` use the transposed arrays; `to_sparse` and `get_conn_elements`
  keep the original ones.
- `matrix_to_jvmc_operator` switches its σy basis element to the standard
  `[[0, -1j], [1j, 0]]`, otherwise it would now build `Uᵀ`.

### Execution time of option B

Four alternating rounds on CPU, 4096 samples, `CpxRBM(numHidden=16)`:

| | repo | patched |
|---|---|---|
| operator compile, once per operator (2D Ising 6×6, 144 strings) | 8–10 ms | 21–23 ms |
| operator compile, Hubbard L=12 | 5 ms | 12 ms |
| first `get_O_loc`, incl. JIT | 0.57–0.65 s | 0.57–0.65 s |
| steady `get_O_loc`, Ising / Hubbard / Ising with `batchSize=64` | 85–91 / 51–55 / 323–353 ms | 88–89 / 52–62 / 320–389 ms |
| `to_sparse` | 2.2–2.4 ms | 2.3 ms |

Only the one-time compile is slower (it builds the arrays twice). The compiled kernel has the same
number and length of strings, so per-call time is unchanged within noise. `Infidelity` calls the
same function in the same way; not benchmarked separately.

## After applying the patch

- Add a test that compares `get_O_loc` with `(to_sparse(basis) @ psi) / psi` for complex and
  non-Hermitian operators (`SigmaY`, `SigmaX*SigmaY`, `SigmaPlus`, `Creation(0)*Annihilation(2)`,
  a complex hopping), with a complex random `CpxRBM`.
- Add a test that `to_sparse` of `matrix_to_jvmc_operator(U)` equals `U`.
- Re-run the notebook `examples/ex8_infidelity_optimization.ipynb`.
- Check the docstring of `matrix_to_jvmc_operator` (it says the local basis is I, σx, σy, σz,
  which becomes true).

## Patch (option B)

Applied cleanly with `git apply` to the tree of 2026-11-03.

```diff
--- a/jVMC_exp/operator/discrete/branch_free.py
+++ b/jVMC_exp/operator/discrete/branch_free.py
@@ -30,6 +30,18 @@
     
     return spin
 
+def _transpose_local(map, mat_els):
+    """Map and matrix elements of the transpose of a branch-free elementary operator."""
+    nonzero = np.flatnonzero(mat_els)
+    if len(set(map[nonzero].tolist())) != len(nonzero):
+        raise ValueError("The transpose of an elementary operator has to be branch-free.")
+    map_t = np.zeros_like(map)
+    mat_els_t = np.zeros_like(mat_els)
+    map_t[map[nonzero]] = nonzero
+    mat_els_t[map[nonzero]] = mat_els[nonzero]
+
+    return map_t, mat_els_t
+
 class OperatorString(list):
     """
     A list of operators to be applied sequentially, with an associated scale factor.
@@ -182,6 +194,13 @@
         return strings_stack[0]
 
     def _compile(self):
+        self._compile_strings(transpose=False)
+        compiled = (self.idxC, self.mapC, self.matElsC, self.fermionicC, self.spinC, self.spinfulC, self.firstC)
+        self._compile_strings(transpose=True)
+        self._transposed = (self.idxC, self.mapC, self.matElsC, self.fermionicC, self.spinC, self.spinfulC, self.firstC)
+        self.idxC, self.mapC, self.matElsC, self.fermionicC, self.spinC, self.spinfulC, self.firstC = compiled
+
+    def _compile_strings(self, transpose):
         strings = self._get_list_of_strings()
         max_length = max(len(s) for s in strings)
         max_ldim = max(self._site_ldim.values())
@@ -210,12 +229,16 @@
             
             n = len(op_string)
             for k in range(max_length):
-                k_rev = n - 1 - k
-                if k_rev >= 0:
+                # The transpose of a string applies the transposed operators in reverse order
+                k_rev = k if transpose else n - 1 - k
+                if k < n:
                     op = op_string[k_rev]
+                    op_map, op_mat_els = (
+                        _transpose_local(op.map, op.mat_els) if transpose else (op.map, op.mat_els)
+                    )
                     idx_row.append(op.idx)
-                    map_row.append(np.pad(op.map, (0, max_ldim - op.ldim)))
-                    matels_row.append(np.pad(op.mat_els, (0, max_ldim - op.ldim)))
+                    map_row.append(np.pad(op_map, (0, max_ldim - op.ldim)))
+                    matels_row.append(np.pad(op_mat_els, (0, max_ldim - op.ldim)))
                     fermionic_row.append(int(op.fermionic))
                     spin_row.append(op.spin if op.spin is not None else 0)
                     spinful_row.append(1 if op.spin is not None else 0)
@@ -264,7 +287,11 @@
         self._n_conn = sum(not d for d in diagonal)
         self._is_compiled = True
 
-    def _get_conn_elements(self, s, kwargs):
+    def _get_conn_elements(self, s, kwargs, transpose=False):
+        idxC, mapC, matElsC, fermionicC, spinC, spinfulC, firstC = (
+            self._transposed if transpose else
+            (self.idxC, self.mapC, self.matElsC, self.fermionicC, self.spinC, self.spinfulC, self.firstC)
+        )
         sampleShape = s.shape
         s = s.ravel()
         dim = s.shape[0]
@@ -339,8 +366,7 @@
             return s_p.reshape(sampleShape), matEl
         
         s_p, mat_els = jax.vmap(process_string, in_axes=(None,) + (0,) * 8)(
-            s, prefactors, self.idxC, self.mapC, self.matElsC,
-            self.fermionicC, self.spinC, self.spinfulC, self.firstC
+            s, prefactors, idxC, mapC, matElsC, fermionicC, spinC, spinfulC, firstC
         )
         
         mat_el_diag = jnp.sum(mat_els[self.diagC])
--- a/jVMC_exp/operator/discrete/base.py
+++ b/jVMC_exp/operator/discrete/base.py
@@ -296,7 +296,8 @@
         self, s, log_psi_s, *, 
         parameters, psi: NQS, chunk_size, batch_size, **kwargs
     ):
-        s_p, mat_els, mat_el_diag = self._get_conn_elements(s, kwargs)
+        # <s|O|s'> is a column of the transposed operator
+        s_p, mat_els, mat_el_diag = self._get_conn_elements(s, kwargs, transpose=True)
 
         # Purely diagonal operator: no network evaluation needed
         if s_p.shape[0] == 0:
--- a/jVMC_exp/objective_function/infidelity.py
+++ b/jVMC_exp/objective_function/infidelity.py
@@ -238,7 +238,7 @@
         self, s, log_phi_s, *,
         parameters, operator: Operator, psi: NQS, chunk_size, batch_size
     ):
-        s_p, mat_els, mat_el_diag = operator._get_conn_elements(s, {})
+        s_p, mat_els, mat_el_diag = operator._get_conn_elements(s, {}, transpose=True)
         s_p = jnp.concatenate([s[None], s_p], axis=0)
         mat_els = jnp.concatenate([mat_el_diag[None], mat_els], axis=0)
 
--- a/jVMC_exp/util/util.py
+++ b/jVMC_exp/util/util.py
@@ -203,7 +203,7 @@
     local_basis = [
         np.eye(2, dtype=global_defs.DT_OPERATORS_CPX),
         np.array([[0.0, 1.0], [1.0, 0.0]], dtype=global_defs.DT_OPERATORS_CPX),
-        np.array([[0.0, 1.0j], [-1.0j, 0.0]], dtype=global_defs.DT_OPERATORS_CPX),
+        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=global_defs.DT_OPERATORS_CPX),
         np.array([[1.0, 0.0], [0.0, -1.0]], dtype=global_defs.DT_OPERATORS_CPX),
     ]
     jvmc_basis = [
```

## How to check the fix

```python
import numpy as np, jax, jax.numpy as jnp
from itertools import product
import jVMC_exp.nets as nets, jVMC_exp.operator.discrete as op
from jVMC_exp.vqs import NQS
from jVMC_exp import global_defs

L = 3
basis = jnp.array(list(product((0, 1), repeat=L)), dtype=global_defs.DT_SAMPLES)
psi = NQS(nets.CpxRBM(numHidden=3, bias=False), L, 2**8, seed=3)
psi.parameters = 0.4 * jax.random.normal(jax.random.PRNGKey(1), psi.parameters_flat.shape)
amplitudes = np.exp(np.asarray(psi(basis)))

for o in (op.SigmaY(0), op.SigmaX(0) * op.SigmaY(1), op.SigmaPlus(0), op.Creation(0) * op.Annihilation(2)):
    M = np.asarray(o.to_sparse(basis).toarray())
    print(np.allclose(np.asarray(o.get_O_loc(basis, psi)), (M @ amplitudes) / amplitudes))
```

All four print `False` today and `True` with the patch.
