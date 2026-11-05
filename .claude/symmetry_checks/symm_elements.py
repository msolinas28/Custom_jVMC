"""Single-element matrices: composition order, perm_sign convention, momentum convention."""
import sys, itertools
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, sys.argv[1])
from symm_dense import make_basis
import jVMC_exp.operator.discrete.branch_free as op
from jVMC_exp.symmetry import SymmetryProjector, compose_projectors
from jVMC_exp.symmetry.lattice_symetries import (particle_hole_symmetry, square_translation_symmetry,
                                                 chain_reflection_symmetry, spin_flip_symmetry)

def element_matrices(symm, basis):
    """A_g with (A_g psi)(s) = w_g(s) / chi_g * psi(g s), i.e. without the character."""
    index = {tuple(b): k for k, b in enumerate(basis)}
    states = np.asarray(jax.vmap(symm.transformed_states)(jnp.asarray(basis)))
    signs = np.asarray(jax.vmap(symm.fermion_signs)(jnp.asarray(basis)))
    D = len(basis)
    A = np.zeros((symm.nsymm, D, D), dtype=np.complex128)
    for k in range(D):
        for g in range(symm.nsymm):
            A[g, k, index[tuple(states[k, g])]] += signs[k, g]
    return A

# 1. composed elements vs products of the factors
L = 4
b = make_basis(L, 2)
T = square_translation_symmetry(L, 1, "spinless_fermion")
C = particle_hole_symmetry(L, 1, "spinless_fermion")
TC = compose_projectors(T, C)
AT, AC, ATC = element_matrices(T, b), element_matrices(C, b), element_matrices(TC, b)
print("composed elements:", TC.nsymm)
for g in range(TC.nsymm):
    l, r = divmod(g, C.nsymm)
    print(f"  ({l},{r}) == A_T A_C: {np.allclose(ATC[g], AT[l] @ AC[r])}, == A_C A_T: {np.allclose(ATC[g], AC[r] @ AT[l])}")
print("A_T[1] A_C[1] == -A_C[1] A_T[1]:", np.allclose(AT[1] @ AC[1], -AC[1] @ AT[1]))

# 2. perm_sign convention, antiperiodic spinless chain
L = 5
b = make_basis(L, 2)
H = -sum(op.Creation(i) * op.Annihilation(i + 1) + op.Creation(i + 1) * op.Annihilation(i) for i in range(L - 1)) \
    + (op.Creation(L - 1) * op.Annihilation(0) + op.Creation(0) * op.Annihilation(L - 1)) \
    + 0.8 * sum(op.Number(i) * op.Number((i + 1) % L) for i in range(L))
Hd = np.asarray(H.to_dense(jnp.asarray(b)))
perm = [[(j - a) % L for j in range(L)] for a in range(L)]          # same rows as the built-in translation
assert np.array_equal(np.asarray(T.perm_array if L == 4 else square_translation_symmetry(L, 1, "spin").perm_array), perm)
sign_input = [[-1 if i + a >= L else 1 for i in range(L)] for a in range(L)]   # particle leaving input site i crosses
sign_output = [[-1 if j - a < 0 else 1 for j in range(L)] for a in range(L)]   # particle arriving at output site j crossed
R = chain_reflection_symmetry(L, "spinless_fermion")
for name, sign in [("input-site", sign_input), ("output-site", sign_output)]:
    Ta = SymmetryProjector(Lx=L, Ly=1, particle_type="spinless_fermion", perm=tuple(map(tuple, perm)),
                           character=(1.0,) * L, perm_sign=tuple(map(tuple, sign)), name="apbc")
    A = element_matrices(Ta, b)
    comm = max(np.abs(A[g] @ Hd - Hd @ A[g]).max() for g in range(L))
    out = [f"{name:12s}: max |[A_g, H_apbc]| = {comm:.1e}"]
    for label, comp in [("T x R", compose_projectors(Ta, R)), ("R x T", compose_projectors(R, Ta))]:
        Ac = element_matrices(comp, b)
        out.append(f"{label}: {max(np.abs(Ac[g] @ Hd - Hd @ Ac[g]).max() for g in range(comp.nsymm)):.1e}")
    print(", ".join(out))

# 3. momentum convention for spins: psi_P(s shifted by +a) = exp(i theta) psi_P(s)
L = 5
b = make_basis(L, 2)
index = {tuple(x): k for k, x in enumerate(b)}
rng = np.random.default_rng(0)
psi = rng.normal(size=len(b)) + 1j * rng.normal(size=len(b))
for q in (1, 2):
    S = square_translation_symmetry(L, 1, "spin", q=(q, 0))
    states = np.asarray(jax.vmap(S.transformed_states)(jnp.asarray(b)))
    w = np.asarray(jax.vmap(S.weights)(jnp.asarray(b)))
    psi_P = np.array([np.mean([w[k, g] * psi[index[tuple(states[k, g])]] for g in range(L)]) for k in range(len(b))])
    k0 = index[(1, 1, 0, 0, 0)]
    shifted = index[(0, 1, 1, 0, 0)]       # (s_a)_i = s_{i-a} for a = 1
    ratio = psi_P[shifted] / psi_P[k0]
    print(f"q={q}: psi_P(s shifted by +1)/psi_P(s) = exp(i*{np.angle(ratio):.4f}), "
          f"2 pi q/L = {2 * np.pi * q / L:.4f}, |ratio| = {abs(ratio):.3f}")
