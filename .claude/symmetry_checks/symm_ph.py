"""Particle-hole projector: involution property and commutation with a C-even Hamiltonian."""
import sys, itertools
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, sys.argv[1])
from symm_dense import make_basis, dense_P, bonds
import jVMC_exp.operator.discrete.branch_free as op
from jVMC_exp.symmetry import compose_projectors
from jVMC_exp.symmetry.lattice_symetries import particle_hole_symmetry, square_translation_symmetry

# to_dense convention: <b_i|c|b_j> for one spinless site, basis [0], [1]
print("c on one site:\n", np.asarray(op.Annihilation(0).to_dense(jnp.asarray([[0], [1]]))).real)

def check(name, symm, basis, H):
    P = dense_P(symm, basis)
    V = 2 * P - np.eye(len(basis))          # the particle-hole matrix itself, P = (1 + V) / 2
    print(f"{name:42s} V^2 = {'+1' if np.allclose(V @ V, np.eye(len(basis))) else ('-1' if np.allclose(V @ V, -np.eye(len(basis))) else '??')}"
          f"  |P^2-P|={np.abs(P @ P - P).max():.1e}  |[P,H]|={np.abs(P @ H - H @ P).max():.1e}")

for kind, sizes in [("spinless_fermion", [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]),
                    ("spinful_fermion", [(2, 1), (3, 1)])]:
    for Lx, Ly in sizes:
        N = Lx * Ly
        ldim = 2 if kind == "spinless_fermion" else 4
        b = make_basis(N, ldim)
        M = N if kind == "spinless_fermion" else 2 * N
        if kind == "spinless_fermion":
            n = lambda i: op.Number(i) - 0.5 * op.IdentityOperator(2, i)
            # C-even: interactions around half filling and imaginary pairing
            H = sum(0.8 * n(i) * n(j) for i, j in bonds(Lx, Ly) if i != j) + 0.3 * sum(n(i) for i in range(N)) * 0 \
                + sum(0.6j * (op.Creation(i) * op.Creation(j) - op.Annihilation(j) * op.Annihilation(i))
                      for i, j in bonds(Lx, Ly) if i != j)
        else:
            nu = lambda i, s: op.Number(i, s) - 0.5 * op.IdentityOperator(4, i)
            H = sum(2.0 * nu(i, "up") * nu(i, "down") for i in range(N)) \
                + sum(0.7 * (nu(i, "up") + nu(i, "down")) * (nu(j, "up") + nu(j, "down")) for i, j in bonds(Lx, Ly) if i != j)
        Hd = np.asarray(H.to_dense(jnp.asarray(b)))
        assert np.allclose(Hd, Hd.conj().T)
        C = particle_hole_symmetry(Lx, Ly, kind)
        check(f"{kind} L={N} (M={M}, M(M-1)/2={M * (M - 1) // 2})", C, b, Hd)
        if N > 1:
            T = square_translation_symmetry(Lx, Ly, kind)
            check(f"   translation(q=0) x particle-hole", compose_projectors(T, C), b, Hd)
