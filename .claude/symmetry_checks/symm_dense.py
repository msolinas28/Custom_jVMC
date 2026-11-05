"""Dense checks of the symmetry projectors against library Hamiltonians."""
import itertools
import numpy as np
import jax
import jax.numpy as jnp

import jVMC_exp.operator.discrete.branch_free as op
from jVMC_exp.symmetry import SymmetryProjector, compose_projectors
from jVMC_exp.symmetry.lattice_symetries import (
    square_translation_symmetry, square_d4_symmetry, rectangle_d2_symmetry,
    chain_reflection_symmetry, particle_hole_symmetry, spin_flip_symmetry,
)

def make_basis(n, ldim):
    return np.array(list(itertools.product(range(ldim), repeat=n)), dtype=np.int32)

def dense_P(symm, basis):
    index = {tuple(b): k for k, b in enumerate(basis)}
    states = np.asarray(jax.jit(jax.vmap(symm.transformed_states))(jnp.asarray(basis)))
    weights = np.asarray(jax.jit(jax.vmap(symm.weights))(jnp.asarray(basis)))
    D = len(basis)
    P = np.zeros((D, D), dtype=np.complex128)
    for k in range(D):
        for g in range(symm.nsymm):
            P[k, index[tuple(states[k, g])]] += weights[k, g] / symm.nsymm
    return P

def check(name, symm, basis, H):
    P = dense_P(symm, basis)
    idem = np.abs(P @ P - P).max()
    herm = np.abs(P - P.conj().T).max()
    comm = np.abs(P @ H - H @ P).max()
    ok = idem < 1e-10 and herm < 1e-10 and comm < 1e-10
    print(f"{'ok ' if ok else 'BAD'} {name:55s} |P^2-P|={idem:.1e} |P-P^+|={herm:.1e} |[P,H]|={comm:.1e} rank={np.trace(P).real:.2f}")
    return P

def dense(H, basis):
    M = np.asarray(H.to_dense(jnp.asarray(basis)))
    assert np.abs(M - M.T).max() < 1e-12 and np.abs(M.imag).max() < 1e-12, "use a real symmetric H"
    return M

def bonds(Lx, Ly):
    idx = lambda x, y: (x % Lx) + Lx * (y % Ly)
    out = []
    for y in range(Ly):
        for x in range(Lx):
            if Lx > 1: out.append((idx(x, y), idx(x + 1, y)))
            if Ly > 1: out.append((idx(x, y), idx(x, y + 1)))
    return out

def hop(i, j, spin=None):
    return op.Creation(i, spin) * op.Annihilation(j, spin) + op.Creation(j, spin) * op.Annihilation(i, spin)

if __name__ == "__main__":
    # ---------------- spins ----------------
    for Lx, Ly in [(6, 1), (3, 3)]:
        b = make_basis(Lx * Ly, 2)
        H = sum(op.SigmaZ(i) * op.SigmaZ(j) + 0.7 * op.SigmaX(i) * op.SigmaX(j) for i, j in bonds(Lx, Ly)) \
            + 0.3 * sum(op.SigmaX(i) for i in range(Lx * Ly))
        Hd = dense(H, b)
        tot = 0
        for q in itertools.product(range(Lx), range(Ly)):
            tot = tot + check(f"spin {Lx}x{Ly} translation q={q}", square_translation_symmetry(Lx, Ly, "spin", q=q), b, Hd)
        print("    sum over q = identity:", np.abs(tot - np.eye(len(b))).max() < 1e-10)
        check(f"spin {Lx}x{Ly} point group", rectangle_d2_symmetry(Lx, Ly, "spin") if Ly == 1 else square_d4_symmetry(Lx, Ly, "spin"), b, Hd)
        check(f"spin {Lx}x{Ly} spin flip", spin_flip_symmetry(Lx, Ly), b, Hd)

    # ---------------- spinless fermions ----------------
    for Lx, Ly in [(6, 1), (5, 1), (3, 3)]:
        b = make_basis(Lx * Ly, 2)
        n = lambda i: op.Number(i)
        H = -sum(hop(i, j) for i, j in bonds(Lx, Ly)) + 0.8 * sum(n(i) * n(j) for i, j in bonds(Lx, Ly))
        Hd = dense(H, b)
        tot = 0
        for q in itertools.product(range(Lx), range(Ly)):
            tot = tot + check(f"spinless {Lx}x{Ly} translation q={q}", square_translation_symmetry(Lx, Ly, "spinless_fermion", q=q), b, Hd)
        print("    sum over q = identity:", np.abs(tot - np.eye(len(b))).max() < 1e-10)
        pg = rectangle_d2_symmetry(Lx, Ly, "spinless_fermion") if Ly == 1 else square_d4_symmetry(Lx, Ly, "spinless_fermion")
        check(f"spinless {Lx}x{Ly} point group", pg, b, Hd)
        T0 = square_translation_symmetry(Lx, Ly, "spinless_fermion")
        check(f"spinless {Lx}x{Ly} translation(q=0) x point group", compose_projectors(T0, pg), b, Hd)
        check(f"spinless {Lx}x{Ly} point group x translation(q=0)", compose_projectors(pg, T0), b, Hd)
        # Particle-hole: H with interactions around half filling and pairing
        Hc = 0.8 * sum((n(i) - 0.5 * op.IdentityOperator(2, i)) * (n(j) - 0.5 * op.IdentityOperator(2, j)) for i, j in bonds(Lx, Ly)) \
            + 0.6 * sum(op.Creation(i) * op.Creation(j) + op.Annihilation(j) * op.Annihilation(i) for i, j in bonds(Lx, Ly))
        Hcd = dense(Hc, b)
        C = particle_hole_symmetry(Lx, Ly, "spinless_fermion")
        check(f"spinless {Lx}x{Ly} particle-hole (H: interaction+pairing)", C, b, Hcd)
        check(f"spinless {Lx}x{Ly} translation(q=0) x particle-hole", compose_projectors(T0, C), b, Hcd)
        check(f"spinless {Lx}x{Ly} particle-hole x translation(q=0)", compose_projectors(C, T0), b, Hcd)

    # ---------------- spinful fermions ----------------
    for Lx, Ly in [(4, 1), (2, 2)]:
        b = make_basis(Lx * Ly, 4)
        N = Lx * Ly
        nu = lambda i, s: op.Number(i, s)
        H = -sum(hop(i, j, s) for i, j in bonds(Lx, Ly) for s in ("up", "down")) \
            + 2.0 * sum(nu(i, "up") * nu(i, "down") for i in range(N))
        Hd = dense(H, b)
        tot = 0
        for q in itertools.product(range(Lx), range(Ly)):
            tot = tot + check(f"spinful {Lx}x{Ly} translation q={q}", square_translation_symmetry(Lx, Ly, "spinful_fermion", q=q), b, Hd)
        print("    sum over q = identity:", np.abs(tot - np.eye(len(b))).max() < 1e-10)
        pg = rectangle_d2_symmetry(Lx, Ly, "spinful_fermion") if Ly == 1 else square_d4_symmetry(Lx, Ly, "spinful_fermion")
        check(f"spinful {Lx}x{Ly} point group", pg, b, Hd)
        F = spin_flip_symmetry(Lx, Ly, "spinful_fermion")
        check(f"spinful {Lx}x{Ly} spin flip", F, b, Hd)
        T0 = square_translation_symmetry(Lx, Ly, "spinful_fermion")
        check(f"spinful {Lx}x{Ly} translation x spin flip", compose_projectors(T0, F), b, Hd)
        check(f"spinful {Lx}x{Ly} spin flip x point group", compose_projectors(F, pg), b, Hd)
        I = lambda i: op.IdentityOperator(4, i)
        Hc = 2.0 * sum((nu(i, "up") - 0.5 * I(i)) * (nu(i, "down") - 0.5 * I(i)) for i in range(N)) \
            + 0.7 * sum((nu(i, "up") + nu(i, "down") - I(i)) * (nu(j, "up") + nu(j, "down") - I(j)) for i, j in bonds(Lx, Ly))
        Hcd = dense(Hc, b)
        C = particle_hole_symmetry(Lx, Ly, "spinful_fermion")
        check(f"spinful {Lx}x{Ly} particle-hole (H: interactions)", C, b, Hcd)
        check(f"spinful {Lx}x{Ly} translation x particle-hole", compose_projectors(T0, C), b, Hcd)
        check(f"spinful {Lx}x{Ly} particle-hole x spin flip", compose_projectors(C, F), b, Hcd)
