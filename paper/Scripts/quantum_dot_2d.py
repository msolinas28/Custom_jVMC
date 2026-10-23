"""Ground state of N = 6 spin-polarized electrons in a 2D isotropic harmonic trap (omega = 1).

Same setup as examples/ex13_2dho_ground_state_search.ipynb, where the physics and the ansatz are explained:
- FCI reference in the harmonic-oscillator basis of the first E_c shells, for several E_c
- VMC with a Slater-Jastrow-backflow ansatz optimized by stochastic reconfiguration

Writes Data/quantum_dot_2d.h5:
    attrs                    system and optimization hyperparameters
    fci/E_c, fci/energy      FCI energy for each basis cutoff
    vmc/steps                optimization steps (starting at 0)
    vmc/energy, vmc/variance, vmc/energy_error
                             energy mean, variance and MC error of the mean at each step
    vmc/final_energy, vmc/final_energy_error
                             energy of the optimized state from a separate run of FINAL_SAMPLES samples
"""
from itertools import combinations
from math import comb, factorial, pi
from pathlib import Path

import h5py
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import optax
from scipy.special import jv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

import os
os.environ["JVMC_BATCH_LOOP"] = "compiled"
import jVMC_exp
import jVMC_exp.operator.continuous as op

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

# System
N = 6
n_dim = 2
omega = 1.0

# FCI
E_C_FCI = [5, 6, 7, 8, 9]

# Ansatz and optimization
E_c_det, E_c_jastrow, E_c_bf = 6, 4, 4
n_samples = 2**11
n_steps = 100
seed = 123
FINAL_SAMPLES = 2**14

# ---------------------------------------------------------------------------
# Harmonic-oscillator orbitals
# ---------------------------------------------------------------------------

def powers(x, n):
    # [1, x, ..., x^n] on the last axis
    p = [jnp.ones_like(x)]
    for _ in range(n):
        p.append(p[-1] * x)
    return jnp.stack(p, axis=-1)

class HOOrbitals:
    # 2D harmonic-oscillator orbitals of the first E_c shells, sorted by energy
    def __init__(self, E_c, omega=1.0):
        # shell s = 2n + |m| holds the orbitals m = -s, -s + 2, ..., s
        self.n, self.m = np.array(
            [
                ((s - abs(m)) // 2, m)
                for s in range(E_c)
                for m in range(-s, s + 1, 2)
            ]
        ).T
        self.omega = omega
        self.energies = omega * (2 * self.n + np.abs(self.m) + 1)
        self.num_orbitals = len(self.n)
        self._abs_m = np.abs(self.m)

        # c_nm times the coefficients of L_n^a(t) = sum_k (-1)^k binom(n + a, n - k) t^k / k!
        norm = np.sqrt(
            [
                omega * factorial(n) / (pi * factorial(n + a))
                for n, a in zip(self.n, self._abs_m)
            ]
        )
        self._coeffs = norm[:, None] * np.array([
            [
                (-1) ** k * comb(n + a, n - k) / factorial(k) if k <= n else 0.
                for k in range(self.n.max() + 1)
            ] for n, a in zip(self.n, self._abs_m)
        ])

    def __call__(self, x):
        # x: positions with shape (n_particles, 2) -> phi_a(r_i) with shape (n_particles, num_orbitals)
        t = self.omega * jnp.sum(x ** 2, axis=-1)
        # (sqrt(w) r)^|m| e^{i m theta} = z^m or conj(z)^|m|
        z = jnp.sqrt(self.omega) * (x[:, 0] + 1j * x[:, 1])
        a_max = self._abs_m.max()
        z_m = jnp.where(
            self.m >= 0,
            powers(z, a_max)[:, self._abs_m],
            powers(z.conj(), a_max)[:, self._abs_m]
        )

        return z_m * (powers(t, self.n.max()) @ self._coeffs.T) * jnp.exp(-t / 2)[:, None]

    def radial(self, r):
        # real radial parts R_a(r) = phi_a(r) e^{-i m_a theta}, with shape (num_orbitals, len(r))
        t = self.omega * r ** 2
        return np.sqrt(t) ** self._abs_m[:, None] * (self._coeffs @ t ** np.arange(self.n.max() + 1)[:, None]) * np.exp(-t / 2)

# ---------------------------------------------------------------------------
# FCI reference
# ---------------------------------------------------------------------------

def coulomb_integrals(orbs, n_quad=300, r_max=12., q_max=25.):
    x, w = np.polynomial.legendre.leggauss(n_quad)
    r, w_r = (x + 1) * r_max / 2, w * r_max / 2
    q, w_q = (x + 1) * q_max / 2, w * q_max / 2

    R = orbs.radial(r)
    rho = R[:, None, :] * R[None, :, :] * r * w_r  # (i, k, r)
    dm = np.abs(orbs.m[:, None] - orbs.m[None, :])
    G = np.zeros(dm.shape + (n_quad,))
    for mu in np.unique(dm):
        G[dm == mu] = rho[dm == mu] @ jv(mu, np.outer(q, r)).T

    V = 4 * pi ** 2 * np.einsum("ikq, jlq, q -> ijkl", G, G, w_q, optimize=True)
    m = orbs.m
    V[m[:, None, None, None] + m[None, :, None, None] != m[None, None, :, None] + m[None, None, None, :]] = 0.

    return V

def fci_ground_state_energy(orbs, n_particles, V, M=0):
    # Slater determinants with total angular momentum M as bitmasks: |D> = c_a^+ c_b^+ ... |0> with a < b < ...
    n_orb = orbs.num_orbitals
    dets = np.sort(
        [
            sum(1 << a for a in occ)
            for occ in combinations(range(n_orb), n_particles) if orbs.m[list(occ)].sum() == M
        ]
    )

    def apply(d, a, create):
        # c_a^+ (create) or c_a on the determinants d: new determinants, fermionic sign, mask of non-vanishing results
        valid = ((d >> a) & 1) == (0 if create else 1)
        sign = 1 - 2 * (np.bitwise_count(d & ((1 << a) - 1)).astype(int) % 2)
        return d ^ (1 << a), sign, valid

    occupation = (dets[:, None] >> np.arange(n_orb)) & 1
    rows, cols, vals = [np.arange(len(dets))], [np.arange(len(dets))], [occupation @ orbs.energies]

    # Interaction as sum_{i<j, k<l} (V_ijkl - V_ijlk) c_i^+ c_j^+ c_l c_k
    pairs = np.array(list(combinations(range(n_orb), 2)))
    W = (V - V.transpose(0, 1, 3, 2))[pairs[:, 0], pairs[:, 1]][:, pairs[:, 0], pairs[:, 1]]
    for (k, l), W_kl in zip(pairs, W.T):
        d, s_k, ok_k = apply(dets, k, False)
        d, s_l, ok_l = apply(d, l, False)
        idx = np.nonzero(ok_k & ok_l)[0]
        d, s = d[idx], (s_k * s_l)[idx]
        nonzero = np.abs(W_kl) > 1e-12
        for (i, j), w in zip(pairs[nonzero], W_kl[nonzero]):
            d_j, s_j, ok_j = apply(d, j, True)
            d_i, s_i, ok_i = apply(d_j, i, True)
            ok = ok_j & ok_i
            rows.append(np.searchsorted(dets, d_i[ok]))
            cols.append(idx[ok])
            vals.append(w * (s * s_j * s_i)[ok])

    H = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(dets),) * 2)
    return eigsh(H, k=1, which="SA")[0][0]

# ---------------------------------------------------------------------------
# Ansatz
# ---------------------------------------------------------------------------

complex_normal = lambda std: nn.initializers.normal(stddev=std, dtype=jnp.complex128)

class OrbitalJastrow(nn.Module):
    orbitals: HOOrbitals
    std_init: float = 1e-1

    @nn.compact
    def __call__(self, x):
        phi = self.orbitals(x)
        A = self.param(
            "A",
            complex_normal(self.std_init),
            (self.orbitals.num_orbitals,) * 2,
            jnp.complex128
        )

        return (phi @ (A + A.T) / 2 @ phi.T)[jnp.triu_indices(x.shape[0], 1)].sum()

class ElectronCusp(nn.Module):
    cusp: float = 1 / 3  # 1 / (d + 1) for same-spin electrons in d dimensions

    @nn.compact
    def __call__(self, x):
        p = self.param("p", nn.initializers.ones, (), jnp.complex128)
        r = jnp.linalg.norm(
            (x[:, None, :] - x[None, :, :])[jnp.triu_indices(x.shape[0], 1)],
            axis=-1
        )

        return -jnp.sum(self.cusp * p ** 2 / (p + r))

class OrbitalBackflow(nn.Module):
    orbitals: HOOrbitals
    std_init: float = 1e-2

    @nn.compact
    def __call__(self, x):
        n_particles = x.shape[0]
        phi = self.orbitals(x)
        phi_others = (phi.sum(axis=0) - phi) / (n_particles - 1)
        A = self.param(
            "A",
            complex_normal(self.std_init),
            (n_particles,) + (self.orbitals.num_orbitals,) * 2,
            jnp.complex128
        )

        return jnp.einsum("ia,kab,ib->ik", phi, A, phi_others)

class SlaterJastrowBackflow(nn.Module):
    n_particles: int
    orbitals: HOOrbitals
    jastrow: nn.Module
    cusp: nn.Module
    backflow: nn.Module
    std_init: float = 1e-3

    @nn.compact
    def __call__(self, x):
        x = x.reshape(self.n_particles, 2)
        C = self.param(
            "C",
            lambda key, shape, dtype: jnp.eye(*shape, dtype=dtype) + complex_normal(self.std_init)(key, shape, dtype),
            (self.orbitals.num_orbitals, self.n_particles),
            jnp.complex128
        )

        sign, log_abs_det = jnp.linalg.slogdet((self.orbitals(x) @ C) * (1 + self.backflow(x)))

        return log_abs_det + jnp.log(sign) + self.jastrow(x) + self.cusp(x)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

E_fci = {}
for E_c in E_C_FCI:
    orbs = HOOrbitals(E_c, omega)
    E_fci[E_c] = fci_ground_state_energy(orbs, N, coulomb_integrals(orbs))
    print(f"E_c = {E_c} ({orbs.num_orbitals} orbitals): E_FCI = {E_fci[E_c]:.6f}")

# Open boundaries: the extent only sets where the walkers start and the initial step of the random walk
geometry = jVMC_exp.geometry.HyperRectangle(N, n_dim, PBC=False, extent=(6., 6.))

hamiltonian = (
    op.TotalKineticOperator(geometry)
    + op.PotentialOperator(geometry, lambda x: 0.5 * omega ** 2 * jnp.sum(x ** 2))
    + op.CoulombInteraction(geometry)
)

net = SlaterJastrowBackflow(
    n_particles=N,
    orbitals=HOOrbitals(E_c_det, omega),
    jastrow=OrbitalJastrow(HOOrbitals(E_c_jastrow, omega)),
    cusp=ElectronCusp(),
    backflow=OrbitalBackflow(HOOrbitals(E_c_bf, omega)),
)
psi = jVMC_exp.vqs.NQS(net, sampleShape=n_dim * N, batchSize=n_samples, seed=seed)
proposer = jVMC_exp.propose.RWM(geometry)
sampler = jVMC_exp.sampler.MCSamplerCont(
    psi, proposer, key=seed, numChains=n_samples // 4, numSamples=n_samples,
    sweepSteps=2 * N, thermalizationSweeps=5
)
print("Number of parameters:", psi.numParameters)

lr = optax.piecewise_constant_schedule(
    5e-2, {int(0.4 * n_steps): 0.3, int(0.7 * n_steps): 0.3, int(0.9 * n_steps): 0.3}
)
stepper = jVMC_exp.stepper.Euler(lr)
loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
solver = jVMC_exp.solver.Pinv()
opt = jVMC_exp.optimizer.SR(sampler, psi, diagonalShift=1e-4, solver=solver)

out = opt.ground_state_search(n_steps, loss_function, stepper)

E = sampler(hamiltonian, num_samples=FINAL_SAMPLES)
E_vmc, E_err = E.mean.real.item(), E.error_of_mean.real.item()
print(f"E_VMC = {E_vmc:.4f} +- {E_err:.4f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

DATA_DIR.mkdir(exist_ok=True)
with h5py.File(DATA_DIR / "quantum_dot_2d.h5", "w") as f:
    f.attrs.update(
        N=N, n_dim=n_dim, omega=omega,
        E_c_det=E_c_det, E_c_jastrow=E_c_jastrow, E_c_bf=E_c_bf,
        n_samples=n_samples, n_steps=n_steps, seed=seed, final_samples=FINAL_SAMPLES,
        num_parameters=psi.numParameters,
    )
    f["fci/E_c"] = np.array(sorted(E_fci))
    f["fci/energy"] = np.array([E_fci[E_c] for E_c in sorted(E_fci)])
    f["vmc/steps"] = np.array(out["times"])
    f["vmc/energy"] = np.array(out["energy"]["mean"])
    f["vmc/variance"] = np.array(out["energy"]["variance"])
    f["vmc/energy_error"] = np.array(out["energy"]["MC_error"])
    f["vmc/final_energy"] = E_vmc
    f["vmc/final_energy_error"] = E_err
