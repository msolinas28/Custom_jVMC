import math
import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import jVMC_exp
import jVMC_exp.operator.continuous as cop
import jVMC_exp.operator.discrete as op
from jVMC_exp.objective_function import Observable
from jVMC_exp.sampler import ExactSampler
from jVMC_exp.util import LevenbergMarquardt, PTVMC, ProductExpansion
from jVMC_exp.vqs import NQS

L = 4
TAU = 0.05


class ProductState(nn.Module):
    """One complex amplitude per site and local state: its tangent space is well conditioned"""

    @nn.compact
    def __call__(self, s):
        a = self.param(
            "a", lambda key: 0.3 * jax.random.normal(key, (L, 2), dtype=jnp.complex128)
        )
        return jnp.sum(a[jnp.arange(L), s.ravel().astype(int)])


def ising_hamiltonian():
    hamiltonian = 0
    for l in range(L):
        hamiltonian += (
            -1 * op.SigmaZ(l) * op.SigmaZ((l + 1) % L)
            - 0.7 * op.SigmaX(l)
            + 0.1 * op.SigmaZ(l)
        )
    return hamiltonian


def taylor(matrix, order):
    return sum(
        np.linalg.matrix_power(-1j * TAU * matrix, n) / math.factorial(n)
        for n in range(order + 1)
    )


class TestProductExpansion(unittest.TestCase):
    def test_discrete_product_is_taylor_polynomial(self):
        hamiltonian = ising_hamiltonian()
        basis = ExactSampler(NQS(jVMC_exp.nets.CpxRBM(2, True), L, batch_size=2**L)).basis
        H = np.asarray(hamiltonian.to_dense(basis))

        for order in (1, 2, 4):
            product = np.eye(2**L, dtype=complex)
            for stage in ProductExpansion(TAU, order).stages(hamiltonian):
                O = np.asarray(stage.operator.to_dense(basis))
                O_dag = np.asarray(stage.conjugated_operator.to_dense(basis))
                with self.subTest(order=order):
                    self.assertFalse(stage.unitary)
                    self.assertTrue(np.allclose(O_dag, O.conj().T))
                product = O @ product

            with self.subTest(order=order):
                self.assertTrue(np.allclose(product, taylor(H, order), atol=1e-12))

    def test_continuous_stages(self):
        geometry = jVMC_exp.geometry.HyperRectangle(1, 1, PBC=False, extent=(6.0,))
        hamiltonian = cop.TotalKineticOperator(geometry) + cop.PotentialOperator(
            geometry, lambda x: 0.5 * jnp.sum(x**2)
        )
        psi = NQS(jVMC_exp.nets.CpxRBM(2, True), 1, batch_size=8, seed=1)
        x = jnp.linspace(-2.0, 2.0, 8)[:, None]
        H_loc = np.asarray(hamiltonian.get_O_loc(x, psi))

        propagator = ProductExpansion(TAU, 3)
        for a, stage in zip(propagator.coefficients, propagator.stages(hamiltonian)):
            self.assertTrue(
                np.allclose(stage.operator.get_O_loc(x, psi), 1 - 1j * a * TAU * H_loc)
            )
            self.assertTrue(
                np.allclose(
                    stage.conjugated_operator.get_O_loc(x, psi),
                    1 + 1j * np.conj(a) * TAU * H_loc,
                )
            )


class TestPTVMC(unittest.TestCase):
    """
    With exact sampling, the evolved state must follow the normalized state vector propagated
    with the same product expansion.
    """

    def test_follows_propagated_state(self):
        order, t_max = 2, 0.25
        hamiltonian = ising_hamiltonian()
        psi = NQS(jVMC_exp.nets.CpxRBM(8, True), L, batch_size=2**L, seed=123)
        sampler = ExactSampler(psi)
        basis = sampler.basis
        T = taylor(np.asarray(hamiltonian.to_dense(basis)), order)

        states = []
        optimizer = jVMC_exp.optimizer.SR(sampler, psi, diagonal_shift=1e-4)
        driver = PTVMC(optimizer, jVMC_exp.stepper.Euler(0.5), fit_steps=20)
        observables, meta_data = driver.time_evolution(
            t_max,
            Observable(hamiltonian),
            ProductExpansion(TAU, order),
            callback=lambda: states.append(np.asarray(psi.to_array(basis))),
            save_meta_data=True,
        )

        num_steps = round(t_max / TAU)
        self.assertTrue(np.allclose(observables["times"], TAU * np.arange(num_steps + 1)))
        self.assertEqual(len(meta_data["times"]), num_steps)
        self.assertEqual(np.shape(meta_data["infidelity"]["mean"]), (num_steps, order))

        v = states[0]
        for state in states[1:]:
            v = T @ v
            v /= np.linalg.norm(v)
            self.assertLess(1 - np.abs(np.vdot(v, state)) ** 2, 1e-6)

    def test_effective_tdvp_error(self):
        # For one stage, converged fits and a small parameter change, the infidelity of the step
        # is tau^2 r^2 Var(H), with the TDVP error r^2
        hamiltonian, tau = ising_hamiltonian(), 0.01

        def new_state():
            psi = NQS(ProductState(), L, batch_size=2**L, seed=1)
            return psi, ExactSampler(psi)

        psi, sampler = new_state()
        tdvp = jVMC_exp.optimizer.TDVP(
            sampler, psi, make_real=True, solver=jVMC_exp.solver.Pinv()
        )
        tdvp(psi.parameters_flat, 0, int_step=0, objective_function=Observable(hamiltonian))

        psi, sampler = new_state()
        optimizer = jVMC_exp.optimizer.SR(
            sampler, psi, diagonal_shift=0.0, solver=jVMC_exp.solver.Pinv()
        )
        driver = PTVMC(
            optimizer, jVMC_exp.stepper.Euler(0.5), fit_steps=300, show_fit_progress=False
        )
        sampler.sample()
        driver.step(0, ProductExpansion(tau, 1), Observable(hamiltonian))

        effective = driver.meta_data["effective_tdvp_error"]["mean"].item()
        self.assertAlmostEqual(effective / tdvp.meta_data["tdvp_error"], 1, delta=1e-2)

    def test_rejects_other_objective_functions(self):
        psi = NQS(jVMC_exp.nets.CpxRBM(2, True), L, batch_size=2**L)
        optimizer = jVMC_exp.optimizer.SR(ExactSampler(psi), psi)
        driver = PTVMC(optimizer, jVMC_exp.stepper.Euler(0.5), fit_steps=1)
        loss = jVMC_exp.objective_function.Infidelity(ExactSampler(psi))

        with self.assertRaises(TypeError):
            driver.time_evolution(TAU, loss, ProductExpansion(TAU))


class TestLevenbergMarquardt(unittest.TestCase):
    def setUp(self):
        self.hamiltonian = ising_hamiltonian()

    def make_driver(self, optimizer_class, shift, eta, **driver_kwargs):
        psi = NQS(jVMC_exp.nets.CpxRBM(4, True), L, batch_size=2**L, seed=3)
        optimizer = optimizer_class(ExactSampler(psi), psi, diagonal_shift=shift)
        return PTVMC(
            optimizer,
            jVMC_exp.stepper.Euler(eta),
            show_fit_progress=False,
            **driver_kwargs,
        )

    def test_linear_term_of_predicted_change(self):
        # grad(L) . delta = -2 / eta (delta^T S delta + shift |delta|^2) against finite
        # differences of the exact infidelity, along the update of one step
        shift, eta, eps = 0.1, 0.1, 1e-4
        for optimizer_class in (jVMC_exp.optimizer.SR, jVMC_exp.optimizer.MinSR):
            driver = self.make_driver(optimizer_class, shift, eta, fit_steps=1)
            psi, sampler = driver.psi, driver.sampler
            loss = driver._get_losses(ProductExpansion(TAU, 2), Observable(self.hamiltonian))[0]
            sampler.sample()
            driver._reference_sampler.set_state(sampler)

            previous = psi.parameters_flat
            psi.parameters, _ = driver.optimizer.step(
                0, driver.fit_stepper, loss, sample_ref_state=False
            )
            delta = psi.parameters_flat - previous

            lm = LevenbergMarquardt()
            d = lm._directional_derivative(psi, previous, delta, sampler.samples)
            w = sampler.weights
            s_norm = float(jnp.sum(w * jnp.abs(d) ** 2) - jnp.abs(jnp.sum(w * d)) ** 2)
            effective_shift = shift / 2 if optimizer_class is jVMC_exp.optimizer.MinSR else shift
            linear = lm._predicted_change(driver, previous, delta, effective_shift) - s_norm / 2

            losses = []
            for sign in (1, -1):
                psi.parameters = previous + sign * eps * delta
                sampler.sample()
                losses.append(jnp.real(loss(sampler, sample_ref_state=False).mean).item())
            finite_difference = (losses[0] - losses[1]) / (2 * eps)

            with self.subTest(optimizer=optimizer_class.__name__):
                self.assertAlmostEqual(linear / finite_difference, 1, places=5)

    def test_adapts_shift(self):
        lm = LevenbergMarquardt()
        driver = self.make_driver(
            jVMC_exp.optimizer.SR, 1e-3, 0.5, fit_steps=5, fit_callback=lm
        )
        driver.sampler.sample()
        driver.step(0, ProductExpansion(TAU, 2), Observable(self.hamiltonian))

        self.assertEqual(len(lm.history), 10)
        self.assertNotEqual(driver.optimizer.diag_shift, 1e-3)


if __name__ == "__main__":
    unittest.main()
