import copy
import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from jVMC_exp.objective_function import Infidelity, Observable
from jVMC_exp.operator import continuous, discrete
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.sampler.base import AbstractSampler
from jVMC_exp.stepper import AbstractStepper, Euler
from jVMC_exp.optimizer.base import AbstractOptimizer
from jVMC_exp.optimizer.minsr import MinSR
from jVMC_exp.util.util import ObservableEntry, measure
from jVMC_exp.vqs import NQS


class Stage(NamedTuple):
    """
    One factor :math:`\\hat O_k` of a propagator, fitted by one infidelity minimization.

    Attributes:
        * ``operator``: :math:`\\hat O_k`, ``None`` for the identity.
        * ``conjugated_operator``: :math:`\\hat O_k^\\dagger`, ``None`` for the identity.
        * ``unitary``: Whether :math:`\\hat O_k` is unitary, see ``Infidelity``.
    """

    operator: AbstractOperator | None
    conjugated_operator: AbstractOperator | None
    unitary: bool


class AbstractPropagator(ABC):
    """
    Base class of the propagators of ``PTVMC``: approximations of
    :math:`e^{-i\\hat H\\tau}` by a product of operators :math:`\\prod_k\\hat O_k`, the stages.

    Args:
        * ``time_step``: Time step :math:`\\tau`.
    """

    def __init__(self, time_step: float):
        self.time_step = time_step

    @abstractmethod
    def stages(self, generator: AbstractOperator) -> List[Stage]:
        """
        Stages of the propagator for ``generator``, in the order in which they are applied.

        Args:
            * ``generator``: The generator :math:`\\hat H`, a Hermitian operator.

        Returns:
            List of ``Stage``.
        """


def _identity_like(operator: AbstractOperator):
    """Identity acting on the same kind of configurations as ``operator``."""
    if isinstance(operator, discrete.Operator):
        site, l_dim = next(iter(operator._site_l_dim.items()))
        return discrete.IdentityOperator(l_dim, site)

    return continuous.IdentityOperator(operator.geometry)


class ProductExpansion(AbstractPropagator):
    """
    Taylor expansion of :math:`e^{-i\\hat H\\tau}` up to the order ``order``, written as a product
    of linear factors,

    .. math::
        \\sum_{n=0}^{o}\\frac{(-i\\hat H\\tau)^n}{n!} = \\prod_{k=1}^{o}(1 - ia_k\\tau\\hat H),
        \\qquad a_k = -1/z_k,

    where :math:`z_k` are the roots of :math:`\\sum_{n=0}^{o}z^n/n!`. The error of a step is of
    order :math:`(\\|\\hat H\\|\\tau)^{o+1}/(o+1)!`. The factors are not unitary.

    Args:
        * ``time_step``: Time step :math:`\\tau`.
        * ``order``: Order :math:`o` of the expansion, the number of stages.
    """

    def __init__(self, time_step: float, order: int = 2):
        super().__init__(time_step)
        if order < 1:
            raise ValueError(f"The order has to be at least 1, got {order}.")
        self.order = order

    @property
    def coefficients(self):
        """The coefficients :math:`a_k`."""
        roots = np.roots([1 / math.factorial(n) for n in range(self.order, -1, -1)])

        return -1 / roots

    def stages(self, generator: AbstractOperator) -> List[Stage]:
        """
        The factors :math:`\\hat O_k = 1 - ia_k\\tau\\hat H`, with
        :math:`\\hat O_k^\\dagger = 1 + ia_k^*\\tau\\hat H`.
        """
        identity = _identity_like(generator)
        
        return [
            Stage(
                identity - complex(1j * a * self.time_step) * generator,
                identity + complex(1j * np.conj(a) * self.time_step) * generator,
                False,
            )
            for a in self.coefficients
        ]


class _ReferenceSampler(AbstractSampler):
    def set_state(self, sampler: AbstractSampler):
        self.psi.parameters = sampler.psi.parameters
        self._samples = sampler.samples
        self._log_psi = sampler.log_psi
        self._weights = sampler.weights

    def sample(self, num_samples=None):
        raise RuntimeError("The reference sampler of PTVMC does not draw samples.")


class PTVMC:
    """
    Real-time evolution with projected time-dependent variational Monte Carlo.

    A step approximates :math:`e^{-i\\hat H\\tau}` by the stages :math:`\\prod_k\\hat O_k` of a
    propagator, e.g. ``ProductExpansion``. In stage :math:`k` the reference state
    :math:`|\\chi\\rangle` is the variational state at the start of the stage, and ``optimizer``
    minimizes the ``Infidelity`` of :math:`|\\psi_\\theta\\rangle` with
    :math:`\\hat O_k|\\chi\\rangle` with ``fit_steps`` steps of ``fit_stepper``.

    Since :math:`\\chi=\\psi_\\theta` at the start of a stage, the samples of :math:`\\chi` are
    those of the sampler of ``optimizer`` at that moment: no further sampling is needed. The
    reference state is a copy of ``optimizer.psi`` that keeps the parameters of the start of the
    stage.

    The operators of the generator receive the time :math:`t` of the start of the step as the
    keyword argument ``t``.

    The data are recorded in ``optimizer.output_manager``, as in ``AbstractOptimizer``: the mean,
    the variance and the statistical error of the local energy in ``"energy"``, and the
    requested observables. With ``save_meta_data``, every step also records

    * ``dt``: The time step,
    * ``infidelity``: The ``mean``, the ``variance`` and the ``MC_error`` of the infidelity of
      every stage at the end of its fit, one entry per stage,
    * ``total_infidelity``: The same for :math:`\\sum_k\\mathcal I_k`, the infidelity of the step
      to leading order when the errors of the stages are uncorrelated.

    Args:
        * ``optimizer``: Optimizer of the fits, e.g. ``jVMC_exp.optimizer.SR``, with the sampler
          and the variational state to evolve.
        * ``fit_stepper``: Stepper of the fits, with an ``update_dt(step)`` method, e.g.
          ``jVMC_exp.stepper.Euler(learning_rate)``. Its schedule restarts at every stage.
        * ``fit_steps``: Number of steps of every fit.
        * ``control_variate``, ``moving_average_width``: See ``Infidelity``.
        * ``fit_callback`` [optional]: Callable, or list of callables, called after every step
          of a fit with the keyword arguments ``driver`` (this object), ``loss`` (the
          ``Infidelity`` of the stage), ``step`` (the index of the step in the fit), ``t`` (the
          time) and ``previous_parameters`` (the flat parameters before the step). At that
          moment ``psi`` holds the new parameters, and the sampler the samples of the step,
          drawn at ``previous_parameters``. E.g. ``LevenbergMarquardt()``.
        * ``show_fit_progress``: Whether to show a progress bar for every fit, which is removed
          when the fit ends.
    """

    def __init__(
        self,
        optimizer: AbstractOptimizer,
        fit_stepper: AbstractStepper,
        fit_steps: int,
        control_variate: Literal["simple", "adaptive"] | None = None,
        moving_average_width: int = 1,
        fit_callback: Callable | List[Callable] | None = None,
        show_fit_progress: bool = True,
    ):
        if not hasattr(fit_stepper, "update_dt"):
            raise ValueError(
                "The stepper of the fits must implement a method called 'update_dt'."
            )

        self._optimizer = optimizer
        self.fit_stepper = fit_stepper
        self.fit_steps = fit_steps
        self._control_variate = control_variate
        self._moving_average_width = moving_average_width
        if fit_callback is None:
            fit_callback = []
        elif callable(fit_callback):
            fit_callback = [fit_callback]
        self.fit_callback = list(fit_callback)
        self.show_fit_progress = show_fit_progress
        self.meta_data = {}

        # The reference state shares the network and the compiled functions of psi, which take
        # the parameters as an argument, but keeps its own parameters
        chi = copy.copy(optimizer.psi)
        chi._sharded_cache = {}
        self._reference_sampler = _ReferenceSampler(chi)

        self._losses = None
        self._losses_key = None

    @property
    def optimizer(self):
        """Optimizer of the fits."""
        return self._optimizer

    @property
    def sampler(self):
        """Sampler of the variational state."""
        return self.optimizer.sampler

    @property
    def psi(self) -> NQS:
        """Variational state."""
        return self.optimizer.psi

    @property
    def output_manager(self):
        """``OutputManager`` of the optimizer, which records the data."""
        return self.optimizer.output_manager

    def step(
        self,
        t,
        propagator: AbstractPropagator,
        objective_function: Observable,
        **kwargs,
    ):
        """
        One step of the propagator, which changes ``psi.parameters``.

        The sampler must hold samples of the current state, e.g. after ``sampler.sample()``:
        they are the samples of the reference state of the first stage. Afterwards it holds
        samples of the new state. The infidelities of the stages are stored in ``meta_data``.

        Args:
            * ``t``: Time.
            * ``propagator``: ``AbstractPropagator``.
            * ``objective_function``: ``Observable`` of the generator.
            * ``**kwargs``: Keyword arguments of the fit stepper and of the infidelity.

        Returns:
            The time step.
        """
        means, variances, errors = [], [], []
        losses = self._get_losses(propagator, objective_function)
        for k, loss in enumerate(losses):
            self._reference_sampler.set_state(self.sampler)

            pbar = tqdm.trange(
                self.fit_steps,
                desc=f"fit {k + 1}/{len(losses)}",
                leave=False,
                position=1,
                disable=not self.show_fit_progress or jax.process_index() != 0,
            )
            for n in pbar:
                self.fit_stepper.update_dt(n)
                self.optimizer.update_hyperparams(n)
                previous_parameters = self.psi.parameters_flat
                self.psi.parameters, _ = self.optimizer.step(
                    t, self.fit_stepper, loss, sample_ref_state=False, **kwargs
                )
                if not pbar.disable:
                    # The loss at the parameters before the step
                    loss_value = jnp.real(self.optimizer.o_loc.mean).squeeze()
                    pbar.set_postfix(I=f"{float(loss_value):.2e}")

                for callback in self.fit_callback:
                    callback(
                        driver=self,
                        loss=loss,
                        step=n,
                        t=t,
                        previous_parameters=previous_parameters,
                    )
            pbar.close()

            # These samples also serve the reference state of the next stage
            self.sampler.sample()
            infidelity = loss(self.sampler, sample_ref_state=False, t=t)
            means.append(jnp.real(infidelity.mean).squeeze())
            variances.append(jnp.real(infidelity.var).squeeze())
            errors.append(jnp.real(infidelity.error_of_mean).squeeze())

        means, variances, errors = (jnp.stack(x) for x in (means, variances, errors))
        # The stages are estimated from independent samples: their variances add
        self.meta_data = dict(
            dt=propagator.time_step,
            infidelity=dict(mean=means, variance=variances, MC_error=errors),
            total_infidelity=dict(
                mean=jnp.sum(means),
                variance=jnp.sum(variances),
                MC_error=jnp.sqrt(jnp.sum(errors**2)),
            ),
        )

        return propagator.time_step

    def time_evolution(
        self,
        t_max,
        objective_function: Observable,
        propagator: AbstractPropagator,
        observables: Dict[str, ObservableEntry] | None = None,
        callback: List[Callable] | None = None,
        save_meta_data: bool = False,
        **kwargs,
    ):
        """
        Evolve the state up to the time ``t_max`` with steps of ``propagator``.

        The observables are measured at the start of every step and at the final time, which
        is the first multiple of the time step that reaches ``t_max``.

        Args:
            * ``t_max``: Final time.
            * ``objective_function``: ``Observable`` of the generator.
            * ``propagator``: ``AbstractPropagator``, e.g. ``ProductExpansion(tau, order)``.
            * ``observables``, ``callback``, ``save_meta_data``: As in
              ``AbstractOptimizer.ground_state_search()``.
            * ``**kwargs``: Keyword arguments of ``step()``.

        Returns:
            As ``AbstractOptimizer.time_evolution()``.
        """
        if not isinstance(objective_function, Observable):
            raise TypeError(
                "The objective function of PTVMC has to be the Observable of the generator, "
                f"got {type(objective_function).__name__}."
            )

        dt = propagator.time_step
        num_steps = math.ceil(t_max / dt - 1e-9)

        self.sampler.sample()
        pbar = tqdm.trange(num_steps, position=0, disable=jax.process_index() != 0)
        for n in pbar:
            t = n * dt
            energy = self._measure_and_store(t, objective_function, observables, callback)
            self.step(t, propagator, objective_function, **kwargs)
            if save_meta_data:
                self.output_manager.write_metadata(t, **self.meta_data)

            pbar.set_postfix(
                {
                    "t": f"{t + dt:.4f}/{t_max}",
                    "E": f"{energy:.6f}",
                    "I": f"{float(self.meta_data['total_infidelity']['mean']):.2e}",
                }
            )

        self._measure_and_store(num_steps * dt, objective_function, observables, callback)
        pbar.close()
        self.output_manager.print_total_timings()

        if save_meta_data:
            return (
                self.output_manager.data["observables"],
                self.output_manager.data["metadata"],
            )

        return self.output_manager.data["observables"]

    def _get_losses(
            self,
            propagator: AbstractPropagator,
            objective_function: Observable
        ):
        """The infidelities of the stages, rebuilt when the propagator or the time step change."""
        key = (propagator, propagator.time_step, objective_function)
        if key != self._losses_key:
            self._losses = [
                Infidelity(
                    self._reference_sampler,
                    stage.operator,
                    stage.conjugated_operator,
                    unitary=stage.unitary,
                    control_variate=self._control_variate,
                    moving_average_width=self._moving_average_width,
                    batched_jacobian=objective_function.batched_jacobian,
                )
                for stage in propagator.stages(objective_function.operator)
            ]
            self._losses_key = key

        return self._losses

    def _measure_and_store(self, t, objective_function, observables, callback):
        """Measure on the current samples; returns the mean energy."""
        o_loc = objective_function(self.sampler, t=t)
        energy = dict(
            mean=jnp.real(o_loc.mean).squeeze(),
            variance=jnp.real(o_loc.var).squeeze(),
            MC_error=jnp.real(o_loc.error_of_mean).squeeze(),
        )
        measures = measure(observables, self.sampler) if observables is not None else {}
        self.output_manager.write_observables(t, energy=energy, **measures)

        if callback is not None:
            if callable(callback):
                callback = (callback,)
            for cb in callback:
                cb()

        return float(energy["mean"])


class LevenbergMarquardt:
    """
    Fit callback of ``PTVMC`` that adapts the diagonal shift :math:`\\lambda` (``diag_shift``) of
    an ``SR`` or ``MinSR`` optimizer, with the controller of Gravina, Savona and Vicentini,
    Quantum 9, 1803 (2025), Sec. 3.3.1.

    After every step :math:`\\delta\\theta` of a fit, the gain ratio (Eq. 34)

    .. math::
        \\rho_k = \\frac{L(\\theta_k+\\delta\\theta_k)-L(\\theta_k)}{M(\\delta\\theta_k)-L(\\theta_k)},
        \\qquad
        M(\\delta)-L(\\theta_k) = \\nabla L^T\\delta + \\tfrac12\\,\\delta^T S\\,\\delta

    compares the change of the loss with the one predicted by the quadratic model of Eq. 27,
    whose curvature is the quantum geometric tensor :math:`S`. :math:`\\delta^T S\\delta` is the
    variance of the derivative of :math:`\\ln\\psi` along :math:`\\delta`. For an update that
    solves :math:`(S+\\lambda)\\dot\\theta=-F` with the force :math:`F` and
    :math:`\\delta=\\eta\\dot\\theta` (``Euler(eta)``), the linear term is
    :math:`\\nabla L^T\\delta = 2F^T\\delta = -\\frac{2}{\\eta}(\\delta^T S\\delta + \\lambda\\|\\delta\\|^2)`.
    The solve has to be exact on the modes it keeps, as with a sharp cutoff; with the smooth
    cutoff of ``PinvSNR`` this holds when :math:`\\lambda` is well above the cutoff.

    Both losses are evaluated on the samples of the step: :math:`L(\\theta_k)` is the loss of
    the step, and :math:`L(\\theta_k+\\delta\\theta_k)` is computed with the samples reweighted to
    the new parameters. The shift is then multiplied by (Eqs. 36, 37)

    .. math::
        m_k = \\min\\big[m_{\\max}, \\max\\big[m_{\\min},
        E^{\\beta_1}(\\rho_k)\\,E^{-\\beta_2}(\\rho_{k-1})\\big]\\big],
        \\qquad
        E(\\rho) = \\mathrm{sgn}(\\rho^*-\\rho)\\,|\\rho^*-\\rho|^q + 1,

    which increases :math:`\\lambda` for :math:`\\rho_k<\\rho^*` and decreases it otherwise.
    :math:`E` is bounded from below by :math:`m_{\\min}`, so that its powers are defined. Steps
    with :math:`\\rho_k<\\rho_{\\min}` or :math:`\\rho_k>\\rho_{\\max}` are rejected: the parameters
    go back to :math:`\\theta_k`, and the next step of the fit starts from there with the new
    shift. :math:`\\rho_{k-1}` starts from :math:`\\rho^*` at the start of every fit, while
    :math:`\\lambda` carries over from one fit to the next. Steps whose predicted decrease is
    below ``atol``, or below ``noise_tol`` times the statistical error of the loss, leave
    :math:`\\lambda` unchanged and are accepted.

    The defaults are the parameters of Fig. 4 of the paper.

    Args:
        * ``target``: Target :math:`\\rho^*`.
        * ``rho_min``, ``rho_max``: Steps with :math:`\\rho_k` outside of this range are rejected.
        * ``q``, ``beta_1``, ``beta_2``: Parameters of the controller.
        * ``m_min``, ``m_max``: Bounds of the factor :math:`m_k`.
        * ``min_shift``, ``max_shift``: Bounds of :math:`\\lambda`.
        * ``atol``: Smallest predicted decrease of the loss for which :math:`\\rho_k` is used.
        * ``noise_tol``: The predicted decrease must also exceed ``noise_tol`` times the
          statistical error of the loss of the step. Not part of the scheme of the paper.

    Attributes:
        * ``history``: List with a dictionary for every step: ``rho``, the ``diag_shift`` of the
          step and whether it was ``accepted``.
    """

    def __init__(
        self,
        target: float = 0.35,
        rho_min: float = 0.1,
        rho_max: float = 3.0,
        q: float = 1.0,
        beta_1: float = 0.9,
        beta_2: float = 0.1,
        m_min: float = 0.05,
        m_max: float = 2.0,
        min_shift: float = 1e-12,
        max_shift: float | None = None,
        atol: float = 1e-12,
        noise_tol: float = 0.0,
    ):
        self.target = target
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.q = q
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m_min = m_min
        self.m_max = m_max
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.atol = atol
        self.noise_tol = noise_tol
        self.history = []

        self._rho_previous = target
        self._jvp_fn = None
        self._jvp_psi = None

    def __call__(self, *, driver, loss, step, t, previous_parameters, **kwargs):
        optimizer = driver.optimizer
        if not isinstance(driver.fit_stepper, Euler):
            raise TypeError("LevenbergMarquardt requires the Euler stepper for the fits.")
        if not hasattr(optimizer, "diag_shift"):
            raise TypeError(
                f"{type(optimizer).__name__} has no diagonal shift to adapt; use SR or MinSR."
            )
        if getattr(optimizer, "diag_scale", 0) > 1e-15:
            raise ValueError("LevenbergMarquardt requires diagonal_scale=0.")

        if step == 0:
            self._rho_previous = self.target

        shift = float(optimizer.diag_shift)
        # For a holomorphic network the tangent kernel of MinSR counts every complex parameter
        # twice, as Gamma and i Gamma: its shift acts on S as shift / 2
        effective_shift = (
            shift / 2 if isinstance(optimizer, MinSR) and driver.psi.holomorphic else shift
        )
        delta = driver.psi.parameters_flat - previous_parameters
        predicted = self._predicted_change(
            driver, previous_parameters, delta, effective_shift
        )

        error = float(jnp.real(optimizer.o_loc.error_of_mean).squeeze())
        if not predicted < -max(self.atol, self.noise_tol * error):
            self.history.append(dict(rho=np.nan, diag_shift=shift, accepted=True))
            return

        old_loss = float(jnp.real(optimizer.o_loc.mean).squeeze())
        new_loss = self._reweighted_loss(driver, loss, t)
        rho = (new_loss - old_loss) / predicted

        m = self._error(rho) ** self.beta_1 * self._error(self._rho_previous) ** (-self.beta_2)
        new_shift = max(self.min_shift, min(self.m_max, max(self.m_min, m)) * shift)
        if self.max_shift is not None:
            new_shift = min(self.max_shift, new_shift)
        optimizer.diag_shift = new_shift
        self._rho_previous = rho

        accepted = self.rho_min <= rho <= self.rho_max
        if not accepted:
            driver.psi.parameters = previous_parameters
        self.history.append(dict(rho=rho, diag_shift=shift, accepted=accepted))

    def _error(self, rho):
        """E(rho) of Eq. 37, bounded from below by m_min."""
        diff = self.target - rho
        return max(self.m_min, np.sign(diff) * abs(diff) ** self.q + 1)

    def _predicted_change(self, driver, previous_parameters, delta, shift):
        """M(delta) - L = 2 F^T delta + delta^T S delta / 2, see the class description."""
        sampler = driver.sampler
        d = self._directional_derivative(driver.psi, previous_parameters, delta, sampler.samples)
        w = sampler.weights
        s_norm = float(jnp.sum(w * jnp.abs(d) ** 2) - jnp.abs(jnp.sum(w * d)) ** 2)
        eta = driver.fit_stepper.dt

        return -2 / eta * (s_norm + shift * float(jnp.sum(delta**2))) + 0.5 * s_norm

    def _directional_derivative(self, psi: NQS, parameters, delta, samples):
        """Derivative of ln psi(s) at the flat ``parameters`` along the flat ``delta``."""
        if self._jvp_psi is not psi:
            apply_fun, batch_size = psi.apply_fun, psi.batch_size

            @jax.jit
            def jvp(params, tangent, s):
                single = lambda x: jax.jvp(
                    lambda p: apply_fun(p, x), (params,), (tangent,)
                )[1]
                return jax.lax.map(single, s, batch_size=batch_size)

            self._jvp_fn, self._jvp_psi = jvp, psi

        reference, treedef = jax.tree_util.tree_flatten(psi.eval_parameters)

        def as_tree(flat):
            leaves = jax.tree_util.tree_leaves(psi._param_unflatten(flat))
            return treedef.unflatten([x.astype(r.dtype) for x, r in zip(leaves, reference)])

        return self._jvp_fn(as_tree(parameters), as_tree(delta), samples)

    def _reweighted_loss(self, driver, loss, t):
        """Loss at the current parameters on the samples of the step, reweighted."""
        sampler = driver.sampler
        samples, log_psi, weights = sampler.samples, sampler.log_psi, sampler.weights
        new_log_psi = driver.psi(samples)
        log_ratio = jnp.real(new_log_psi - log_psi) / getattr(sampler, "log_prob_factor", 0.5)
        new_weights = weights * jnp.exp(log_ratio - jnp.max(log_ratio))

        sampler._log_psi, sampler._weights = new_log_psi, new_weights / jnp.sum(new_weights)
        try:
            # The coefficient of the adaptive control variate stays the one of the step
            value = loss(
                sampler,
                sample_ref_state=False,
                control_variate_coeff=loss._control_variate_coeff,
                t=t,
            )
        finally:
            sampler._log_psi, sampler._weights = log_psi, weights

        return float(jnp.real(value.mean).squeeze())
