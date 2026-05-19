from functools import cached_property
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Callable

class AbstractStepper(ABC):
    @abstractmethod
    def step(self, t, f, yInitial, **kwargs):
        pass

class Euler(AbstractStepper):
    ''' 
    This class implements Euler integration
    '''

    def __init__(self, timeStep: float | Callable=1e-3):
        self._scheduler = None
        if isinstance(timeStep, Callable):
            self._scheduler = timeStep
            self.dt = timeStep(0)
        else:
            self.dt = timeStep

    def update_dt(self, step: int):
        if self._scheduler is not None:
            self.dt = self._scheduler(step)

    def step(self, t, f, yInitial, **rhsArgs):
        """ This function performs an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes the Euler integration step

        :math:`y_{n+1} = y_n+\\Delta t f(y_n,t_n,p)`

        Arguments:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as ``f(y, t, **rhsArgs, intStep=k)``, \
                    where k is an integer indicating the step number of the underlying Runge-Kutta integration scheme.
            * ``y``: Initial value of :math:`y`.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right hand side function \
                    ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """

        dy = f(yInitial, t, **rhsArgs, intStep=0)

        return yInitial + self.dt * dy, self.dt

class Heun(AbstractStepper):
    """ This class implements an adaptive second order consistent integration scheme.

    Initializer arguments:
        * ``timeStep``: Initial time step (will be adapted automatically)
    """

    def __init__(self, timeStep=1e-3):

        self.dt = timeStep

    
    def set_dt(self, timeStep):

        self.dt = timeStep


    def step(self, t, f, yInitial, **rhsArgs):
        """ This function performs an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes a second-order consistent integration step for :math:`y`.

        Arguments:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as ``f(y, t, **rhsArgs, intStep=k)``, \
                    where k is an integer indicating the step number of the underlying Runge-Kutta integration scheme.
            * ``yInitial``: Initial value of :math:`y`.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right hand side function \
                    ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """

        dt = self.dt
        if "dt" in rhsArgs:
            dt = rhsArgs["dt"]
        k0 = f(yInitial, t, **rhsArgs, intStep=0)
        y = yInitial + dt * k0
        k1 = f(y, t + dt, **rhsArgs, intStep=1)
        dy0 = 0.5 * dt * (k0 + k1)

        return yInitial + dy0, dt

class AdaptiveHeun(AbstractStepper):
    """ This class implements an adaptive second order consistent integration scheme.

    Initializer arguments:
        * ``timeStep``: Initial time step (will be adapted automatically)
        * ``tol``: Tolerance for integration errors.
        * ``maxStep``: Maximal allowed time step.
    """

    def __init__(self, timeStep=1e-3, tol=1e-8, maxStep=1):
        self.dt = timeStep
        self.tolerance = tol
        self.maxStep = maxStep

        self._butcher_tableau = (
            jnp.array([[0, 0], [1, 0]]),
            jnp.array([0.5, 0.5]),
            jnp.array([0, 1])
        )

    def step(self, t, f, y, normFunction=jnp.linalg.norm, **rhsArgs):
        """ This function performs an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes a second-order consistent integration step for :math:`y`.
        The time step :math:`\\Delta t` is chosen such that the integration error (quantified
        by a given norm) is smaller than the given tolerance.

        Arguments:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as ``f(y, t, **rhsArgs, intStep=k)``, \
                    where k is an integer indicating the step number of the underlying Runge-Kutta integration scheme.
            * ``y``: Initial value of :math:`y`.
            * ``normFunction``: Norm function to be used to quantify the magnitude of errors.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right hand side function \
                    ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """
        fe = 0.5
        dt = self.dt

        while fe < 1.:
            k0 = f(y, t, **rhsArgs, intStep=0)
            dy0 = _get_rk_step(
                self._butcher_tableau, 
                t, f, y, dt, k0, **rhsArgs
            )
            dy1 = _get_rk_step(
                self._butcher_tableau, 
                t, f, y, 0.5 * dt, k0, 1, **rhsArgs
            )
            dy1 += _get_rk_step(
                self._butcher_tableau, 
                t + 0.5 * dt, f, y + dy1, 0.5 * dt, start_step=3, **rhsArgs
            )

            # compute deviation
            updateDiff = normFunction(dy1 - dy0)
            fe = self.tolerance / updateDiff

            if 0.2 > 0.9 * fe**0.33333:
                tmp = 0.2
            else:
                tmp = 0.9 * fe**0.33333
            if tmp > 2.:
                tmp = 2.

            realDt = dt
            dt *= tmp

            if dt > self.maxStep:
                dt = self.maxStep

        self.dt = dt

        return y + dy1, realDt

def _get_rk_step(butcher_tableau, t, f, y0, dt, k0=None, start_step=0, **rhs_kwargs):
    '''
    The Butcher tableau of the explicit Runge-Kutta scheme has to be given as a tuple
    (A, b, c).

    The tableau defines the integration scheme through:

        k_i = f(
            y_n + dt * sum_j A[i,j] * k_j,
            t_n + c[i] * dt
        )

    and the final update:

        y_{n+1} = y_n + dt * sum_i b[i] * k_i

    where:
        * A : strictly lower-triangular matrix of stage coefficients
        * b : vector of coefficients for the final weighted sum
        * c : vector of time shifts for each stage

    The tableau must follow the standard explicit RK convention:
        * len(A) == len(b) == len(c)
        * c[0] == 0
        * A[0,:] == 0
        * A[i,j] = 0 for j >= i
    '''
    A, b, c = butcher_tableau

    K = [f(y0, t, **rhs_kwargs, intStep=i + start_step)] if k0 is None else [k0]
    step = b[0] * K[0]

    for i in range(1, len(b)):
        y_stage = y0 + dt * sum(A[i][j] * K[j] for j in range(i))
        t_stage = t + c[i] * dt

        K.append(f(y_stage, t_stage, **rhs_kwargs, intStep=i + start_step))
        step += b[i] * K[i]

    return dt * step