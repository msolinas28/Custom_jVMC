import jax 
import jax.numpy as jnp
from abc import ABC, abstractmethod
from geometry import Geometry

class AbstractPropose(ABC):
    def __init__(self, geometry, use_custom_thermalization=False):
        if not isinstance(geometry, Geometry):
            raise ValueError("The argument 'geometry' has to be an instance of the class 'jVMC.geometry.Geometry'.")
        self._geometry = geometry
        self._use_custom_thermalization = use_custom_thermalization

    @property
    def geometry(self):
        return self._geometry

    @abstractmethod
    def __call__(self, key, s, info):
        """
        Proposal move.

        Args:
            key: An instance of ``jax.random.PRNGKey``.
            s: The configuration.
            info: Additional info for the proposal.

        Returns:
            s_prime: The new proposed configuration.
        """
        pass

    @abstractmethod
    @property
    def updateProposerArg(self):
        pass

    @abstractmethod
    @updateProposerArg.setter
    def updateProposerArg(self, **kwargs):
        pass
    
    def _custom_therm_fun(self, states, logAccProb, key, numProposed, numAccepted, 
                          params, sweepSteps, thermSweeps, sweepFunction):
        """
        Optional custom thermalization function.

        Subclasses can override this method if
        `use_custom_thermalization=True`.
        """
        if self._use_custom_thermalization:
            raise NotImplementedError(
                "Custom thermalization was enabled, "
                "but '_custom_therm_fun' is not implemented."
            )
        else:
            raise RuntimeError(
                "Custom thermalization not enabled for this proposer."
            )

class ProposeRWM(AbstractPropose):
    def __init__(self, geometry, sigma, use_thermalization, adapt_rate=5e-2, target_rate=0.5):
        super().__init__(geometry, use_thermalization)
        self.sigma = sigma
        self.adapt_rate = adapt_rate
        self.target_rate = target_rate

    @property
    def updateProposerArg(self):
        return self.sigma

    @updateProposerArg.setter
    def updateProposerArg(self, sigma):
        self.sigma = sigma
    
    def __call__(self, key, s, sigma):
        """
        Proposal move for random walk metropolis (RWM).
        
        Args:
            key: An instance of ``jax.random.PRNGKey``.
            s: The configuration.

        Return: 
            s': The new proposed configuration. 
        """
        dx = (sigma * jax.random.normal(key, s.shape, dtype=s.dtype))   
        return ((s + dx) % self.geometry.modulus).astype(s.dtype)
    
    def _get_new_sigma(self, sigma, numAccepted, numProposed):
        acc_rate = numAccepted / jnp.maximum(numProposed, 1)
        new_sigma = sigma * jnp.exp(self.adapt_rate * (acc_rate - self.target_rate))
        return new_sigma # Might add clipping

    def _custom_therm_fun(self,
                          states, logAccProb, key,
                          numProposed, numAccepted,
                          params, sweepSteps, thermSweeps, sweepFunction, updateProposerArg):
        def therm_sweep_fun(i, carry):
            states, logAccProb, key, numProposed, numAccepted, sigma = carry
            tmp_numAccepted = numAccepted

            states, logAccProb, key, numProposed, numAccepted = sweepFunction(states, logAccProb, key, numProposed, numAccepted,
                                      params, sweepSteps, self, sigma)
            numAcceptedsweep = numAccepted - tmp_numAccepted
            new_sigma = self._get_new_sigma(sigma, numAcceptedsweep, sweepSteps)

            return (states, logAccProb, key, numProposed, numAccepted, new_sigma)

        carry = (states, logAccProb, key, numProposed, numAccepted, updateProposerArg)
        (states, logAccProb, key, numProposed, numAccepted, updateProposerArg) = \
            jax.lax.fori_loop(0, thermSweeps, therm_sweep_fun, carry)

        return states, logAccProb, key, numProposed, numAccepted, updateProposerArg