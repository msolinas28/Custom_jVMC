import jax 
import jax.numpy as jnp
from abc import ABC, abstractmethod
import jax.random as random

from jVMC.geometry import AbstractGeometry
from jVMC.util.key_gen import format_key

def propose_spin_flip(key, s, info):
    idx = random.randint(key, (1,), 0, s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + 1) % 2
    return s.at[idx].set(update)


def propose_POVM_outcome(key, s, info):
    key, subkey = random.split(key)
    idx = random.randint(subkey, (1,), 0, s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + random.randint(key, (1,), 0, 3) % 4)
    return s.at[idx].set(update)

# Changed
def propose_spin_flip_Z2(key, s, info):
    idxKey, flipKey = jax.random.split(key)
    s = propose_spin_flip(idxKey, s, info)
    # On average, do a global spin flip every 30 updates to
    # reflect Z_2 symmetry
    doFlip = random.randint(flipKey, (1,), 0, 5)[0]
    return jax.lax.cond(doFlip == 0, lambda x: 1 - x, lambda x: x, s)


def propose_spin_flip_zeroMag(key, s, info):
    # propose spin flips that stay in the zero magnetization sector

    idxKeyUp, idxKeyDown, flipKey = jax.random.split(key, num=3)

    # can't use jnp.where because then it is not jit-compilable
    # find indices based on cumsum
    bound_up = jax.random.randint(idxKeyUp, (1,), 1, s.shape[0] * s.shape[1] / 2 + 1)
    bound_down = jax.random.randint(idxKeyDown, (1,), 1, s.shape[0] * s.shape[1] / 2 + 1)

    id_up = jnp.searchsorted(jnp.cumsum(s), bound_up)
    id_down = jnp.searchsorted(jnp.cumsum(1 - s), bound_down)

    idx_up = jnp.unravel_index(id_up, s.shape)
    idx_down = jnp.unravel_index(id_down, s.shape)

    s = s.at[idx_up[0], idx_up[1]].set(0)
    s = s.at[idx_down[0], idx_down[1]].set(1)

    # On average, do a global spin flip every 30 updates to
    # reflect Z_2 symmetry
    doFlip = random.randint(flipKey, (1,), 0, 5)[0]
    return jax.lax.cond(doFlip == 0, lambda x: 1 - x, lambda x: x, s)

class AbstractProposeCont(ABC):
    def __init__(self, geometry, use_custom_thermalization=False):
        if not isinstance(geometry, AbstractGeometry):
            raise ValueError("The argument 'geometry' has to be an instance of the class 'jVMC.geometry.AbstractGeometry'.")
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

    @property
    @abstractmethod 
    def Arg(self): 
        pass

    @Arg.setter
    @abstractmethod    
    def Arg(self, **kwargs):
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

# TODO: implement sigma as a vector and update each component separately to account for different lengths
class RWM(AbstractProposeCont):
    def __init__(self, geometry: AbstractGeometry, sigma=None, use_thermalization=True, adapt_rate=5e-2, target_rate=0.5):
        super().__init__(geometry, use_thermalization)
        self.adapt_rate = adapt_rate
        self.target_rate = target_rate

        if sigma is None: 
            sigma = jnp.asarray(min(geometry.extent) * 0.1)
        elif not isinstance(sigma, jax.Array):
            sigma = jnp.asarray(sigma)
        self._arg = sigma

    @property
    def Arg(self):
        return self._arg

    @Arg.setter
    def Arg(self, value):
        self._arg = value
    
    def __call__(self, key, s, sigma):
        """
        Proposal move for random walk metropolis (RWM).
        
        Args:
            key: An instance of ``jax.random.PRNGKey``.
            s: The configuration.

        Return: 
            s': The new proposed configuration. 
        """
        dx = (sigma * jax.random.normal(format_key(key), s.shape, dtype=s.dtype))   
        return self.geometry.apply_PBC(s + dx)
    
    def _get_new_sigma(self, sigma, acceptance_rate):
        new_sigma = sigma * jnp.exp(self.adapt_rate * (acceptance_rate - self.target_rate))
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
            acceptance_rate = (numAccepted - tmp_numAccepted) / jnp.maximum(numProposed, 1)
            new_sigma = self._get_new_sigma(sigma, acceptance_rate)

            return (states, logAccProb, key, numProposed, numAccepted, new_sigma)

        carry = (states, logAccProb, key, numProposed, numAccepted, updateProposerArg)
        (states, logAccProb, key, numProposed, numAccepted, updateProposerArg) = \
            jax.lax.fori_loop(0, thermSweeps, therm_sweep_fun, carry)

        return states, logAccProb, key, numProposed, numAccepted, updateProposerArg