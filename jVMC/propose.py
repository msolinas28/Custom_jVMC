import jax 
import jax.numpy as jnp
from abc import ABC, abstractmethod
import jax.random as random 
from functools import partial

from jVMC.geometry import AbstractGeometry
from jVMC.util.key_gen import format_key
from jVMC.vqs import NQS
import jVMC.global_defs as global_defs

def shard_array_across_chains(arr, numChains):
    arr_atleast_1d = jnp.atleast_1d(arr)
    sharded_arr = jnp.stack([arr_atleast_1d] * (global_defs.device_count() * numChains), axis=0)
    return sharded_arr.reshape((global_defs.device_count(), numChains) + arr_atleast_1d.shape)

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
    def __init__(self, geometry: AbstractGeometry, use_custom_thermalization=False):
        if not isinstance(geometry, AbstractGeometry):
            raise ValueError("The argument 'geometry' has to be an instance of the class 'jVMC.geometry.AbstractGeometry'.")
        self._geometry = geometry
        self._use_custom_thermalization = use_custom_thermalization
        self._proposerArg = None

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
            log_prob_correction: Correction given by non-symmetric proposal moves. 
                                 Please return 0 as second arg if no correctionn is needed. 
        """
        pass

    @property
    @abstractmethod 
    def proposerArg_in_axes(self):
        """
        Return a dictionary with the same shape as self._proposerArg,
        which indicates which arguments of the proposer must be sharded across
        different devices.
        """
        pass

    @abstractmethod 
    def init_proposerArg(self, vqs, numChains):
        """
        Initialize all the arguments of the proposer with the right shape 
        for sharding across devices and chains.
        """
        pass

    @abstractmethod
    def update_proposerArg(self, vqs):
        """
        Update the arguments of the proposer that depend on the network.
        """
    
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

class RWM(AbstractProposeCont):
    def __init__(self, geometry, sigma=None, use_thermalization=True, adapt_rate=0.1, target_rate=0.5):
        super().__init__(geometry, use_thermalization)
        self._adapt_rate = adapt_rate
        self._target_rate = target_rate
        self._sigma = sigma

    @property
    def adapt_rate(self):
        return self._adapt_rate
    
    @property
    def target_rate(self):
        return self._target_rate
            
    @property
    def proposerArg_in_axes(self):
        return {"sigma": 0}
    
    def init_proposerArg(self, vqs, numChains):
        if self._sigma is None:
            sigma = jnp.array(self.geometry.extent) * 0.1
        else: 
            sigma = jnp.asarray(self._sigma) if not isinstance(self._sigma, jax.Array) else self._sigma
            if sigma.size == 1:
                sigma = jnp.full(self.geometry.n_dim, sigma.item())
            elif sigma.size != self.geometry.n_dim:
                raise ValueError(
                    f"'sigma' must have the same dimension as 'geometry.n_dim' "
                    f"(expected {self.geometry.n_dim}, got {sigma.size})."
                )
        sigma = shard_array_across_chains(jnp.asarray(sigma), numChains)

        self._proposerArg = {"sigma": sigma}
    
    def update_proposerArg(self, vqs):
        pass
    
    def __call__(self, key, s, updateProposerArg):
        """
        Proposal move for random walk metropolis (RWM).
        
        Args:
            key: An instance of ``jax.random.PRNGKey``.
            s: The configuration.

        Return: 
            s': The new proposed configuration. 
        """
        sigma = updateProposerArg["sigma"]
        dx = (sigma * jax.random.normal(format_key(key), s.shape, dtype=s.dtype))   
        return self.geometry.apply_PBC(s + dx), 0
    
    def _get_new_sigma(self, sigma, acceptance_rate):
        new_sigma = sigma * jnp.exp(self.adapt_rate * (acceptance_rate - self.target_rate))
        return jnp.clip(new_sigma, max=jnp.array(self.geometry.extent) / 2)

    def _custom_therm_fun(self, states, logAccProb, key, numProposed, numAccepted,
                          params, sweepSteps, thermSweeps, sweepFunction, updateProposerArg):
        def therm_sweep_fun(i, carry):
            states, logAccProb, key, numProposed, numAccepted, updateProposerArg = carry
            tmp_numAccepted = numAccepted

            states, logAccProb, key, numProposed, numAccepted = sweepFunction(states, logAccProb, key, numProposed, numAccepted,
                                      params, sweepSteps, self, updateProposerArg)
            acceptance_rate = (numAccepted - tmp_numAccepted) / jnp.maximum(sweepSteps, 1)
            # Vectorize across chains
            new_sigma = jax.vmap(self._get_new_sigma)(updateProposerArg["sigma"], acceptance_rate)
            updateProposerArg = {**updateProposerArg, "sigma": new_sigma}

            return (states, logAccProb, key, numProposed, numAccepted, updateProposerArg)

        carry = (states, logAccProb, key, numProposed, numAccepted, updateProposerArg)
        return jax.lax.fori_loop(0, thermSweeps, therm_sweep_fun, carry)
    
class MALA(AbstractProposeCont):
    def __init__(self, geometry, appy_fun, tau=None, use_thermalization=True, adapt_rate=0.1, target_rate=0.5, mu=2):
        super().__init__(geometry, use_thermalization)
        self._adapt_rate = adapt_rate
        self._target_rate = target_rate
        self._apply_fun = appy_fun
        self._mu = mu
        self._tau = tau
        
    @property
    def apply_fun(self):
        return self._apply_fun

    @property
    def mu(self):
        return self._mu

    @property
    def adapt_rate(self):
        return self._adapt_rate
    
    @property
    def target_rate(self):
        return self._target_rate

    @property
    def proposerArg_in_axes(self):
        return {"params": None, "tau": 0}
    
    def init_proposerArg(self, vqs, numChains):
        # TODO: implement version for tau in more dims
        if self._tau is None:
            tau = jnp.array(self.geometry.extent)[0] * 0.01
        elif self._tau < 0:
            raise ValueError('"tau" can not be negative.')
        else:
            tau = self._tau
        tau = shard_array_across_chains(jnp.asarray(tau), numChains)

        self._proposerArg = {"tau": tau, "params": vqs.parameters}
    
    def update_proposerArg(self, vqs):
        if self._proposerArg is None:
            raise RuntimeError("Must call init_proposerArg before update_proposerArg.")
        self._proposerArg["params"] = vqs.parameters
    
    def __call__(self, key, s, updateProposerArg):
        tau = updateProposerArg["tau"]
        params = updateProposerArg["params"]
        log_prob_fun = partial(lambda x, p: self.mu * jnp.real(self.apply_fun(p, x)), p=params)
        log_prob_fun_grad = jax.grad(log_prob_fun)

        xi = jax.random.normal(format_key(key), s.shape, dtype=s.dtype)
        drift = tau * log_prob_fun_grad(s)
        dx = drift + jnp.sqrt(2 * tau).astype(s.dtype) * xi
        sp = s + dx

        log_q_s_sp = - jnp.sum((s - sp - tau * log_prob_fun_grad(sp)) ** 2) / (4 * tau)
        log_q_sp_s = - jnp.sum(xi ** 2) * 0.5
        log_prob_correction = log_q_s_sp - log_q_sp_s

        return self.geometry.apply_PBC(sp), log_prob_correction.squeeze()
    
    def _get_new_tau(self, tau, acceptance_rate):
        new_tau = tau * jnp.exp(self.adapt_rate * (acceptance_rate - self.target_rate))
        return jnp.clip(new_tau, max=jnp.array(self.geometry.extent)[0] / 2) # TODO: change when tau gets more dims

    def _custom_therm_fun(self, states, logAccProb, key, numProposed, numAccepted,
                          params, sweepSteps, thermSweeps, sweepFunction, updateProposerArg):
        def therm_sweep_fun(i, carry):
            states, logAccProb, key, numProposed, numAccepted, updateProposerArg = carry
            tmp_numAccepted = numAccepted

            states, logAccProb, key, numProposed, numAccepted = sweepFunction(states, logAccProb, key, numProposed, numAccepted,
                                      params, sweepSteps, self, updateProposerArg)
            acceptance_rate = (numAccepted - tmp_numAccepted) / jnp.maximum(sweepSteps, 1)
            # Vectorize across chains
            new_tau = jax.vmap(self._get_new_tau)(updateProposerArg["tau"], acceptance_rate)
            updateProposerArg = {**updateProposerArg, "tau": new_tau}

            return (states, logAccProb, key, numProposed, numAccepted, updateProposerArg)

        carry = (states, logAccProb, key, numProposed, numAccepted, updateProposerArg)
        return jax.lax.fori_loop(0, thermSweeps, therm_sweep_fun, carry)