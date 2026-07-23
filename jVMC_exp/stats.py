from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial, cached_property
from typing import Callable
import warnings

# from jVMC_exp.sampler import AbstractSampler
from jVMC_exp.sharding_config import (
    DEVICE_SHARDING,
    REPLICATED_SHARDING,
    DEVICE_SPEC,
    REPLICATED_SPEC,
    MESH,
    sharded,
    SizedIterable,
)

@jax.jit
def _get_mean(data, weights):
    return jnp.tensordot(weights, data, axes=(0, 0))

@jax.jit
def _get_var(norm_data):
    return jnp.sum(jnp.abs(norm_data)**2, axis=0)

@jax.jit
def _get_var_not_normed(data, weights):
    return jnp.tensordot(weights, jnp.abs(data)**2, axes=(0, 0)) - jnp.abs(jnp.tensordot(weights, data, axes=(0, 0)))**2

@jax.jit
def _get_error_of_mean(var, weights):
    return jnp.sqrt(var * jnp.sum(weights ** 2))

@jax.jit
def _center(data, mean):
    return data - mean

@jax.jit
def _normalize(data, weights, mean):
    return jnp.einsum("i, i... -> i...", jnp.sqrt(weights), data - mean)

@jax.jit
def _normalize_no_center(data, weights):
    return jnp.einsum("i, i... -> i...", jnp.sqrt(weights), data)

@jax.jit(donate_argnums=(0,))
def _normalize_no_copy(data, weights):
    mean = jnp.tensordot(weights, data, axes=(0, 0))
    return jnp.einsum("i, i... -> i...", jnp.sqrt(weights), data - mean)

@jax.jit
def _get_covar(norm_data_1, norm_data_2):
    return jnp.tensordot(jnp.conj(norm_data_1), norm_data_2, axes=(0, 0))

@jax.jit
@jax.vmap
def _get_covar_per_sample(centered_data_1, centered_data_2):
    return jnp.outer(jnp.conj(centered_data_1), centered_data_2)

@jax.jit
def _get_covar_var(centered_data_1, centered_data_2, weights):
    covar_per_sample = _get_covar_per_sample(centered_data_1, centered_data_2)
    covar = jnp.tensordot(weights, covar_per_sample, axes=(0, 0))
    covar_sqrd = jnp.tensordot(weights, jnp.abs(covar_per_sample) ** 2, axes=(0, 0))

    return covar_sqrd - jnp.abs(covar) ** 2

@jax.jit
@jax.vmap
def _outer_per_sample(data_1, data_2):
    return jnp.outer(data_1, data_2)

@jax.jit
def _get_covar_var_moments(data_1, data_2, weights):
    """
    Weighted raw (uncentered) moments of a single batch, needed to accumulate
    both the covariance and the variance of the covariance across batches
    without having to center the data first (which would require the global
    mean, i.e. a full pass over the data, before it can even start).
    """
    data_1 = data_1.reshape(data_1.shape[0], -1)
    data_2 = data_2.reshape(data_2.shape[0], -1)
    sq_1 = jnp.abs(data_1) ** 2
    sq_2 = jnp.abs(data_2) ** 2

    mean_1 = jnp.tensordot(weights, data_1, axes=(0, 0))
    mean_2 = jnp.tensordot(weights, data_2, axes=(0, 0))
    var_1 = jnp.tensordot(weights, sq_1, axes=(0, 0))
    var_2 = jnp.tensordot(weights, sq_2, axes=(0, 0))

    m_11 = jnp.tensordot(weights, _get_covar_per_sample(data_1, data_2), axes=(0, 0))
    m_20 = jnp.tensordot(weights, _outer_per_sample(data_1, data_2), axes=(0, 0))
    m_sq1 = jnp.tensordot(weights, _outer_per_sample(sq_1, data_2), axes=(0, 0))
    m_sq2 = jnp.tensordot(weights, _outer_per_sample(data_1, sq_2), axes=(0, 0))
    m_sqsq = jnp.tensordot(weights, _outer_per_sample(sq_1, sq_2), axes=(0, 0))

    return mean_1, mean_2, var_1, var_2, m_11, m_20, m_sq1, m_sq2, m_sqsq

@jax.jit
def _finalize_covar_and_covar_var(mean_1, mean_2, var_1, var_2, m_11, m_20, m_sq1, m_sq2, m_sqsq):
    covar = m_11 - jnp.outer(jnp.conj(mean_1), mean_2)

    second_moment = (
        m_sqsq
        - 2 * jnp.real(jnp.conj(mean_2)[None, :] * m_sq1)
        - 2 * jnp.real(jnp.conj(mean_1)[:, None] * m_sq2)
        + 2 * jnp.real(jnp.outer(jnp.conj(mean_1), jnp.conj(mean_2)) * m_20)
        + 2 * jnp.real(jnp.outer(mean_1, jnp.conj(mean_2)) * m_11)
        + jnp.outer(var_1, jnp.abs(mean_2) ** 2)
        + jnp.outer(jnp.abs(mean_1) ** 2, var_2)
        - 3 * jnp.outer(jnp.abs(mean_1) ** 2, jnp.abs(mean_2) ** 2)
    )
    covar_var = second_moment - jnp.abs(covar) ** 2

    return covar, covar_var

@jax.jit(static_argnums=(1,))
def _apply_and_project(data, apply_fn, projection):
    return jnp.matmul(projection, apply_fn(data)) 

@jax.jit
def _get_tangent_kernel(norm_data):
    return jnp.matmul(norm_data, jnp.conj(jnp.transpose(norm_data)))

@jax.jit(static_argnums=(1, 2))
@partial(jax.vmap, in_axes=(0, None, None))
def _get_autocorrelation_time(x, c, dim):
    n_fft = 2 ** (dim + 1)  
    fft = jnp.abs(jnp.fft.fft(x - jnp.mean(x), n=n_fft))
    correlation = jnp.fft.ifft(fft ** 2)[:x.size].real
    correlation = correlation / correlation[0]
    tau = 2 * jnp.cumsum(correlation) - 1

    m = jnp.arange(len(tau)) < c * tau
    window = jax.lax.cond(jnp.any(m), lambda: jnp.argmin(m), lambda: len(tau) - 1)

    return tau[window]

class SampledObs():
    def __init__(self, observations, weights=None):
        """
        Initializes SampledObs class.

        Args:
            * ``observations``: Observations :math:`O_n` in the sample. This can be the value of an observable `O(s_n)`. \
                The array must have a leading batch dimension.
            * ``weights``: Weights :math:`w_n` associated with observation :math:`O_n`.
        """
        if len(observations.shape) == 1:
            self._num_obs = 1
            observations = observations.reshape((-1, 1))
        else:
            self._num_obs = jnp.prod(jnp.array(observations.shape[1:]))
        self._num_samples = observations.shape[0]
        num_devices = MESH.shape["devices"]

        if weights is None:
            weights = jnp.ones(self._num_samples, dtype=observations.dtype) / self._num_samples
        elif weights.shape != (self._num_samples,):
            raise ValueError(f"Weights must have shape ({self._num_samples},), got {weights.shape}")
        weights /= jnp.sum(weights)

        remainder = self._num_samples % num_devices
        if remainder != 0:
            num_pad = num_devices - remainder
            pad = ((0, num_pad),) + ((0, 0),) * (observations.ndim - 1)
            observations = jnp.pad(observations, pad, mode='constant')
            weights = jnp.pad(weights, (0, num_pad), constant_values=0)
        
        self._weights = jax.device_put(weights, DEVICE_SHARDING)
        self._observations = jax.device_put(observations, DEVICE_SHARDING, donate=True)

        self._consumed = False

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        if self._num_obs == 1:
            return f"{self.mean.item():.4e} ± {self.error_of_mean.item():.4e} (Var = {self.var.item():.4e})"
        else:
            return f"SampledObs with {self._num_obs} features"

    @property
    def observations(self):
        if self._consumed:
            raise RuntimeError(
                "This SampledObs was consumed by _get_normalized_obs_and_consume() "
                "(its buffer was donated) and can no longer be used."
            )
        return self._observations
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def mean(self):
        return _get_mean(self.observations, self.weights)

    @property
    def _centered_obs(self):
        return _center(self.observations, self.mean)

    @property
    def _normalized_obs(self):
        return _normalize(self.observations, self.weights, self.mean)
    
    @property
    def var(self):
        return _get_var(self._normalized_obs)
    
    @property
    def error_of_mean(self):
        return _get_error_of_mean(self.var, self.weights)
    
    @property
    def tangent_kernel(self):
        return _get_tangent_kernel(self._normalized_obs)

    def get_covar(self, other: SampledObs | None = None):
        """
        Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        if other is None:
            other = self
        
        return _get_covar(self._normalized_obs, other._normalized_obs)
    
    def get_covar_var(self, other: SampledObs | None = None):
        """
        Returns the variance of the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        if other is None:
            other = self
        
        return _get_covar_var(self._centered_obs, other._centered_obs, self.weights)
    
    def get_covar_obs(self, other: SampledObs | None = None) -> SampledObs:
        """
        Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        if other is None:
            other = self
        covar_per_sample = _get_covar_per_sample(self._centered_obs, other._centered_obs)

        return SampledObs(covar_per_sample, self.weights)
    
    def get_covar_and_covar_var(self, other: SampledObs | None = None):
        covar = self.get_covar(other)
        covar_var = self.get_covar_var(other)

        return covar, covar_var
    
    def get_R_hat(self, n_chains):
        """
        Compute the Gelman-Rubin R-hat convergence diagnostic.

        The samples are split into `n_chains` equal-length chains and the
        between-chain and within-chain variances are compared. If the chains
        have converged to the same distribution, R-hat will be close to 1.
        Convergence is typically assumed when R-hat < 1.01.

        Args:
            n_chains : int
                Number of chains 
        """
        if self._num_obs != 1:
            raise NotImplementedError

        chain_length = self._num_samples // n_chains
        chain_obs = SampledObs(self.observations.reshape((n_chains, chain_length)).T)

        B = jnp.var(chain_obs.mean.real, ddof=1)
        W = jnp.mean(chain_obs.var * chain_length / (chain_length - 1))
        
        return jnp.sqrt(((chain_length - 1) / chain_length * W + B) / W)
    
    def get_autocorrelation_time(self, n_chains, c=5) -> SampledObs:
        if self._num_obs != 1:
            raise NotImplementedError

        chain_length = self._num_samples // n_chains
        obs = self.observations.reshape((n_chains, chain_length))
        z = obs - jnp.mean(obs, axis=-1)[:, None]
        dim = int(jnp.ceil(jnp.log2(chain_length)))
        
        return SampledObs(_get_autocorrelation_time(z, c, dim))
    
    def transform(self, element_wise_fn=lambda x: x, linear_map=None) -> SampledObs:
        """
        Apply a transformation to observations and return a new SampledObs.
            
        The transformation is applied in two stages:
        1. Element-wise function applied to each observation
        2. (Optional) Linear projection via matrix multiplication
        
        Args:
            element_wise_fun: Function applied element-wise to each observation.
            linear_map: Optional linear transformation matrix applied after element_wise_fun.
        
        Returns:
            SampledObs: New instance with transformed observations.
        """
        if linear_map is not None:
            new_obs = _apply_and_project(self.observations, element_wise_fn, linear_map)
        else:
            new_obs = jax.jit(element_wise_fn)(self.observations)
        
        return SampledObs(new_obs, self.weights)
    
    def select(self, idx):
        """
        Returns a `SampledObs` for the data selection indicated by the given indices.

        Args:
            * ``idx``: Indices of selected data.
        """
        return SampledObs(self.observations[:, idx], self.weights)
    
    def get_subset(self, start=None, end=None, step=None) -> SampledObs:
        """
        Returns a `SampledObs` for a subset of the observartions.

        Args:
            * ``start``: Start sample index for subset selection
            * ``end``: End sample index for subset selection
            * ``step``: Sample index step for subset selection
        """ 
        sl = slice(start, end, step)
        new_weights = self.weights[sl]

        return SampledObs(self.observations[sl], new_weights / jnp.sum(new_weights))
    
    def _get_normalized_obs_and_consume(self):
        norm_obs = _normalize_no_copy(self.observations, self.weights)
        self._consumed = True

        return norm_obs

from jVMC_exp.sharding_config import SizedIterable

def _reshape_in_batches(data, n_batches: int):
    remainder = data.shape[0] % n_batches
    if remainder != 0:
        num_pad = n_batches - remainder
        data = jnp.pad(data, ((0, num_pad),) + ((0, 0),) * (data.ndim - 1), constant_values=0)
    batch_size = data.shape[0] // n_batches

    batched_data = []
    for i in range(n_batches):
        batched_data.append(
            jax.device_put(data[i * batch_size:(i + 1) * batch_size], DEVICE_SHARDING)
        )
    
    return batched_data

class LazySampledObs():
    def __init__(self, observations: SizedIterable, weights):
        if weights is None:
            raise ValueError("LazySampledObs require weights to be an array")
        weights /= jnp.sum(weights)
        self._num_samples = len(weights)
        
        self._weights = _reshape_in_batches(weights, len(observations))
        self._observations = observations
    
    @property
    def weights(self):
        return jnp.concatenate(self._weights)
    
    @cached_property
    def mean(self):
        mean = 0
        for batch, weights in zip(self._observations, self._weights):
            mean += _get_mean(batch, weights)
        
        return mean

    @cached_property
    def var(self):
        var = 0
        for batch, weights in zip(self._observations, self._weights):
            var += _get_var_not_normed(batch, weights)
        
        return var
    
    @property
    def error_of_mean(self):
        return _get_error_of_mean(self.var, self.weights)

    def get_covar(self, other: SampledObs | LazySampledObs | None = None):
        """
        Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        covar = 0
        if other is None:
            mean = 0
            for batch, weights in zip(self._observations, self._weights):
                weighted_data = _normalize_no_center(batch, weights)
                mean += _get_mean(batch, weights)
                covar += _get_covar(weighted_data, weighted_data)

            return covar - jnp.tensordot(jnp.conj(mean), mean, axes=0)

        elif isinstance(other, SampledObs):
            normalized_obs_other = _reshape_in_batches(other._normalized_obs, len(self._observations))
            
            for batch, weights, batch_other in zip(self._observations, self._weights, normalized_obs_other):
                weighted_data = _normalize_no_center(batch, weights)
                covar += _get_covar(weighted_data, batch_other)

            return covar
        
        elif isinstance(other, LazySampledObs):
            mean_1 = 0
            mean_2 = 0
            for batch_1, weights_1, batch_2, weights_2 in zip(self._observations, self._weights, other._observations, other._weights):
                weighted_data_1 = _normalize_no_center(batch_1, weights_1)
                weighted_data_2 = _normalize_no_center(batch_2, weights_2)
                mean_1 += _get_mean(batch_1, weights_1)
                mean_2 += _get_mean(batch_2, weights_2)
                covar += _get_covar(weighted_data_1, weighted_data_2)

            return covar - jnp.tensordot(jnp.conj(mean_1), mean_2, axes=0)
        
        else:
            raise NotImplementedError(
                "Can only compute the variance with a SampledObs or a LazySampledObs, "
                f"got {other}"
            )
    
    def get_covar_and_covar_var(self, other: SampledObs | LazySampledObs | None = None):
        moments = None
        def _accumulate(moments, batch_1, batch_2, weights):
            new_moments = _covar_stats(batch_1, batch_2, weights)
            if moments is None:
                return new_moments
            
            return tuple(m + b for m, b in zip(moments, new_moments))     

        if other is None:
            for batch, weights in zip(self._observations, self._weights):
                moments = _accumulate(moments, batch, batch, weights)

        elif isinstance(other, SampledObs):
            observations_other = _reshape_in_batches(other._observations, len(self._observations))
            for batch, weights, batch_other in zip(self._observations, self._weights, observations_other):
                moments = _accumulate(moments, batch, batch_other, weights)

        elif isinstance(other, LazySampledObs):
            for batch, weights, batch_other in zip(self._observations, self._weights, other._observations):
                moments = _accumulate(moments, batch, batch_other, weights)
        
        else:
            raise NotImplementedError(
                "Can only compute the variance with a SampledObs or a LazySampledObs, "
                f"got {other}"
            )

        covar = moments[2] - moments[0] * moments[1]
        covar_var = moments[3] - moments[2]**2

        return covar, covar_var

def _covar_stats(data_1, data_2, weights):
    xy_i = jnp.einsum('i..., i... -> i...', jnp.conj(data_1), data_2)
    xy_sq_i = jnp.abs(xy_i)**2
    xy_mean = jnp.einsum('i, i...', weights, xy_i)
    xy_sq_mean = jnp.einsum('i, i...', weights, xy_sq_i)
    x_mean = jnp.einsum('i, i...', weights, data_1)
    y_mean = jnp.einsum('i, i...', weights, data_2)

    return x_mean, y_mean, xy_mean, xy_sq_mean
        
    # def get_covar_and_covar_var(self, other: SampledObs | LazySampledObs | None = None):
    #     """
    #     Returns the covariance and the variance of the covariance.

    #     Both quantities are accumulated together over a single pass through
    #     the underlying iterable(s), so this should be preferred over calling
    #     `get_covar` and `get_covar_var` separately.

    #     Args:
    #         * ``other`` [optional]: Another instance of `SampledObs` or `LazySampledObs`.
    #     """
    #     def _accumulate(moments, batch_1, batch_2, weights):
    #         batch_moments = _get_covar_var_moments(batch_1, batch_2, weights)
    #         if moments is None:
    #             return batch_moments
    #         return tuple(m + b for m, b in zip(moments, batch_moments))

    #     moments = None

    #     if other is None:
    #         for batch, weights in zip(self._observations, self._weights):
    #             moments = _accumulate(moments, batch, batch, weights)

    #     elif isinstance(other, SampledObs):
    #         other_batches = _reshape_in_batches(other.observations, len(self._observations))

    #         for batch, weights, batch_other in zip(self._observations, self._weights, other_batches):
    #             moments = _accumulate(moments, batch, batch_other, weights)

    #     elif isinstance(other, LazySampledObs):
    #         for batch_1, weights_1, batch_2 in zip(self._observations, self._weights, other._observations):
    #             moments = _accumulate(moments, batch_1, batch_2, weights_1)

    #     else:
    #         raise NotImplementedError(
    #             "Can only compute the variance with a SampledObs or a LazySampledObs, "
    #             f"got {other}"
    #         )

    #     return _finalize_covar_and_covar_var(*moments)

    # def transform(self, element_wise_fn=lambda x: x, linear_map=None) -> LazySampledObs:
    #     if linear_map is not None:
    #         raise NotImplementedError(
    #             "A linear map can't be applied withoud materializing all the observables"
    #         )
        
    #     jitted_fn = jax.jit(element_wise_fn)
    #     transormed_iterable = lambda: (jitted_fn(batch) for batch in self._observations)
    #     new_obs = SizedIterable(transormed_iterable, self._observations.n_iterations)

    #     return LazySampledObs(new_obs, self.weights)