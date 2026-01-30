from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

import jVMC
import jVMC.mpi_wrapper as mpi
import jVMC.global_defs as global_defs
from jVMC.global_defs import pmap_for_my_devices

from jVMC.sharding_config import DEVICE_SHARDING

import numpy as np

from functools import partial

@jax.jit
def get_covar(a, b):
    return jnp.tensordot(jnp.conj(a), b, axes=(0, 0))

def get_covar_var(a, b, w):
    covar = jnp.outer(jnp.conj(a), b)

    return jnp.sum(jnp.abs(covar) ** 2 / w[:, None, None]) - jnp.abs(jnp.sum(covar)) ** 2

_mean_helper = None
_data_prep = None
_covar_helper = None
_covar_var_helper = None
_covar_data_helper = None
_trafo_helper_1 = None
_trafo_helper_2 = None
_select_helper = None
_get_subset_helper = None
_subset_mean_helper = None
_subset_data_prep = None

statsPmapDevices = None

def jit_my_stuff():
    # This is a helper function to make sure that pmap'd functions work with the actual choice of devices
    # at all times.

    global _mean_helper
    global _covar_helper
    global _covar_var_helper
    global _covar_data_helper
    global _trafo_helper_1
    global _trafo_helper_2
    global _select_helper
    global _data_prep
    global _get_subset_helper
    global _subset_mean_helper
    global _subset_data_prep

    global statsPmapDevices

    if jVMC.global_defs.pmap_devices_updated(statsPmapDevices):

        statsPmapDevices = global_defs.myPmapDevices

        _mean_helper = jVMC.global_defs.pmap_for_my_devices(lambda data, w: jnp.tensordot(w, data, axes=(0,0)), in_axes=(0, 0))
        _data_prep = jVMC.global_defs.pmap_for_my_devices(lambda data, w, mean: jax.vmap(lambda d, w, m: jnp.sqrt(w) * (d - m), in_axes=(0,0,None))(data, w, mean), in_axes=(0, 0, None))
        _covar_helper = jVMC.global_defs.pmap_for_my_devices(
                                lambda data1, data2:
                                    jnp.tensordot(
                                        jnp.conj(data1),
                                        data2, axes=(0,0)), 
                                in_axes=(0, 0)
                                )
        _covar_var_helper = jVMC.global_defs.pmap_for_my_devices(
                                    lambda data1, data2, w: 
                                        jnp.sum(
                                            jnp.abs( 
                                                jax.vmap(lambda a,b: jnp.outer(a,b))(jnp.conj(data1), data2),
                                            )**2 / w[...,None,None],
                                            axis=0),
                                    in_axes=(0, 0, 0)
                                    )
        _covar_data_helper = jVMC.global_defs.pmap_for_my_devices(lambda data1, data2, w: jax.vmap(lambda a,b,w: jnp.outer(a,b) / w)(jnp.conj(data1), data2, w), in_axes=(0, 0, 0))
        _trafo_helper_1 = jVMC.global_defs.pmap_for_my_devices(
                                lambda data, w, mean, f: f(
                                    jax.vmap(lambda x,y: x/jnp.sqrt(y), in_axes=(0,0))(data, w) 
                                    + mean
                                    ), 
                                in_axes=(0, 0, None), static_broadcasted_argnums=(3,))
        _trafo_helper_2 = jVMC.global_defs.pmap_for_my_devices(
                                lambda data, w, mean, v, f: 
                                    jnp.matmul(v, 
                                                f(
                                                jax.vmap(lambda x,y: x/jnp.sqrt(y), in_axes=(0,0))(data, w) 
                                                + mean
                                                )
                                    ), 
                                in_axes=(0, 0, None, None), static_broadcasted_argnums=(4,))
        _select_helper = jVMC.global_defs.pmap_for_my_devices( lambda ix,g: jax.vmap(lambda ix,g: g[ix], in_axes=(None, 0))(ix,g), in_axes=(None, 0) )
        _get_subset_helper = jVMC.global_defs.pmap_for_my_devices(lambda x, ixs: x[slice(*ixs)], in_axes=(0,), static_broadcasted_argnums=(1,))
        _subset_mean_helper = jVMC.global_defs.pmap_for_my_devices(lambda d, w, m: jnp.tensordot(jnp.sqrt(w), d, axes=(0,0)) + m, in_axes=(0,0,None))
        _subset_data_prep = jVMC.global_defs.pmap_for_my_devices(jax.vmap(lambda d, w, m1, m2: d+jnp.sqrt(w)*(m1-m2), in_axes=(0,0,None,None)), in_axes=(0,0,None,None))

def flat_grad(fun):

    def grad_fun(*args):
        grad_tree = jax.grad(fun)(*args)

        dtypes = [a.dtype for a in tree_flatten(args[0])[0]]
        if dtypes[0] == np.single or dtypes[0] == np.double:
            grad_vec = tree_flatten(
                        tree_map(
                            lambda x: x.ravel(), 
                            grad_tree
                            )
                        )[0]
        else:
            grad_vec = tree_flatten(
                    tree_map(
                        lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], 
                        grad_tree
                        )
                    )[0]
            
        return jnp.concatenate(grad_vec)
    
    return grad_fun


class SampledObs():
    """
    This class implements the computation of statistics from Monte Carlo or exact samples.

    Initializer arguments:
        * ``observations``: Observations :math:`O_n` in the sample. This can be the value of an observable `O(s_n)` or the \
                plain configuration `s_n`. The array must have a leading device dimension plus a batch dimension.
        * ``weights``: Weights :math:`w_n` associated with observation :math:`O_n`.
        * ``estimator``: [optional] Function :math:`O(\\theta, s)` that computes an estimator parametrized by :math:`\\theta`
        * ``params``: [optional] A set of parameters for the estimator function.
    """

    def __init__(self, observations=None, weights=None):
        """
        Initializes SampledObs class.

        Args:
            * ``observations``: Observations :math:`O_n` in the sample. This can be the value of an observable `O(s_n)` or the \
                plain configuration `s_n`. The array must have a leading batch dimension.
            * ``weights``: Weights :math:`w_n` associated with observation :math:`O_n`.
        """

        self._weights = jax.device_put(weights, DEVICE_SHARDING)
        self._observations = jax.device_put(observations, DEVICE_SHARDING)

    @property
    def observations(self):
        return self._observations
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def mean(self):
        return jnp.mean(self.observations * self.weights)

    @property
    def _normalized_obs(self):
        return jnp.sqrt(self.weights) * (self.observations - self.mean)
    
    @property
    def var(self):
        return (jnp.abs(self._normalized_obs)**2)

    def covar(self, other: SampledObs | None = None):
        """
        Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        if other is None:
            other = self
        
        return get_covar(self._normalized_obs, other._normalized_obs)
    
    def covar_var(self, other: SampledObs | None = None):
        """
        Returns the variance of the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        if other is None:
            other = self
        

    def transform(self, nonLinearFun=lambda x: x, linearFun=None):
        """Returns a `SampledObs` for the transformed data.

        Args:
            * ``fun``: A function.
        """

        if linearFun is None:
            return SampledObs( _trafo_helper_1(self._data, self._weights, self._mean, nonLinearFun), self._weights )
        
        return SampledObs( _trafo_helper_2(self._data, self._weights, self._mean, linearFun, nonLinearFun), self._weights )


    def select(self, ixs):
        """Returns a `SampledObs` for the data selection indicated by the given indices.

        Args:
            * ``ixs``: Indices of selected data.
        """

        newObs = SampledObs()
        newObs._data = _select_helper(ixs, self._data)
        newObs._mean = self._mean[ixs]
        newObs._weights = self._weights

        return newObs
    
    
    def subset(self, start=None, end=None, step=None):
        """Returns a `SampledObs` for a subset of the data.

        Args:
            * ``start``: Start sample index for subset selection
            * ``end``: End sample index for subset selection
            * ``step``: Sample index step for subset selection
        """ 

        newObs = SampledObs()
        newObs._weights = _get_subset_helper(self._weights, (start, end, step))
        normalization = mpi.global_sum(newObs._weights)
        newObs._data = _get_subset_helper(self._data, (start, end, step))
        newObs._weights = newObs._weights / normalization
        newObs._data = newObs._data / jnp.sqrt(normalization)

        newObs._mean = mpi.global_sum( _subset_mean_helper(newObs._data, newObs._weights, 0.0)[:,None,...] )  + self._mean
        newObs._data = _subset_data_prep(newObs._data, newObs._weights, self._mean, newObs._mean)

        return newObs


    def tangent_kernel(self):

        all_data = mpi.gather(self._data)
        
        return jnp.matmul(all_data, jnp.conj(jnp.transpose(all_data)))


