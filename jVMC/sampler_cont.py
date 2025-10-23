import jax
from sampler import MCSampler
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import vmap
import time

import jVMC.mpi_wrapper as mpi
from jVMC.nets.sym_wrapper import SymNet

from functools import partial

import jVMC.global_defs as global_defs

def propose_RWM(key, s, info):
    """
    Proposal move for random walk metropolis (RWM).
    
    Args:
        key: An instance of ``jax.random.PRNGKey``.
        s: The configuration.
        Info: A dictonary containing the standard deviation Σ for the random displacement.

    Return: 
        s': The new proposed configuration. 
    """
    geometry = info['Geometry']

    try:
        sigma = info['Sigma']
    except:
        raise ValueError('The value of sigma has to be passed to the updateProposer.')

    dx = (sigma * jax.random.normal(key, s.shape, dtype=s.dtype))   
    modulus = jnp.where(jnp.array(geometry.PBC), jnp.array(geometry.L), jnp.inf)
    return ((s + dx) % modulus).astype(s.dtype)

class Geometry():
    def __init__(self, n_particles, n_dim, PBC, extent=None):
        self._n_particles = n_particles
        self._n_dim = n_dim 

        if not isinstance(PBC, tuple):
            PBC = (PBC, ) * self.n_dim
        self._PBC = PBC

        if extent is None:
            extent = (1,) * self.n_dim
        elif not isinstance(extent, tuple):
            raise ValueError(f'Extent has to be a tuple, got {extent}.')
        self._extent = extent

    @property
    def n_particles(self):
        return self._n_particles
    
    @property
    def n_dim(self):
        return self._n_dim
    
    @property
    def PBC(self):
        return self._PBC
    
    @property
    def extent(self):
        return self._extent
    
class MCSamplerCont(MCSampler):
    def __init__(self, net, geometry, key=None, updateProposer=None, numChains=1, updateProposerArg=None, 
                numSamples=100, thermalizationSweeps=None, sweepSteps=10, initState=None, mu=2, logProbFactor=0.5):
        
        if not isinstance(geometry, Geometry):
            raise ValueError(f'The argument \'geomery\' has to be an istance of the class \'jVMC.sampler_cont.Geometry\'.')
        sampleShape = (geometry.n_particles, geometry.n_dim)
        self.geometry = geometry

        if updateProposerArg is None:
            updateProposerArg = {'Geometry': geometry}
        elif not isinstance(updateProposerArg, list):
            raise ValueError('The argument \'updateProposerArg\' has to be a dictonary or None.')
        else:
            updateProposerArg['Geometry'] = geometry

        super().__init__(net, sampleShape, key, updateProposer, numChains, updateProposerArg, 
                        numSamples, thermalizationSweeps, sweepSteps, initState, mu, logProbFactor)    

    def _get_samples(self, params, numSamples,
                     thermSweeps, sweepSteps,
                     states, logAccProb, key,
                     numProposed, numAccepted,
                     updateProposer, updateProposerArg,
                     sampleShape, sweepFunction=None):

        # Thermalize
        states, logAccProb, key, numProposed, numAccepted =\
            sweepFunction(states, logAccProb, key, numProposed, numAccepted, params, thermSweeps * sweepSteps, updateProposer, updateProposerArg)

        # Collect samples
        def scan_fun(c, x):

            states, logAccProb, key, numProposed, numAccepted =\
                sweepFunction(c[0], c[1], c[2], c[3], c[4], params, sweepSteps, updateProposer, updateProposerArg)

            return (states, logAccProb, key, numProposed, numAccepted), states

        meta, configs = jax.lax.scan(scan_fun, (states, logAccProb, key, numProposed, numAccepted), None, length=numSamples)

        # return meta, configs.reshape((configs.shape[0]*configs.shape[1], -1))
        return meta, configs.reshape((configs.shape[0] * configs.shape[1],) + sampleShape)

    def _sweep(self, states, logAccProb, key, numProposed, numAccepted, params, numSteps, updateProposer, updateProposerArg, net=None):

        def perform_mc_update(i, carry):

            # Generate update proposals
            newKeys = random.split(carry[2], carry[0].shape[0] + 1)
            carryKey = newKeys[-1]
            newStates = vmap(updateProposer, in_axes=(0, 0, None))(newKeys[:len(carry[0])], carry[0], updateProposerArg)

            # Compute acceptance probabilities
            newLogAccProb = jax.vmap(lambda y: self.mu * jnp.real(net(params, y)), in_axes=(0,))(newStates)
            P = jnp.exp(newLogAccProb - carry[1])

            # Roll dice
            newKey, carryKey = random.split(carryKey,)
            accepted = random.bernoulli(newKey, P).reshape((-1,))

            # Bookkeeping
            numProposed = carry[3] + len(newStates)
            numAccepted = carry[4] + jnp.sum(accepted)

            # Perform accepted updates
            def update(acc, old, new):
                return jax.lax.cond(acc, lambda x: x[1], lambda x: x[0], (old, new))
            carryStates = vmap(update, in_axes=(0, 0, 0))(accepted, carry[0], newStates)

            carryLogAccProb = jnp.where(accepted == True, newLogAccProb, carry[1])

            return (carryStates, carryLogAccProb, carryKey, numProposed, numAccepted)

        (states, logAccProb, key, numProposed, numAccepted) =\
            jax.lax.fori_loop(0, numSteps, perform_mc_update, (states, logAccProb, key, numProposed, numAccepted))

        return states, logAccProb, key, numProposed, numAccepted