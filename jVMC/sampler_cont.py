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
from propose_cont import AbstractPropose
    
class MCSamplerCont(MCSampler):
    def __init__(self, net, key=None, updateProposer=None, numChains=1, 
                numSamples=100, thermalizationSweeps=None, sweepSteps=10, initState=None, mu=2, logProbFactor=0.5):
        
        if not isinstance(updateProposer, AbstractPropose):
            raise ValueError(f'The argument \'updateProposer\' has to be an istance of the class \'jVMC.propose_cont.AbstractPropose\'.')
        sampleShape = (updateProposer.geometry.n_particles, updateProposer.geometry.n_dim)
        updateProposerArg = updateProposer.updateProposerArg

        super().__init__(net, sampleShape, key, updateProposer, numChains, updateProposerArg, 
                        numSamples, thermalizationSweeps, sweepSteps, initState, mu, logProbFactor)

    def _get_samples(self, params, numSamples,
                     thermSweeps, sweepSteps,
                     states, logAccProb, key,
                     numProposed, numAccepted,
                     updateProposer, updateProposerArg,
                     sampleShape, sweepFunction=None):

        # Thermalize
        if updateProposer._use_custom_thermalization:
            states, logAccProb, key, numProposed, numAccepted, updateProposerArg =\
                updateProposer._custom_therm_fun(states, logAccProb, key, numProposed, numAccepted, params, sweepSteps, thermSweeps, sweepFunction, updateProposerArg)
            updateProposer.updateProposerArg = updateProposerArg
        else:
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