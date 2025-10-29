import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import vmap
import time
from functools import partial

import jVMC.global_defs as global_defs
import jVMC.mpi_wrapper as mpi
from jVMC.nets.sym_wrapper import SymNet
from jVMC.propose import AbstractProposeCont
from jVMC.util.key_gen import format_key

# Deprecated import 
import warnings
from jVMC.propose import (
    propose_POVM_outcome as _propose_POVM_outcome,
    propose_spin_flip as _propose_spin_flip,
    propose_spin_flip_Z2 as _propose_spin_flip_Z2,
    propose_spin_flip_zeroMag as _propose_spin_flip_zeroMag,
)
_deprecated_funcs = {
    "propose_POVM_outcome": _propose_POVM_outcome,
    "propose_spin_flip": _propose_spin_flip,
    "propose_spin_flip_Z2": _propose_spin_flip_Z2,
    "propose_spin_flip_zeroMag": _propose_spin_flip_zeroMag,
}
def _warn_deprecated(name):
    warnings.warn(
        f"Importing `{name}` from jVMC.sampler is deprecated. "
        f"Please import it from jVMC.propose instead.",
        DeprecationWarning,
        stacklevel=3,
    )
def __getattr__(name):
    if name in _deprecated_funcs:
        _warn_deprecated(name)
        return _deprecated_funcs[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
__all__ = list(_deprecated_funcs.keys())
# End deprecated import

class MCSampler:
    """A sampler class.

    This class provides functionality to sample computational basis states from \
    the distribution 

        :math:`p_{\\mu}(s)=\\frac{|\\psi(s)|^{\\mu}}{\\sum_s|\\psi(s)|^{\\mu}}`.

    For :math:`\\mu=2` this corresponds to sampling from the Born distribution. \
    :math:`0\\leq\\mu<2` can be used to perform importance sampling \
    (see `[arXiv:2108.08631] <https://arxiv.org/abs/2108.08631>`_).

    Sampling is automatically distributed accross MPI processes and locally available \
    devices.

    Initializer arguments:
        * ``net``: Network defining the probability distribution.
        * ``sampleShape``: Shape of computational basis configurations.
        * ``key``: An instance of ``jax.random.PRNGKey``. Alternatively, an ``int`` that will be used \
                   as seed to initialize a ``PRNGKey``.
        * ``updateProposer``: A function to propose updates for the MCMC algorithm. \
        It is called as ``updateProposer(key, config, **kwargs)``, where ``key`` is an instance of \
        ``jax.random.PRNGKey``, ``config`` is a computational basis configuration, and ``**kwargs`` \
        are optional additional arguments.
        * ``numChains``: Number of Markov chains, which are run in parallel.
        * ``updateProposerArg``: An optional argument that will be passed to the ``updateProposer`` \
        as ``kwargs``.
        * ``numSamples``: Default number of samples to be returned by the ``sample()`` member function.
        * ``thermalizationSweeps``: Number of sweeps to perform for thermalization of the Markov chain.
        * ``sweepSteps``: Number of proposed updates per sweep.
        * ``mu``: Parameter for the distribution :math:`p_{\\mu}(s)`, see above.
        * ``logProbFactor``: Factor for the log-probabilities, aquivalent to the exponent for the probability \
        distribution. For pure wave functions this should be 0.5, and 1.0 for POVMs. In the POVM case, the \
        ``mu`` parameter must be set to 1.0, to sample the unchanged POVM distribution.
    """

    def __init__(self, net, sampleShape, key=None, updateProposer=None, numChains=1, updateProposerArg=None,
                 numSamples=100, thermalizationSweeps=10, sweepSteps=None, initState=None, mu=2, logProbFactor=0.5):
        """Initializes the MCSampler class.
        """

        # Changed
        if isinstance(sampleShape, tuple):
            self.sampleShape = sampleShape
        else: 
            self.sampleShape = (sampleShape,)

        self.net = net
        if (not net.is_generator) and (updateProposer is None):
            raise RuntimeError("Instantiation of MCSampler: `updateProposer` is `None` and cannot be used for MCMC sampling.")
        self.orbit = None
        if isinstance(self.net.net, SymNet):
            self.orbit = self.net.net.orbit.orbit

        stateShape = (global_defs.device_count(), numChains) + self.sampleShape
        if initState is None:
            initState = jnp.zeros(self.sampleShape, dtype=np.int32)
        self.states = jnp.stack([initState] * (global_defs.device_count() * numChains), axis=0).reshape(stateShape)

        # Make sure that net is initialized
        self.net(self.states)
        self.sampler_net, _ = self.net.get_sampler_net()

        self.logProbFactor = logProbFactor
        self.mu = mu
        if mu < 0 or mu > 2:
            raise ValueError("mu must be in the range [0, 2]")
        self.updateProposer = updateProposer
        self.updateProposerArg = updateProposerArg

        # CHANGED
        self.key = key
        self.key = jax.random.split(self.key, mpi.commSize)[mpi.rank]
        self.key = jax.random.split(self.key, global_defs.device_count())

        # CHANGED
        if sweepSteps is None:
            sweepSteps = sampleShape[-1]
        self.sweepSteps = sweepSteps
        self.thermalizationSweeps = thermalizationSweeps
        self.numSamples = numSamples

        # TODO: Remove this, since it is already in self.mc_init()
        # shape = (global_defs.device_count(),) + (1,)
        # self.numProposed = jnp.zeros(shape, dtype=np.int64)
        # self.numAccepted = jnp.zeros(shape, dtype=np.int64)

        self.numChains = numChains

        # jit'd member functions
        self._get_samples_jitd = {}  # will hold a jit'd function for each number of samples
        self._randomize_samples_jitd = {}  # will hold a jit'd function for each number of samples


        # pmap'd helper function
        self._logAccProb_pmapd = global_defs.pmap_for_my_devices(self._logAccProb,
                                                                 in_axes=(0, None, None, None),
                                                                 static_broadcasted_argnums=(2,))

    def _logAccProb(self, x, mu, sampler_net, netParams):
        # vmap is over parallel MC chains
        return jax.vmap(lambda y: mu * jnp.real(sampler_net(netParams, y)), in_axes=(0,))(x)


    def set_number_of_samples(self, N):
        """Set default number of samples.

        Arguments:
            * ``N``: Number of samples.
        """

        self.numSamples = N

    # TODO: Now can be deleted
    def set_random_key(self, key):
        """Set key for pseudo random number generator.

        Args:
            * ``key``: Key (jax.random.PRNGKey)
        """

        self.key = jax.random.split(key, global_defs.device_count())

    # CHANGED
    @property
    def key(self):
        return self._key
    
    @key.setter
    def key(self, value):
        self._key = format_key(value)

    def get_last_number_of_samples(self):
        """Return last number of samples.

        This function is required, because the actual number of samples might \
        exceed the requested number of samples when sampling is distributed \
        accross multiple processes or devices.

        Returns:
            Number of samples generated by last call to ``sample()`` member function.
        """
        return self.globNumSamples

    def sample(self, parameters=None, numSamples=None, multipleOf=1):
        """Generate random samples from wave function.

        If supported by ``net``, direct sampling is peformed. Otherwise, MCMC is run \
        to generate the desired number of samples. For direct sampling the real part \
        of ``net`` needs to provide a ``sample()`` member function that generates \
        samples from :math:`p_{\\mu}(s)`.

        Sampling is automatically distributed accross MPI processes and available \
        devices. In that case the number of samples returned might exceed ``numSamples``.

        Arguments:
            * ``parameters``: Network parameters to use for sampling.
            * ``numSamples``: Number of samples to generate. When running multiple processes \
            or on multiple devices per process, the number of samples returned is \
            ``numSamples`` or more. If ``None``, the default number of samples is returned \
            (see ``set_number_of_samples()`` member function).
            * ``multipleOf``: This argument allows to choose the number of samples returned to \
            be the smallest multiple of ``multipleOf`` larger than ``numSamples``. This feature \
            is useful to distribute a total number of samples across multiple processors in such \
            a way that the number of samples per processor is identical for each processor.

        Returns:
            A sample of computational basis configurations drawn from :math:`p_{\\mu}(s)`.
        """

        if numSamples is None:
            numSamples = self.numSamples

        if self.net.is_generator:
            if parameters is not None:
                tmpP = self.net.params
                self.net.set_parameters(parameters)
            configs, coeffs, ps = self._get_samples_gen(self.net.parameters, numSamples, multipleOf)
            if parameters is not None:
                self.net.params = tmpP
            return configs, coeffs, ps

        configs, logPsi = self._get_samples_mcmc(parameters, numSamples, multipleOf)
        p = jnp.exp((1.0 / self.logProbFactor - self.mu) * jnp.real(logPsi))
        return configs, logPsi, p / mpi.global_sum(p)

    def _randomize_samples(self, samples, key, orbit):
        """ For a given set of samples apply a random symmetry transformation to each sample
        """
        orbit_indices = random.choice(key, orbit.shape[0], shape=(samples.shape[0],))
        samples = samples * 2 - 1
        return jax.vmap(lambda o, idx, s: (o[idx].dot(s.ravel()).reshape(s.shape) + 1) // 2, in_axes=(None, 0, 0))(orbit, orbit_indices, samples)

    def _get_samples_gen(self, params, numSamples, multipleOf=1):

        numSamples, self.globNumSamples = mpi.distribute_sampling(numSamples, localDevices=global_defs.device_count(), numChainsPerDevice=multipleOf)

        tmpKeys = random.split(self.key[0], 3 * global_defs.device_count())
        self.key = tmpKeys[:global_defs.device_count()]
        tmpKey = tmpKeys[global_defs.device_count():2 * global_defs.device_count()]
        tmpKey2 = tmpKeys[2 * global_defs.device_count():]

        samples = self.net.sample(numSamples, tmpKey, parameters=params)

        if not str(numSamples) in self._randomize_samples_jitd:
            self._randomize_samples_jitd[str(numSamples)] = global_defs.pmap_for_my_devices(self._randomize_samples, static_broadcasted_argnums=(), in_axes=(0, 0, None))

        if not self.orbit is None:
            samples = self._randomize_samples_jitd[str(numSamples)](samples, tmpKey2, self.orbit)
            # return self._randomize_samples_jitd[str(numSamples)](samples, tmpKey2, self.orbit)
        
        return samples, self.net(samples), jnp.ones(samples.shape[:2]) / self.globNumSamples

    def _get_samples_mcmc(self, params, numSamples, multipleOf=1):

        tmpP = None
        if params is not None:
            tmpP = self.net.params
            self.net.set_parameters(params)

        net, params = self.net.get_sampler_net()

        if tmpP is not None:
            self.net.params = tmpP        

        # Initialize sampling stuff
        self._mc_init(params)

        numSamples, self.globNumSamples = mpi.distribute_sampling(numSamples, localDevices=global_defs.device_count(), numChainsPerDevice=np.lcm(self.numChains, multipleOf))
        numSamplesStr = str(numSamples)

        # check whether _get_samples is already compiled for given number of samples
        if not numSamplesStr in self._get_samples_jitd:
            self._get_samples_jitd[numSamplesStr] = global_defs.pmap_for_my_devices(partial(self._get_samples, sweepFunction=partial(self._sweep, net=net)),
                                                                                    static_broadcasted_argnums=(1, 2, 3, 9, 11),
                                                                                    in_axes=(None, None, None, None, 0, 0, 0, 0, 0, None, None, None))

        (self.states, self.logAccProb, self.key, self.numProposed, self.numAccepted), configs =\
            self._get_samples_jitd[numSamplesStr](params, numSamples, self.thermalizationSweeps, self.sweepSteps,
                                                  self.states, self.logAccProb, self.key, self.numProposed, self.numAccepted,
                                                  self.updateProposer, self.updateProposerArg, self.sampleShape)

        tmpP = self.net.params
        self.net.params = params["params"]
        coeffs = self.net(configs)
        self.net.params = tmpP

        # return configs, None
        return configs, coeffs

    def _get_samples(self, params, numSamples,
                     thermSweeps, sweepSteps,
                     states, logAccProb, key,
                     numProposed, numAccepted,
                     updateProposer, updateProposerArg,
                     sampleShape, sweepFunction=None):

        # Thermalize
        if self.thermalizationSweeps is not None:
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
            newStates = vmap(updateProposer, in_axes=(0, 0, None))(newKeys[:len(carry[0])], carry[0], updateProposerArg) # Might need to change it aswell with in_axes[3]=0

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

    def _mc_init(self, netParams):

        self.logAccProb = self._logAccProb_pmapd(self.states, self.mu, self.sampler_net, netParams)

        shape = (global_defs.device_count(),) + (1,)

        self.numProposed = jnp.zeros(shape, dtype=np.int64)
        self.numAccepted = jnp.zeros(shape, dtype=np.int64)

    def acceptance_ratio(self):
        """Get acceptance ratio.

        Returns:
            Acceptance ratio observed in the last call to ``sample()``.
        """

        numProp = mpi.global_sum(self.numProposed)
        if numProp > 0:
            return mpi.global_sum(self.numAccepted) / numProp

        return jnp.array([0.])

# ** end class Sampler

class ExactSampler:
    """Class for full enumeration of basis states.

    This class generates a full basis of the many-body Hilbert space. Thereby, it \
    allows to exactly perform sums over the full Hilbert space instead of stochastic \
    sampling.

    Initialization arguments:
        * ``net``: Network defining the probability distribution.
        * ``sampleShape``: Shape of computational basis states.
        * ``lDim``: Local Hilbert space dimension.
        * ``logProbFactor``: Factor for the log-probabilities, aquivalent to the exponent for the probability \
        distribution. For pure wave functions this should be 0.5, and 1.0 for POVMs.
    """

    def __init__(self, net, sampleShape, lDim=2, logProbFactor=0.5):

        self.psi = net
        self.N = jnp.prod(jnp.asarray(sampleShape))
        self.sampleShape = sampleShape
        self.lDim = lDim
        self.logProbFactor = logProbFactor

        # pmap'd member functions
        self._get_basis_ldim2_pmapd = global_defs.pmap_for_my_devices(self._get_basis_ldim2, in_axes=(0, 0, None), static_broadcasted_argnums=2)
        self._get_basis_pmapd = global_defs.pmap_for_my_devices(self._get_basis, in_axes=(0, 0, None, None), static_broadcasted_argnums=(2, 3))
        self._compute_probabilities_pmapd = global_defs.pmap_for_my_devices(self._compute_probabilities, in_axes=(0, None, 0))
        self._normalize_pmapd = global_defs.pmap_for_my_devices(self._normalize, in_axes=(0, None))

        self.get_basis()

        # Make sure that net params are initialized
        self.psi(self.basis)

        self.lastNorm = 0.

    def get_basis(self):

        myNumStates, _ = mpi.distribute_sampling(self.lDim**self.N)
        myFirstState = mpi.first_sample_id()

        deviceCount = global_defs.device_count()

        self.numStatesPerDevice = [(myNumStates + deviceCount - 1) // deviceCount] * deviceCount
        self.numStatesPerDevice[-1] += myNumStates - deviceCount * self.numStatesPerDevice[0]
        self.numStatesPerDevice = jnp.array(self.numStatesPerDevice)

        totalNumStates = deviceCount * self.numStatesPerDevice[0]

        intReps = jnp.arange(myFirstState, myFirstState + totalNumStates)
        intReps = intReps.reshape((global_defs.device_count(), -1))
        self.basis = jnp.zeros(intReps.shape + (self.N,), dtype=np.int32)
        if self.lDim == 2:
            self.basis = self._get_basis_ldim2_pmapd(self.basis, intReps, self.sampleShape)
        else:
            self.basis = self._get_basis_pmapd(self.basis, intReps, self.lDim, self.sampleShape)

    def _get_basis_ldim2(self, states, intReps, sampleShape):

        def make_state(state, intRep):

            def for_fun(i, x):
                return (jax.lax.cond(x[1] >> i & 1, lambda x: x[0].at[x[1]].set(1), lambda x: x[0], (x[0], i)), x[1])

            (state, _) = jax.lax.fori_loop(0, state.shape[0], for_fun, (state, intRep))

            return state.reshape(sampleShape)

        basis = jax.vmap(make_state, in_axes=(0, 0))(states, intReps)

        return basis

    def _get_basis(self, states, intReps, lDim, sampleShape):

        def make_state(state, intRep):

            def scan_fun(c, x):
                locState = c % lDim
                c = (c - locState) // lDim
                return c, locState

            _, state = jax.lax.scan(scan_fun, intRep, state)

            return state[::-1].reshape(sampleShape)

        basis = jax.vmap(make_state, in_axes=(0, 0))(states, intReps)

        return basis

    def _compute_probabilities(self, logPsi, lastNorm, numStates):

        # p = jnp.exp(2. * jnp.real(logPsi - lastNorm))
        p = jnp.exp(jnp.real(logPsi - lastNorm) / self.logProbFactor)

        def scan_fun(c, x):
            out = jax.lax.cond(c[1] < c[0], lambda x: x[0], lambda x: x[1], (x, 0.))
            newC = c[1] + 1
            return (c[0], newC), out

        _, p = jax.lax.scan(scan_fun, (numStates, 0), p)

        return p

    def _normalize(self, p, nrm):

        return p / nrm

    def sample(self, parameters=None, numSamples=None, multipleOf=None):
        """Return all computational basis states.

        Sampling is automatically distributed accross MPI processes and available \
        devices.

        Arguments:
            * ``parameters``: Dummy argument to provide identical interface as the \
            ``MCSampler`` class.
            * ``numSamples``: Dummy argument to provide identical interface as the \
            ``MCSampler`` class.
            * ``multipleOf``: Dummy argument to provide identical interface as the \
            ``MCSampler`` class.

        Returns:
            ``configs, logPsi, p``: All computational basis configurations, \
            corresponding wave function coefficients, and probabilities \
            :math:`|\\psi(s)|^2` (normalized).
        """

        if parameters is not None:
            tmpP = self.psi.get_parameters()
            self.psi.set_parameters(parameters)
        logPsi = self.psi(self.basis)
        if parameters is not None:
            self.psi.set_parameters(tmpP)

        p = self._compute_probabilities_pmapd(logPsi, self.lastNorm, self.numStatesPerDevice)

        nrm = mpi.global_sum(p)
        p = self._normalize_pmapd(p, nrm)

        self.lastNorm += self.logProbFactor * jnp.log(nrm)

        return self.basis, logPsi, p

    def set_number_of_samples(self, N):
        pass

    def get_last_number_of_samples(self):
        return jnp.inf

# ** end class ExactSampler

class MCSamplerCont(MCSampler):
    def __init__(self, net, key=None, updateProposer=None, numChains=1, 
                numSamples=2**12, thermalizationSweeps=10, sweepSteps=None, initState=None, mu=2, logProbFactor=0.5):
        
        if not isinstance(updateProposer, AbstractProposeCont):
            raise ValueError(f'The argument \'updateProposer\' has to be an istance of the class \'jVMC.propose.AbstractProposeCont\'.')
        sampleShape = (updateProposer.geometry.n_particles, updateProposer.geometry.n_dim)

        if sweepSteps is None:
            sweepSteps = updateProposer.geometry.n_particles

        super().__init__(net, sampleShape, key, updateProposer, numChains, None, 
                        numSamples, thermalizationSweeps, sweepSteps, initState, mu, logProbFactor)
        
        updateProposerArg = jnp.atleast_1d(self.updateProposer.Arg)
        self.updateProposerArg = jnp.stack([updateProposerArg] * (global_defs.device_count() * numChains), axis=0)
        self.updateProposerArg = self.updateProposerArg.reshape((global_defs.device_count(), numChains) + updateProposerArg.shape)
        
        init_keys = jax.random.split(format_key(key), global_defs.device_count())
        self.states = global_defs.pmap_for_my_devices(
            partial(self._init_state, initState=initState), in_axes=(0)
        )(init_keys)

    def _init_state(self, key, initState):
        stateShape = (self.numChains,) + self.sampleShape
        key_init = jax.random.split(format_key(key), self.numChains)
        # TODO: decide how to initialize
        init_fun = jax.jit(jax.vmap(partial(self.updateProposer.geometry.uniform_populate, dtype=jnp.float64)))

        if initState is None:
            initState = init_fun(key_init)
        elif initState.shape != stateShape:
            if initState.shape != self.sampleShape:
                raise ValueError(
                    "'initState' does not have the correct shape.\n" \
                    f"Allowed shapes are (n_chains, n_particles, n_dim) or (n_particles, n_dim), got {initState.shape}")
            warnings.warn(
                "Got a single sample to initialize all chains in 'initState'! \n" \
                "It is preferred to initialize each chain in a different way.")
            initState = jnp.stack([initState] * self.numChains).reshape(stateShape)
        return initState

    def _mc_init(self, netParams):

        self.logAccProb = self._logAccProb_pmapd(self.states, self.mu, self.sampler_net, netParams)

        shape = (global_defs.device_count(), self.numChains)
        self.numProposed = jnp.zeros(shape, dtype=np.int64)
        self.numAccepted = jnp.zeros(shape, dtype=np.int64)

    def _get_samples_mcmc(self, params, numSamples, multipleOf=1):

        tmpP = None
        if params is not None:
            tmpP = self.net.params
            self.net.set_parameters(params)

        net, params = self.net.get_sampler_net()

        if tmpP is not None:
            self.net.params = tmpP        

        # Initialize sampling stuff
        self._mc_init(params)

        numSamples, self.globNumSamples = mpi.distribute_sampling(numSamples, localDevices=global_defs.device_count(), numChainsPerDevice=np.lcm(self.numChains, multipleOf))
        numSamplesStr = str(numSamples)

        # check whether _get_samples is already compiled for given number of samples
        # pmap parallelizes across different workers
        if not numSamplesStr in self._get_samples_jitd:
            self._get_samples_jitd[numSamplesStr] = global_defs.pmap_for_my_devices(partial(self._get_samples, sweepFunction=partial(self._sweep, net=net)),
                                                                                    static_broadcasted_argnums=(1, 2, 3, 9, 11),
                                                                                    in_axes=(None, None, None, None, 0, 0, 0, 0, 0, None, 0, None))

        (self.states, self.logAccProb, self.key, self.numProposed, self.numAccepted), configs, self.updateProposerArg =\
            self._get_samples_jitd[numSamplesStr](params, numSamples, self.thermalizationSweeps, self.sweepSteps,
                                                  self.states, self.logAccProb, self.key, self.numProposed, self.numAccepted,
                                                  self.updateProposer, self.updateProposerArg, self.sampleShape)

        tmpP = self.net.params
        self.net.params = params["params"]
        coeffs = self.net(configs)
        self.net.params = tmpP

        # return configs, None
        return configs, coeffs

    def _get_samples(self, params, numSamples,
                     thermSweeps, sweepSteps,
                     states, logAccProb, key,
                     numProposed, numAccepted,
                     updateProposer, updateProposerArg,
                     sampleShape, sweepFunction=None):

        # Thermalize
        if self.thermalizationSweeps is not None:
            if updateProposer._use_custom_thermalization:
                states, logAccProb, key, numProposed, numAccepted, updateProposerArg =\
                    updateProposer._custom_therm_fun(states, logAccProb, key, numProposed, numAccepted, params, sweepSteps, thermSweeps, sweepFunction, updateProposerArg)
                # updateProposer.updateProposerArg = self.updateProposerArg = updateProposerArg
            else:
                states, logAccProb, key, numProposed, numAccepted =\
                    sweepFunction(states, logAccProb, key, numProposed, numAccepted, params, thermSweeps * sweepSteps, updateProposer, updateProposerArg)

        # Collect samples
        def scan_fun(c, x):

            states, logAccProb, key, numProposed, numAccepted =\
                sweepFunction(c[0], c[1], c[2], c[3], c[4], params, sweepSteps, updateProposer, updateProposerArg)

            return (states, logAccProb, key, numProposed, numAccepted), states

        meta, configs = jax.lax.scan(scan_fun, (states, logAccProb, key, numProposed, numAccepted), None, length=numSamples)

        return meta, configs.reshape((configs.shape[0] * configs.shape[1],) + sampleShape), updateProposerArg
    
    def _sweep(self, states, logAccProb, key, numProposed, numAccepted, params, numSteps, updateProposer, updateProposerArg, net=None):

        # All the vmaps here parallelize across chains 
        def perform_mc_update(i, carry):
            states, logAccProb, key, numProposed, numAccepted = carry
            
            # Generate update proposals
            newKeys = random.split(key, states.shape[0] + 1)
            carryKey = newKeys[-1]
            newStates = vmap(updateProposer)(newKeys[:len(states)], states, updateProposerArg)

            # Compute acceptance probabilities
            newLogAccProb = jax.vmap(lambda y: self.mu * jnp.real(net(params, y)), in_axes=(0,))(newStates)
            P = jnp.exp(newLogAccProb - logAccProb)

            # Roll dice
            newKey, carryKey = random.split(carryKey,)
            accepted = random.bernoulli(newKey, P).reshape((-1,))

            # Bookkeeping
            numProposed = numProposed + 1
            numAccepted = numAccepted + accepted

            # Perform accepted updates
            def update(acc, old, new):
                return jax.lax.cond(acc, lambda x: x[1], lambda x: x[0], (old, new))
            carryStates = vmap(update)(accepted, states, newStates)

            carryLogAccProb = jnp.where(accepted == True, newLogAccProb, logAccProb)

            return (carryStates, carryLogAccProb, carryKey, numProposed, numAccepted)

        carry = (states, logAccProb, key, numProposed, numAccepted)
        (states, logAccProb, key, numProposed, numAccepted) = jax.lax.fori_loop(0, numSteps, perform_mc_update, carry)

        return states, logAccProb, key, numProposed, numAccepted
