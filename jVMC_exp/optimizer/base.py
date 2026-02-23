import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod

import jVMC_exp
from jVMC_exp.stats import SampledObs
from jVMC_exp.vqs import NQS
from jVMC_exp.sampler import AbstractMCSampler
from jVMC_exp.util.output_manager import OutputManager
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.optimizer.stepper import AbstractStepper

def _eigh_numpy(S):
    e, V = np.linalg.eigh(np.array(S))
    return jnp.array(e), jnp.array(V)

@jax.jit 
def realFun(x):
    return jnp.real(x)

@jax.jit
def imagFun(x):
    return 1j * jnp.imag(x)

class AbstractOptimizer(ABC):
    def __init__(
            self, sampler:AbstractMCSampler, psi: NQS, stepper: AbstractStepper, 
            output_manager: None | OutputManager=None, use_cross_valiadation: bool=False
        ):
        self._sampler = sampler
        self._psi = psi
        self._stepper = stepper
        self._output_manager = output_manager
        self.use_cross_valiadation = use_cross_valiadation
        self.meta_data = {}

    @property
    def output_manager(self):
        return self._output_manager
    
    @property
    def sampler(self):
        return self._sampler
    
    @property
    def psi(self):
        return self._psi

    def __call__(
            self, netParameters, t, *, psi: NQS, hamiltonian: AbstractOperator,
            numSamples=None, intStep=None
        ):
        """ 
        Arguments:
            * ``netParameters``: Parameters of the NQS.
            * ``t``: Current time.
            * ``psi``: NQS ansatz. Instance of ``jVMC.vqs.NQS``.
            * ``hamiltonian``: Hamiltonian operator, i.e., an instance of a derived class of ``jVMC.operator.Operator``. \
                                *Notice:* Current time ``t`` is by default passed as argument when computing matrix elements. 

        Further optional keyword arguments:
            * ``numSamples``: Number of samples to be used by MC sampler.
            * ``outp``: An instance of ``jVMC.OutputManager``. If ``outp`` is given, timings of the individual steps \
                are recorded using the ``OutputManger``.
            * ``intStep``: Integration step number of multi step method like Runge-Kutta. This information is used to store \
                quantities like energy or residuals at the initial integration step.
        """
        tmpParameters = psi.parameters
        psi.parameters = netParameters

        def start_timing(name):
            if self.output_manager is not None:
                self.output_manager.start_timing(name)

        def stop_timing(name, waitFor=None):
            if waitFor is not None:
                waitFor.block_until_ready()
            if self.output_manager is not None:
                self.output_manager(name)

        # Get sample
        start_timing("sampling")
        sampleConfigs, sampleLogPsi, p = self.sampler.sample(numSamples=numSamples)
        stop_timing("sampling", waitFor=sampleConfigs)

        # Evaluate local energy
        start_timing("compute Eloc")
        Eloc = hamiltonian.get_O_loc(sampleConfigs, psi, LogPsiS=sampleLogPsi, t=t)
        stop_timing("compute Eloc", waitFor=Eloc)
        self.Eloc = SampledObs(Eloc, p)

        # Evaluate gradients
        start_timing("compute gradients")
        sampleGradients = psi.gradients(sampleConfigs)
        stop_timing("compute gradients", waitFor=sampleGradients)
        sampleGradients = SampledObs(sampleGradients, p)

        start_timing("solve TDVP eqn.")
        update = self.solve(self.Eloc, sampleGradients)
        stop_timing("solve TDVP eqn.")

        psi.parameters = tmpParameters

        if intStep is not None:
            if intStep == 0:
                self.Eloc0 = self.Eloc
                self._update_meta_data()

                if self.use_cross_valiadation:
                    self.cross_validation()

        return update

    @abstractmethod
    def solve(self, Eloc: SampledObs, gradients: SampledObs):
        '''
        Return the update.
        '''
        pass

    @abstractmethod
    def _update_meta_data(self):
        '''
        Update the dictionary self.meta_data
        '''
        pass

    @abstractmethod
    def cross_validation(self):
        pass