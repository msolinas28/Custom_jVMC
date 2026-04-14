import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
import tqdm
from typing import Dict

from jVMC_exp.stats import SampledObs
from jVMC_exp.vqs import NQS
from jVMC_exp.sampler import AbstractMCSampler, ExactSampler
from jVMC_exp.util.output_manager import OutputManager
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.optimizer.stepper import AbstractStepper
from jVMC_exp.util import ObservableEntry, measure
from jVMC_exp.solver.base import AbstractSolver, SolverState
from jVMC_exp.solver.pinv_snr import PinvSNR

@jax.jit 
def real_fn(x):
    return jnp.real(x)

@jax.jit
def imag_fn(x):
    return 1j * jnp.imag(x)

class AbstractOptimizer(ABC):
    def __init__(
            self, sampler:AbstractMCSampler, psi: NQS, stepper: AbstractStepper, 
            use_cross_valiadation: bool=False
        ):
        self._sampler = sampler
        self._psi = psi
        self._stepper = stepper
        self.use_cross_valiadation = use_cross_valiadation
        self.meta_data = {}
        self._output_manager = OutputManager()
        self.energy = None
        self._elapsed = 0

    @property
    def output_manager(self):
        return self._output_manager

    @property
    def sampler(self):
        return self._sampler
    
    @property
    def psi(self):
        return self._psi
    
    def __call__(self, parameters, t, *, hamiltonian: AbstractOperator, numSamples=None, intStep=None):
        """ 
        Arguments:
            * ``parameters``: Parameters of the NQS.
            * ``t``: Current time.
            * ``hamiltonian``: Hamiltonian operator, i.e., an instance of a derived class of ``jVMC.operator.Operator``. \
                                *Notice:* Current time ``t`` is by default passed as argument when computing matrix elements. 

        Further optional keyword arguments:
            * ``numSamples``: Number of samples to be used by MC sampler.
            * ``outp``: An instance of ``jVMC.OutputManager``. If ``outp`` is given, timings of the individual steps \
                are recorded using the ``OutputManger``.
            * ``intStep``: Integration step number of multi step method like Runge-Kutta. This information is used to store \
                quantities like energy or residuals at the initial integration step.
        """
        tmp_parameters = self.psi.parameters
        self.psi.parameters = parameters
        self._elapsed = 0
        
        def stop_timing(name, waitFor=None):
            if waitFor is not None:
                waitFor.block_until_ready()
            return self.output_manager.stop_timing(name)

        # Get sample
        self.output_manager.start_timing("sampling")
        sampleConfigs, sampleLogPsi, p = self.sampler.sample(numSamples=numSamples)
        self._elapsed += stop_timing("sampling", waitFor=sampleConfigs)

        # Evaluate local energy
        self.output_manager.start_timing("compute Eloc")
        Eloc = hamiltonian.get_O_loc(sampleConfigs, self.psi, LogPsiS=sampleLogPsi, t=t)
        self._elapsed += stop_timing("compute Eloc", waitFor=Eloc)
        self.Eloc = SampledObs(Eloc, p)

        # Evaluate gradients
        self.output_manager.start_timing("compute gradients")
        sampleGradients = self.psi.gradients(sampleConfigs)
        stop_timing("compute gradients", waitFor=sampleGradients)
        sampleGradients = SampledObs(sampleGradients, p)

        self.output_manager.start_timing("solve")
        update = self.solve(self.Eloc, sampleGradients)
        self._elapsed += stop_timing("solve")

        self.psi.parameters = tmp_parameters

        if intStep is not None:
            if intStep == 0:
                self.energy = self.Eloc
                self._update_meta_data()

                if self.use_cross_valiadation:
                    self.output_manager.start_timing("cross_validation")
                    self.cross_validation()
                    self._elapsed += stop_timing("cross_validation")

        return update
    
    def ground_state_search(
            self, 
            steps, 
            hamiltonian: AbstractOperator,
            observables: Dict[str, ObservableEntry] | None = None,
            save_meta_data: bool = False
        ):
        if not hasattr(self._stepper, "update_dt"):
            raise ValueError("For ground state search the stepper must " \
                            f"implement a mehod called 'update_dt'")

        pbar = tqdm.tqdm(range(steps))
        for n in pbar: 
            self._stepper.update_dt(n)
            self.psi.parameters, _ = self.step(hamiltonian)
            
            self._measure_and_store(n, observables, save_meta_data)

            pbar.set_postfix(E=f"{self.energy}")
  
        self.output_manager.print_timings()
        
        if save_meta_data:
            return self.output_manager.data['observables'], self.output_manager.data['metadata']
        
        return self.output_manager.data["observables"]

    def time_evolution(
            self,
            t_max,
            hamiltonian: AbstractOperator,
            observables: Dict[str, ObservableEntry] | None = None,
            save_meta_data: bool = False
        ):

        pbar = tqdm.tqdm(total=t_max)
        t = 0
        while t < t_max:
            self.psi.parameters, dt = self.step(hamiltonian)

            dt = float(dt)
            if t + dt >= t_max:
                dt = t_max - t
            t += dt

            self.meta_data['dt'] = dt
            self._measure_and_store(t, observables, save_meta_data)

            pbar.update(1)
            pbar.set_postfix({
                "t": f"{t:.4f}/{t_max}",
                "dt": f"{dt:.2e}",
                "ETA": f"{(t_max - t) / dt * self._elapsed:.1f}s",
                'Progress': f'{int(t / t_max * 100)}%',
                'E': f"{self.energy}",
            })

        pbar.close()
        self.output_manager.print_timings()

        if save_meta_data:
            return self.output_manager.data['observables'], self.output_manager.data['metadata']
        
        return self.output_manager.data["observables"]
        
    def step(self, hamiltonian: AbstractOperator): # TODO: might need to add t as an optional arg to feed as first arg of the stepper
        """
        Returns new parameters and new time step (if changed).
        """
        return self._stepper.step(0, self, self.psi.parameters_flat, hamiltonian=hamiltonian)

    @abstractmethod
    def solve(self, Eloc: SampledObs, gradients: SampledObs):
        '''
        Return the update and write self.update and self._additional_info
        '''
        self.update = None
        self._additional_info = None

    @abstractmethod
    def cross_validation(self, Eloc: SampledObs, gradients: SampledObs):
        pass

    @abstractmethod
    def _update_meta_data(self):
        '''
        Update the dictionary self.meta_data
        '''
        self.meta_data = {}

    def _measure_and_store(self, t, observables, save_meta_data):
        measures = {}
        energy = dict(
            mean = jnp.real(self.energy.mean).item(),
            variance = jnp.real(self.energy.var).item(),
            MC_error = jnp.real(self.energy.error_of_mean).item()
        )

        if observables is not None:
            measures = measure(observables, self.psi, self.sampler)
        self.output_manager.write_observables(t, energy=energy, **measures)

        if save_meta_data:
            self.output_manager.write_metadata(t, **self.meta_data)

class Evolution(AbstractOptimizer):
    def __init__(
            self, sampler, psi, stepper, 
            imag_time: bool, make_real: bool, use_cross_valiadation: bool=False, 
            diagonal_shift: float=1e-3, solver: AbstractSolver=PinvSNR()
        ):
        self.rhsPrefactor = 1 if imag_time else 1j
        self._lhs_trans_fn = real_fn if make_real else imag_fn
        self._rhs_trans_fn = lambda x: self._lhs_trans_fn((- self.rhsPrefactor) * x)
        self.diagonal_shift = diagonal_shift
        self._solver = solver
        
        if solver._needs_dense_matrix:
            self.get_tdvp_equation = self._get_tdvp_equation_dense
        else:
            pass # TODO: Implement this case

        super().__init__(sampler, psi, stepper, use_cross_valiadation)

        self._solver_state = SolverState(
            covar_grad_eloc=lambda: self._covar_grad_eloc,
            rhs_trans_fn=self._rhs_trans_fn,
            exact_sampler=isinstance(self.sampler, ExactSampler)
        )

    def _get_tdvp_error(self, update):
        return jnp.abs(1. + jnp.real(update.dot(self._S0.dot(update)) - 2. * jnp.real(update.dot(self._F0))) / (self.energy.var + 1e-10))
    
    def _get_tdvp_equation_dense(self, Eloc: SampledObs, gradients: SampledObs):
        '''
        Returns left and right hand side of the TDVP equation
        '''
        self._covar_grad_eloc = gradients.get_covar_obs(Eloc)
        self._F0 = - self.rhsPrefactor * self._covar_grad_eloc.mean.ravel()
        F = self._lhs_trans_fn(self._F0)

        self._S0 = gradients.get_covar()
        S = self._lhs_trans_fn(self._S0)

        if self.diagonal_shift > 1e-10:
            S = S + jnp.diag(self.diagonal_shift * jnp.diag(S))

        return S, F
    
    def solve(self, Eloc: SampledObs, gradients: SampledObs):
        self.S, F = self.get_tdvp_equation(Eloc, gradients)
        F.block_until_ready()
        self.update, self._additional_info = self._solver(self.S, F, self._solver_state)

        return self.update
    
    def cross_validation(self, Eloc, gradients):
        Eloc1 = Eloc.get_subset(start=0, step=2)
        sampleGradients1 = gradients.get_subset(start=0, step=2)
        Eloc2 = Eloc.get_subset(start=1, step=2)
        sampleGradients2 = gradients.get_subset(start=1, step=2)
        update_1, _ = self.solve(Eloc1, sampleGradients1)
        S2, F2 = self.get_tdvp_equation(Eloc2, sampleGradients2)

        validation_tdvpErr = self._get_tdvp_error(update_1)
        _, info = self.solve(Eloc, gradients)
        validation_residual = (jnp.linalg.norm(S2.dot(update_1) - F2) / jnp.linalg.norm(F2)) / info["residual"]

        crossValidationFactor_residual = validation_residual
        crossValidationFactor_tdvpErr = validation_tdvpErr / self.meta_data["tdvp_error"]
        self.meta_data["tdvp_residual_cross_validation_ratio"] = crossValidationFactor_residual
        self.meta_data["tdvp_error_cross_validation_ratio"] = crossValidationFactor_tdvpErr

        self.S, _ = self.get_tdvp_equation(self.Eloc, gradients)
    
    def _update_meta_data(self):
        self.meta_data = dict(
            tdvp_error=self._get_tdvp_error(self.update).item(), 
            **self._additional_info
        )