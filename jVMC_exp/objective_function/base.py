from abc import ABC, abstractmethod
from chex import dataclass
import jax

from jVMC_exp.stats import SampledObs, LazySampledObs
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.sampler import AbstractSampler
from jVMC_exp.sharding_config import sharded
from jVMC_exp.util.grads import pick_gradient

@dataclass
class ObjectiveFunctionOutput():
    o_loc: SampledObs | None = None
    grad_log_psi: SampledObs | LazySampledObs | None = None
    grad: jax.Array | None = None
    
    # @property
    # def sync_target(self):
    #     for obj in (self.grad, self.grad_log_psi, self.o_loc):
    #         if obj is None:
    #             continue
    #         if hasattr(obj, "sync_target"):
    #             return obj.sync_target
    #         if hasattr(obj, "observations"):
    #             return obj.observations
    #     return None
    
    # def transform(self, element_wise_fn=lambda x: x, linear_map=None):
    #     return ObjectiveFunctionOutput(
    #         o_loc=self.o_loc.transform(element_wise_fn, linear_map) if self.o_loc is not None else None,
    #         grad=self.grad.transform(element_wise_fn, linear_map) if self.grad is not None else None,
    #         grad_log_psi=self.grad_log_psi.transform(element_wise_fn, linear_map) if self.grad_log_psi is not None else None
    #     )

class AbstractObjectiveFunction(ABC):
    @abstractmethod
    def __call__(self, sampler: AbstractSampler, **kwargs) -> SampledObs:
        pass

    @abstractmethod
    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool, **kwargs) -> ObjectiveFunctionOutput:
        pass

class Observable(AbstractObjectiveFunction):
    def __init__(self, operator: AbstractOperator, batched_jacobian: bool):
        self._operator = operator
        self._batched_jacobian = batched_jacobian

    @property
    def operator(self):
        return self._operator
    
    def __call__(self, sampler: AbstractSampler, **op_kwargs):
        return sampler(self.operator, **op_kwargs)
    
    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool = True, **op_kwargs):
        o_loc = self(sampler, **op_kwargs)
        if self._batched_jacobian:
            grad_log_psi = LazySampledObs(sampler.psi.lazy_gradients(sampler.samples), sampler.weights)
        else:
            grad_log_psi = SampledObs(sampler.psi.gradients(sampler.samples), sampler.weights)

        if compute_grad:
            grad = grad_log_psi.get_covar(o_loc)
            return ObjectiveFunctionOutput(o_loc=o_loc, grad=grad, grad_log_psi=grad_log_psi)

        return ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad_log_psi)

class Estimator(AbstractObjectiveFunction):
    def __init__(self, estimator_fn: callable):
        self._estimator_fn = estimator_fn
        self._is_grad_init = False
        
    @property
    def estimator_fn(self):
        return self._estimator_fn
    
    def __call__(self, sampler: AbstractSampler):
        observations = self.estimator_fn(sampler.psi.parameters, sampler.samples)

        return SampledObs(observations, sampler.weights)
    
    def value_and_grad(self, sampler: AbstractSampler, **kwargs):
        if not self._is_grad_init:
            _, _, self._grad_fn, _ = pick_gradient(self.estimator_fn, sampler.psi.parameters, sampler.samples[0])
            self._is_grad_init = True

        value = self(sampler)
        grad = SampledObs(self._get_estimator_grad(
            sampler.samples, 
            parameters=sampler.psi.parameters, 
            batch_size=sampler.psi.batchSize
        ), sampler.weights)

        return ObjectiveFunctionOutput(o_loc=value, grad=grad)

    @sharded(automatic_sharding=True) # TODO: Set flag to False once jax problem is solved
    def _get_estimator_grad(self, samples, *, parameters, batch_size):
        return self._grad_fn(self.estimator_fn, parameters, samples)
    
class ParametricObservable(AbstractObjectiveFunction):
    """
    Objective function for an observable O(θ, s) that depends explicitly on
    both the network parameters θ and the configuration s.
    """
    def __init__(
            self, 
            operator: AbstractOperator, 
            estimator_fn: callable, 
            batched_jacobian: bool = False
        ):
        self._observable = Observable(operator, batched_jacobian)
        self._estimator = Estimator(estimator_fn)

    def __call__(self, sampler: AbstractSampler, **op_kwargs):
        return self._observable(sampler, **op_kwargs)

    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool = True, **op_kwargs):
        o_loc = self(sampler, **op_kwargs)
        grad_log_psi = SampledObs(sampler.psi.gradients(sampler.samples), sampler.weights)

        if not compute_grad:
            return ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad_log_psi)

        grad = grad_log_psi.get_covar(o_loc).ravel()
        grad = grad + self._estimator.value_and_grad(sampler, compute_grad=compute_grad).grad.observations
        grad = SampledObs(grad, sampler.weights)

        return ObjectiveFunctionOutput(o_loc=o_loc, grad=grad, grad_log_psi=grad_log_psi)