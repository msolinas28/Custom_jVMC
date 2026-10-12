from abc import ABC, abstractmethod
from chex import dataclass
import jax
import jax.numpy as jnp

from jVMC_exp.stats import SampledObs, LazySampledObs
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.sampler.base import AbstractSampler
from jVMC_exp.sharding_config import sharded, MESH, REPLICATED_SPEC, pad_to_devices
from jVMC_exp.util.grads import pick_gradient

@dataclass
class ObjectiveFunctionOutput():
    o_loc: SampledObs | None = None
    grad_log_psi: SampledObs | LazySampledObs | None = None
    grad: jax.Array | None = None

class AbstractObjectiveFunction(ABC): # TODO: make batched jacobian a flag in the abstract init
    @abstractmethod
    def __call__(self, sampler: AbstractSampler, **kwargs) -> SampledObs:
        pass

    @abstractmethod
    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool, **kwargs) -> ObjectiveFunctionOutput:
        pass

class Observable(AbstractObjectiveFunction):
    def __init__(self, operator: AbstractOperator, batched_jacobian: bool = False):
        self._operator = operator
        self._batched_jacobian = batched_jacobian
        # TODO: might add a batch size here so that one can have different batch sizes

    @property
    def operator(self):
        return self._operator
    
    def __call__(self, sampler: AbstractSampler, **op_kwargs):
        return sampler(self.operator, **op_kwargs)
    
    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool = True, **op_kwargs):
        o_loc = self(sampler, **op_kwargs)
        if self._batched_jacobian:
            grad_log_psi = LazySampledObs(
                sampler.psi.lazy_gradients(sampler.samples), sampler.weights
            )
        else:
            grad_log_psi = SampledObs(
                sampler.psi.gradients(sampler.samples), sampler.weights
            )

        if compute_grad:
            if self._batched_jacobian:
                # Same as grad_log_psi.get_covar(o_loc), sum_s w_s conj(O_s) (E_s - <E>), but from
                # a VJP, which does not compute the batches of the Jacobian once more.
                # o_loc is already padded to a multiple of the number of devices.
                cotangents = jnp.sqrt(o_loc.weights) * o_loc._normalized_obs[:, 0]
                grad = jnp.conj(
                    self._lazy_grad_fn_sh(
                        pad_to_devices(sampler.samples), jnp.conj(cotangents),
                        parameters=sampler.psi.grad_parameters, psi=sampler.psi,
                        chunk_size=sampler.psi.batchSize // MESH.size, batch_size=None
                    )
                )
            else:
                grad = grad_log_psi.get_covar(o_loc)

            return ObjectiveFunctionOutput(
                o_loc=o_loc, grad=grad, grad_log_psi=grad_log_psi
            )

        return ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad_log_psi)

    @sharded(use_vmap=False, out_specs=REPLICATED_SPEC, static_kwarg_names=("psi", "chunk_size"))
    def _lazy_grad_fn_sh(self, samples, cotangents, *, parameters, psi, chunk_size, batch_size):
        """
        Returns ``sum_s cotangents[s] * O_s``, where ``O_s`` is the gradient of ``log psi(s)`` in
        the convention of ``psi.flat_gradient_function``, for ``samples`` and ``cotangents``
        sharded across devices. The result is replicated on all devices.

        All conventions of ``psi.flat_gradient_function`` are linear in ``log psi`` over the
        complex numbers, so the sum is the gradient of ``sum_s cotangents[s] * log psi(s)``:
        every device takes one backward pass per chunk of ``chunk_size`` of its samples,
        without computing the gradient of every single sample.
        """
        n_local = samples.shape[0]
        chunk_size = min(chunk_size, n_local)
        n_chunks = -(-n_local // chunk_size)
        # Pad to a multiple of the chunk size with copies of the first sample, with zero cotangent
        num_pad = n_chunks * chunk_size - n_local
        if num_pad:
            samples = jnp.concatenate([samples, jnp.repeat(samples[:1], num_pad, axis=0)])
            cotangents = jnp.pad(cotangents, (0, num_pad))
        samples = samples.reshape((n_chunks, chunk_size) + samples.shape[1:])
        cotangents = cotangents.reshape(n_chunks, chunk_size)

        # The gradient with respect to parameters replicated on all devices would be summed over
        # the devices for every chunk. Instead, every device sums over its chunks first.
        parameters = jax.lax.pcast(parameters, ("devices",), to="varying")
        log_psi = jax.vmap(psi.apply_fun, in_axes=(None, 0))
        weighted_log_psi = lambda p, chunk: jnp.sum(chunk[1] * log_psi(p, chunk[0]))
        chunk_grad = lambda i: psi.flat_gradient_function(
            weighted_log_psi, parameters, (samples[i], cotangents[i])
        )

        grad_shape = jax.eval_shape(chunk_grad, 0)
        # The sum within a chunk has the precision of the network, the sum over the chunks
        # at least the precision of the cotangents
        dtype = jnp.result_type(grad_shape.dtype, cotangents.dtype)
        grad = jnp.zeros(grad_shape.shape, dtype)
        # The zeros are the same on every device, the loop makes them device dependent
        grad = jax.lax.pcast(grad, ("devices",), to="varying")
        grad = jax.lax.fori_loop(0, n_chunks, lambda i, g: g + chunk_grad(i).astype(dtype), grad)

        return jax.lax.psum(grad, "devices")

class Estimator(AbstractObjectiveFunction):
    def __init__(self, estimator_fn: callable, batched_jacobian: bool = False):
        self._estimator_fn = estimator_fn
        self._is_grad_init = False
        self._batched_jacobian = batched_jacobian
        
    @property
    def estimator_fn(self):
        return self._estimator_fn
    
    def __call__(self, sampler: AbstractSampler):
        observations = self.estimator_fn(sampler.psi.parameters, sampler.samples)

        return SampledObs(observations, sampler.weights)
    
    def value_and_grad(self, sampler: AbstractSampler, **kwargs):
        if not self._is_grad_init:
            _, _, self._grad_fn, _ = pick_gradient(
                self.estimator_fn, sampler.psi.parameters, sampler.samples[0]
            )
            self._is_grad_init = True

        value = self(sampler)
        if self._batched_jacobian:
            grad_obs = LazySampledObs(
                self._lazy_grad_fn_sh(
                    sampler.samples, 
                    parameters=sampler.psi.parameters, 
                    batch_size=sampler.psi.batchSize
                ), 
                sampler.weights
            )
        else:
            grad_obs = SampledObs(
                self._grad_fn_sh(
                    sampler.samples, 
                    parameters=sampler.psi.parameters, 
                    batch_size=sampler.psi.batchSize
                ), 
                sampler.weights
            )

        return ObjectiveFunctionOutput(o_loc=value, grad=grad_obs.mean)

    @sharded(automatic_sharding=True) # TODO: Set flag to False once jax problem is solved
    def _grad_fn_sh(self, samples, *, parameters, batch_size):
        return self._grad_fn(self.estimator_fn, parameters, samples)
    
    @sharded(automatic_sharding=True, yield_iter=True) # TODO: Set flag to False once jax problem is solved
    def _lazy_grad_fn_sh(self, samples, *, parameters, batch_size):
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
        self._estimator = Estimator(estimator_fn, batched_jacobian)

    def __call__(self, sampler: AbstractSampler, **op_kwargs):
        return self._observable(sampler, **op_kwargs)

    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool = True, **op_kwargs):
        obs_out = self._observable.value_and_grad(sampler, compute_grad, **op_kwargs)
        o_loc = obs_out.o_loc
        grad_log_psi = obs_out.grad_log_psi

        if compute_grad:
            grad = obs_out.grad + self._estimator.value_and_grad(sampler, compute_grad=compute_grad).grad

            return ObjectiveFunctionOutput(o_loc=o_loc, grad=grad, grad_log_psi=grad_log_psi)
        
        return ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad_log_psi)