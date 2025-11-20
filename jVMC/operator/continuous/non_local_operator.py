import jax.numpy as jnp
import jax

from jVMC.operator.continuous.base import Operator
from jVMC.geometry import AbstractGeometry

def laplacian(grad_f):
    def lap(x):
        hessian_times_v = lambda v: jax.jvp(jax.grad(grad_f), (x,), (v,))[1]
        basis_vectors = jnp.eye(len(x))
        # Vectorize across all dimensions of x
        diag_hess = jax.vmap(hessian_times_v)(basis_vectors)
        return jnp.sum(jnp.diagonal(diag_hess))
    return lap

class TotalKineticOperator(Operator):
    def __init__(self, geometry: AbstractGeometry, mass: float | list = 1):
        super().__init__(geometry, is_multiplicative=False)

        if isinstance(mass, complex):
            raise ValueError("The property 'mass' can not be complex.")
        elif isinstance(mass, (int, float)):
            self._mass = jnp.asarray(mass)
        elif isinstance(mass, (list, tuple)) or hasattr(mass, '__len__'):
            if len(mass) != geometry.n_particles:
                raise ValueError(f"Got {len(mass)} masses which is not the same as the number of particles ({geometry.n_particles}).")
            self._mass = jnp.asarray(mass)
        else:
            raise ValueError(f"'mass' must be a scalar or array-like, got {type(mass)}.")
            
    @property
    def mass(self):
        return self._mass
    
    def local_value(self, s, apply_fun, parameters):
        log_psi = lambda x: apply_fun(parameters, x)
        grad_log_psi = jax.grad(log_psi) 
        laplacian_logpsi = self.mass * (laplacian(grad_log_psi)(s) + jnp.abs(grad_log_psi(s)) ** 2) 

        return - 0.5 * jnp.real(laplacian_logpsi)