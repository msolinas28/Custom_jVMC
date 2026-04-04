from jVMC_exp.solver.base import AbstractSolver
import jax.numpy as jnp
import warnings
import numpy as np

from jVMC_exp.solver.base import SolverState

def _eigh_numpy(S):
    e, V = np.linalg.eigh(np.array(S))

    return jnp.array(e), jnp.array(V)

class PinvSNR(AbstractSolver):
    def __init__(self, snr_tol=2, pinv_tol=1e-14, pinv_cutoff=1e-8, diagonalize_on_device=True):
        self._snr_tol = snr_tol
        self._pinv_tol = pinv_tol
        self.pinv_cutoff = pinv_cutoff
        self._diagonalize_on_device = diagonalize_on_device

        self._ev = None
        self._V = None

    @property
    def _needs_dense_matrix(self) -> bool:
        return True
    
    @property
    def snr_tol(self):
        return self._snr_tol
    
    @property
    def pinv_tol(self):
        return self._pinv_tol
    
    @property
    def last_eigenvalues(self):
        return self._ev
    
    @property
    def last_eigenvectors(self):
        return self._V
    
    def __call__(self, S, F, solver_state: SolverState):
        # Transform equation to eigenbasis and compute Signal to Noise Ratio
        self._transform_to_eigenbasis(S, F) 
        snr = self._get_snr(solver_state)

        # Discard eigenvalues below numerical precision
        invEv = jnp.where(jnp.abs(self._ev / self._ev[-1]) > 1e-14, 1. / self._ev, 0.)

        residual = 1.0
        cutoff = 1e-2
        F_norm = jnp.linalg.norm(F)
        while residual > self.pinv_tol and cutoff > self.pinv_cutoff:
            cutoff *= 0.8
            # Set regularizer for singular value cutoff
            regularizer = 1. / (1. + (max(cutoff, self.pinv_cutoff) / jnp.abs(self._ev / self._ev[-1]))**6)

            if not solver_state.exact_sampler:
                # Construct a soft cutoff based on the SNR
                regularizer *= 1. / (1. + (self._snr_tol / snr)**6)

            pinvEv = invEv * regularizer

            residual = jnp.linalg.norm((pinvEv * self._ev - jnp.ones_like(pinvEv)) * self._VtF) / F_norm
            update = jnp.real(jnp.dot(self._V, (pinvEv * self._VtF)))

            info = dict(
                residual=residual,
                pinv_cutoff=max(cutoff, self.pinv_cutoff),
                snr=snr,
                spectrum=self.last_eigenvalues
            )

        return update, info
    
    def _transform_to_eigenbasis(self, S, F):
        if self._diagonalize_on_device:
            try:
                self._ev, self._V = jnp.linalg.eigh(S)
            except ValueError:
                warnings.warn(
                    "jax.numpy.linalg.eigh raised an exception. Falling back to " 
                    "numpy.linalg.eigh for diagonalization.", RuntimeWarning
                )
            
                self._ev, self._V = _eigh_numpy(S)
        else:
            self._ev, self._V = _eigh_numpy(S)

        self._VtF = jnp.dot(jnp.transpose(jnp.conj(self._V)), F)

    def _get_snr(self, solver_state: SolverState):
        rho = solver_state.covar_grad_eloc().transform(solver_state.rhs_trans_fn, jnp.transpose(self.last_eigenvectors))
        rho_var = rho.var.ravel()

        return jnp.sqrt(jnp.abs(rho._num_samples * (jnp.conj(self._VtF) * self._VtF) / (rho_var + 1e-10))).ravel()