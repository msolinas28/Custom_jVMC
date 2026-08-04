import jax.numpy as jnp
import warnings
import numpy as np
import jax

from jVMC_exp.solver.base import SolverState, AbstractSolver

def _eigh_numpy(S):
    e, V = np.linalg.eigh(np.array(S))

    return jnp.array(e), jnp.array(V)

def smooth_cutoff_fn(x, c, exp=6):
    return 1 / (1 + (c / x)**exp)

def _diagonalize(S, F, diagonalize_on_device):
    if diagonalize_on_device:
        try:
            ev, V = jnp.linalg.eigh(S)
        except ValueError:
            warnings.warn(
                "jax.numpy.linalg.eigh raised an exception. Falling back to "
                "numpy.linalg.eigh for diagonalization.", RuntimeWarning
            )
            ev, V = _eigh_numpy(S)
    else:
        ev, V = _eigh_numpy(S)

    VtF = jnp.dot(jnp.transpose(jnp.conj(V)), F)
    return ev, V, VtF

@jax.jit(static_argnums=(2,))
def get_snr(VtF, rho_var, num_samples):
    return jnp.sqrt(jnp.abs(num_samples * (jnp.conj(VtF) * VtF) / (rho_var + 1e-14))).ravel()
    
class PinvSNR(AbstractSolver):
    """
    Pseudo-inverse solver based on an eigenvalue decomposition of the covariance
    matrix.

    The inverse is regularized using two smooth filters:

    - an eigenvalue cutoff controlled by ``pinv_cutoff``, which suppresses
      ill-conditioned directions of the covariance matrix;
    - a signal-to-noise ratio (SNR) cutoff controlled by ``snr_tol``, which
      suppresses statistically unresolved update directions.

    The effective eigenvalue cutoff is chosen adaptively such that the residual
    force discarded by the regularization is below ``pinv_tol`` whenever
    possible.

    Parameters
    ----------
    snr_tol : float, default=2
        Minimum signal-to-noise ratio of an eigenmode before it contributes
        significantly to the update.

    pinv_tol : float, default=1e-14
        Target residual. The solver decreases the eigenvalue cutoff until the
        fraction of discarded force falls below this threshold or the minimum
        cutoff ``pinv_cutoff`` is reached.

    pinv_cutoff : float, default=1e-8
        Minimum allowed relative eigenvalue cutoff.

    diagonalize_on_device : bool, default=True
        If True, diagonalize the covariance matrix using JAX. Otherwise,
        fall back to NumPy.
    """
    def __init__(self, snr_tol=2, pinv_tol=1e-14, pinv_cutoff=1e-8, diagonalize_on_device=True):
        self._snr_tol = snr_tol
        self._pinv_tol = pinv_tol
        self.pinv_cutoff = pinv_cutoff
        self._diagonalize_on_device = diagonalize_on_device

        self._ev = None
        self._V = None
    
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
    
    @property
    def _needs_dense_matrix(self) -> bool:
        return True
    
    def __call__(self, S, F, solver_state: SolverState):
        # Transform equation to eigenbasis and compute Signal to Noise Ratio
        self._transform_to_eigenbasis(S, F)
        rho = solver_state.covar_grad_o_loc().transform(
            solver_state.rhs_trans_fn, 
            jnp.transpose(jnp.conj(self.last_eigenvectors))
        )
        snr = get_snr(self._VtF, rho.var.ravel(), rho._num_samples)

        # Discard eigenvalues below numerical precision
        invEv = jnp.where(jnp.abs(self.last_eigenvalues / self.last_eigenvalues[-1]) > 1e-14, 1. / self.last_eigenvalues, 0.)
        
        residual = 1.0
        cutoff = 1e-2
        F_norm = jnp.linalg.norm(F)
        first = True 
        while (residual > self.pinv_tol and cutoff > self.pinv_cutoff) or first:
            residual, cutoff, pinvEv, effective_rank = self._regularizer_step(
                cutoff, snr, self.last_eigenvalues, invEv, self._VtF, F_norm, solver_state.exact_sampler
            )

            first = False

        update = jnp.dot(self.last_eigenvectors, (pinvEv * self._VtF))
        update = update if solver_state.holomorphic else jnp.real(update)
        info = dict(
            residual=residual.item(),
            pinv_cutoff=cutoff.item(),
            snr=snr,
            condition_number=(self.last_eigenvalues[-1] / jnp.min(jnp.abs(self.last_eigenvalues))).item(),
            spectrum=self.last_eigenvalues,
            effective_rank=effective_rank.item()
        )

        return update, info
    
    @jax.jit(static_argnums=(0, 7))
    def _regularizer_step(self, cutoff, snr, eigenvalues, invEv, VtF, F_norm, exact_sampler):
        # Set regularizer for singular value cutoff
        cutoff = jnp.max(jnp.array([0.8 * cutoff, self.pinv_cutoff]))
        regularizer = smooth_cutoff_fn(jnp.abs(eigenvalues / eigenvalues[-1]), cutoff)

        # Construct a soft cutoff based on the SNR
        if not exact_sampler:
            regularizer *= smooth_cutoff_fn(snr, self.snr_tol)

        pinvEv = invEv * regularizer
        residual = jnp.linalg.norm((pinvEv * eigenvalues - 1) * VtF) / F_norm
        effective_rank = jnp.mean(regularizer)

        return residual, cutoff, pinvEv, effective_rank
    
    def _transform_to_eigenbasis(self, S, F):
        self._ev, self._V, self._VtF = _diagonalize(S, F, self._diagonalize_on_device)


class PinvSVD(AbstractSolver):
    """
    Pseudo-inverse solver based on a smooth eigenvalue cutoff with a fixed
    threshold, following the SVD regularization of Medvidovic and Sels.

    Unlike ``PinvSNR``, the cutoff is not searched adaptively: it is fixed by
    ``rcond`` (relative to the largest eigenvalue) and ``acond`` (absolute),
    and applied once, directly to the raw eigenspectrum. There is no
    SNR-based gating.

    Parameters
    ----------
    rcond : float, default=1e-14
        Relative eigenvalue cutoff, as a fraction of the largest eigenvalue.

    acond : float, default=1e-8
        Absolute eigenvalue cutoff.

    exponent : float, default=6
        Sharpness of the smooth cutoff transition.

    diagonalize_on_device : bool, default=True
        If True, diagonalize the covariance matrix using JAX. Otherwise,
        fall back to NumPy.
    """
    def __init__(self, rcond=1e-14, acond=1e-8, exponent=6, diagonalize_on_device=True):
        self.rcond = rcond
        self.acond = acond
        self.exponent = exponent
        self._diagonalize_on_device = diagonalize_on_device

        self._ev = None
        self._V = None

    @property
    def last_eigenvalues(self):
        return self._ev

    @property
    def last_eigenvectors(self):
        return self._V

    @property
    def _needs_dense_matrix(self) -> bool:
        return True

    def __call__(self, S, F, solver_state: SolverState):
        self._ev, self._V, self._VtF = _diagonalize(S, F, self._diagonalize_on_device)
        F_norm = jnp.linalg.norm(F)

        residual, pinvEv, effective_rank, cutoff = self._regularize(
            self.last_eigenvalues, self._VtF, F_norm
        )

        update = jnp.dot(self.last_eigenvectors, pinvEv * self._VtF)
        update = update if solver_state.holomorphic else jnp.real(update)
        info = dict(
            residual=residual.item(),
            cutoff=cutoff.item(),
            condition_number=(self.last_eigenvalues[-1] / jnp.min(jnp.abs(self.last_eigenvalues))).item(),
            spectrum=self.last_eigenvalues,
            effective_rank=effective_rank.item()
        )

        return update, info

    @jax.jit(static_argnums=(0,))
    def _regularize(self, eigenvalues, VtF, F_norm):
        eps = jnp.finfo(eigenvalues.dtype).eps
        cutoff = jnp.max(jnp.array([eps, self.acond, self.rcond * eigenvalues[-1]]))

        ev_safe = jnp.maximum(eigenvalues, 10 * eps)
        regularizer = smooth_cutoff_fn(ev_safe, cutoff, self.exponent)

        pinvEv = regularizer / ev_safe
        residual = jnp.linalg.norm((pinvEv * eigenvalues - 1) * VtF) / F_norm
        effective_rank = jnp.mean(regularizer)

        return residual, pinvEv, effective_rank, cutoff