import math
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
from typing import Callable
from functools import partial

from jVMC_exp.optimizer.base import AbstractOptimizer
from jVMC_exp.vqs import NQS
from jVMC_exp.stats import LazySampledObs
from jVMC_exp.sampler import ExactSampler
from jVMC_exp.sampler.base import AbstractSampler
from jVMC_exp.objective_function.base import AbstractObjectiveFunction, ObjectiveFunctionOutput
from jVMC_exp.util import OutputManager
from jVMC_exp.util import make_real_array, remove_double
from jVMC_exp.util.key_gen import format_key

def _centered_block(O, mean, sqrt_w, idx):
    """
    Columns ``idx`` of the normalized log-derivatives, :math:`\\sqrt{w_n}(O_{n,i}-m_i)`. \
    Indices beyond the last parameter (the padding of the partition) give zero columns.
    """
    O_k = jnp.take(O, idx, axis=1, mode="fill", fill_value=0)
    mean_k = jnp.take(mean, idx, mode="fill", fill_value=0)

    return sqrt_w[:, None] * (O_k - mean_k)

@jax.jit
def _split_weights(weights, key, holdout_fraction):
    """Weights of the training and of the holdout samples, each normalized on its own subset."""
    hold = jax.random.uniform(key, weights.shape) < holdout_fraction
    w_train = jnp.where(hold, 0., weights)
    w_hold = jnp.where(hold, weights, 0.)

    return w_train / jnp.sum(w_train), w_hold / jnp.maximum(jnp.sum(w_hold), 1e-300)

@jax.jit
def _means(O, eps, w):
    """Means of the log-derivatives and of the local losses for each row of weights ``w``."""
    return w @ O, w @ eps

@jax.jit
def _apply(O, mean, sqrt_w, v):
    """:math:`\\sqrt{w_n}\\sum_i(O_{n,i}-m_i)v_i`, the normalized log-derivatives times ``v``."""
    return sqrt_w * (O @ v - mean @ v)

@jax.jit
def _qgt_matvec(O, mean, sqrt_w, v):
    """:math:`Sv=A^\\dagger(Av)`, with the normalized log-derivatives :math:`A`, without forming S."""
    x = sqrt_w * _apply(O, mean, sqrt_w, v)
    # conj(x) @ O instead of conj(O).T @ x, so that no conjugate copy of O is made
    return jnp.conj(jnp.conj(x) @ O) - jnp.conj(mean) * jnp.sum(x)

@partial(jax.jit, static_argnames=("real_params",))
def _block_cholesky(O, mean, sqrt_w, blocks, shift, scale, real_params):
    """
    Cholesky factors of :math:`S_{kk}+\\mathrm{shift}+\\mathrm{scale}\\,\\mathrm{diag}(S_{kk})` \
    for every block. Padding entries get a one on the diagonal, so that their factor is regular \
    and their update is zero.
    """
    n_params = O.shape[1]

    def factor(idx):
        A_k = _centered_block(O, mean, sqrt_w, idx)
        S_kk = A_k.conj().T @ A_k
        if real_params:
            S_kk = jnp.real(S_kk)
        diag = jnp.real(jnp.diagonal(S_kk))
        pad = (idx >= n_params).astype(diag.dtype)

        return jnp.linalg.cholesky(S_kk + jnp.diag(shift + scale * diag + pad))

    return jax.lax.map(factor, blocks)

@partial(jax.jit, static_argnames=("real_params",))
def _sweep(O, mean, sqrt_w, blocks, chol, order, thetadot, r, value_penalty, real_params):
    """
    One Gauss-Seidel sweep over the blocks in the order ``order``: for each block \
    :math:`(S_{kk}+\\mu)\\delta_k=A_k^\\dagger r-\\lambda_0\\dot\\theta_k`, then \
    :math:`\\dot\\theta_k\\to\\dot\\theta_k+\\delta_k` and :math:`r\\to r-A_k\\delta_k`.
    """
    def block_step(carry, k):
        thetadot, r = carry
        idx = blocks[k]
        A_k = _centered_block(O, mean, sqrt_w, idx)
        g = A_k.conj().T @ r
        if real_params:
            g = jnp.real(g)
        g = g - value_penalty * thetadot[idx]
        delta = cho_solve((chol[k], True), g)

        return (thetadot.at[idx].add(delta), r - A_k @ delta), None

    (thetadot, r), _ = jax.lax.scan(block_step, (thetadot, r), order)

    return thetadot, r

@jax.jit
def _relative_residual(r, e):
    return jnp.sum(jnp.abs(r) ** 2) / jnp.sum(jnp.abs(e) ** 2)

class BGSTDVP(AbstractOptimizer):
    """
    Real-time evolution with the time-dependent variational principle, solved by block \
    Gauss-Seidel sweeps instead of a solve with the full :math:`S`, see ``bgs_tdvp.md``.

    The update is the least-squares solution of :math:`O\\dot\\theta\\approx e`, with the \
    normalized log-derivatives :math:`O_{n,i}=\\sqrt{w_n}(\\Gamma_i(s_n)-\\langle\\Gamma_i\\rangle)` \
    and :math:`e_n=-i\\sqrt{w_n}(\\ell(s_n)-\\langle\\ell\\rangle)`, whose normal equation is \
    :math:`S\\dot\\theta=-iF`. For a holomorphic network the parameters are complex; otherwise \
    :math:`\\dot\\theta` is real and the normal equation is \
    :math:`\\mathrm{Re}(S)\\dot\\theta=\\mathrm{Im}(F)`, as ``TDVP`` with ``make_real=True``.

    The parameters are split into ``n_blocks`` random blocks of equal size, and each sweep \
    visits the blocks in a random order, solving

    .. math::

        (S_{kk}+\\mu_k+\\lambda_0)\\,\\delta_k=A_k^\\dagger r-\\lambda_0\\dot\\theta_k,\\qquad
        r\\to r-A_k\\delta_k,

    with the residual :math:`r=e-O\\dot\\theta` and the proximal damping \
    :math:`\\mu_k=\\mathrm{diagonalShift}+\\mathrm{diagonalScale}\\,\\mathrm{diag}(S_{kk})`, which \
    changes the convergence rate but not the converged solution. :math:`\\lambda_0` \
    (``value_penalty``) is a Tikhonov shift, which does change it: :math:`(S+\\lambda_0)\\dot\\theta=-iF`.

    With ``holdout_fraction > 0`` a random fraction of the samples is left out of the sweeps, \
    and the sweeps stop when the relative residual :math:`\\|e-O\\dot\\theta\\|^2/\\|e\\|^2` on \
    these samples improves by less than a fraction ``tol`` of its best value; the update with \
    the best holdout residual is returned. Without a holdout, the training residual is used in \
    the same way. At most ``max_sweeps`` sweeps are done.

    The first evaluation of a step of the stepper (``intStep=0``) draws the partition, the \
    block orders of the sweeps and the holdout, takes the update of the previous evaluation as \
    warm start (zero at the first one) and chooses the number of sweeps with the criterion \
    above. The further evaluations of the step (``intStep >= 1``) reuse all of them, the warm \
    start included, so that within a step the update is one continuous function of the \
    parameters. A call without ``intStep`` starts a new step.

    With ``store_qgt``, ``qgt`` is the function :math:`v\\mapsto Sv` of the last evaluation, \
    computed from the log-derivatives without forming :math:`S`, so that \
    ``jVMC_exp.util.s_norm_fn(opt)`` gives the norm :math:`\\|v\\|_S=\\|Ov\\|` for the adaptive \
    steppers.

    The right hand side is built from the local losses, assuming \
    :math:`F=\\mathrm{Cov}(\\Gamma^*,\\ell)`, as for ``Observable``: the force of the objective \
    function is not computed.

    ``meta_data`` holds ``n_sweeps``, ``tdvp_error``, the relative residual on the training \
    samples, and ``holdout_tdvp_error`` if a holdout is used.

    Args:
        * ``sampler``: Sampler of ``psi``.
        * ``psi``: Variational state.
        * ``n_blocks``: Number of blocks :math:`K` of the partition.
        * ``max_sweeps``: Maximum number of sweeps per step.
        * ``tol``: Relative improvement of the residual below which the sweeps stop.
        * ``holdout_fraction``: Fraction of the samples used for the stopping criterion only. \
          It must be zero for an ``ExactSampler``.
        * ``diagonalShift``: Constant proximal damping, or a callable ``step -> value``.
        * ``diagonalScale``: Proximal damping relative to the diagonal of :math:`S_{kk}`, or a \
          callable ``step -> value``.
        * ``value_penalty``: Tikhonov shift :math:`\\lambda_0`, which keeps the components of \
          the warm start in the kernel of :math:`S` from accumulating.
        * ``seed`` [optional]: Seed of the partitions, of the block orders and of the holdout.
        * ``store_qgt``: Whether to keep the log-derivatives of the last evaluation for \
          ``qgt``, until the next evaluation.
        * ``resample_stepper``: See ``AbstractOptimizer``.
        * ``output_manager`` [optional]: See ``AbstractOptimizer``.
    """
    def __init__(
            self, sampler: AbstractSampler, psi: NQS, n_blocks: int,
            max_sweeps: int = 20, tol: float = 1e-3, holdout_fraction: float = 0.1,
            diagonalShift: float | Callable = 1e-3, diagonalScale: float | Callable = 0.,
            value_penalty: float = 0., seed=None, store_qgt: bool = True,
            resample_stepper: bool = True, output_manager: OutputManager | None = None
        ):
        if not 0 <= holdout_fraction < 1:
            raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction}.")
        if holdout_fraction > 0 and isinstance(sampler, ExactSampler):
            raise ValueError(
                "A holdout leaves configurations out of the exact sum of the ExactSampler: "
                "use holdout_fraction=0."
            )
        if max_sweeps < 1:
            raise ValueError(f"max_sweeps must be at least 1, got {max_sweeps}.")

        super().__init__(
            sampler, psi, resample_stepper, False, output_manager=output_manager
        )

        self.key = seed
        self.diag_shift = diagonalShift
        self.diag_scale = diagonalScale
        self.value_penalty = value_penalty
        self.max_sweeps = max_sweeps
        self.tol = tol
        self.holdout_fraction = holdout_fraction
        self.store_qgt = store_qgt
        # qgt is a matvec, which has no padding (read by jVMC_exp.util.s_norm_fn)
        self._params_pad_size = 0

        self._real_params = not psi.holomorphic
        self._make_real_fn = jax.jit(partial(make_real_array, params_shape=psi.paramShapes))
        self._remove_double_trans = jax.vmap(partial(remove_double, params_shape=psi.paramShapes))

        double_params = (not psi.realParams) and (not psi.holomorphic)
        self._num_params = psi.numParameters * (2 if double_params else 1)
        if not 1 <= n_blocks <= self._num_params:
            raise ValueError(
                f"n_blocks must be between 1 and the number of parameters, "
                f"{self._num_params}, got {n_blocks}."
            )
        self._n_blocks = n_blocks
        self._block_size = math.ceil(self._num_params / n_blocks)

        self._thetadot = None
        self._step = None
        self._new_step = True
        self._metric = None
        self._additional_info = {}

    @property
    def key(self):
        """PRNG key of the next step: partition, block orders and holdout."""
        return self._key

    @key.setter
    def key(self, value):
        self._key = format_key(value)

    @property
    def n_blocks(self):
        """Number of blocks of the partition."""
        return self._n_blocks

    @property
    def thetadot(self):
        """
        Update of the last evaluation, the warm start of the next step: complex for a \
        holomorphic network, real otherwise. ``None`` before the first evaluation or after \
        ``reset()``.
        """
        return self._thetadot

    @property
    def qgt(self):
        """
        The function :math:`v\\mapsto Sv` of the last evaluation, for ``v`` complex for a \
        holomorphic network and real otherwise, if ``store_qgt``. ``None`` otherwise.
        """
        if self._metric is None:
            return None
        return partial(_qgt_matvec, *self._metric)

    def reset(self):
        """Forget the update of the last evaluation, so that the next step starts from zero."""
        self._thetadot = None
        self._step = None

    @property
    def diag_scale(self):
        """
        Proximal damping relative to the diagonal of :math:`S_{kk}` at the current step. It can \
        be set to a number or to a callable ``step -> value``, evaluated by ``update_hyperparams()``.
        """
        return self._diag_scale

    @diag_scale.setter
    def diag_scale(self, value):
        self._diag_scale_fn = value if isinstance(value, Callable) else lambda step: value
        self._diag_scale = self._diag_scale_fn(0)

    @property
    def diag_shift(self):
        """
        Constant proximal damping at the current step. It can be set to a number or to a \
        callable ``step -> value``, evaluated by ``update_hyperparams()``.
        """
        return self._diag_shift

    @diag_shift.setter
    def diag_shift(self, value):
        self._diag_shift_fn = value if isinstance(value, Callable) else lambda step: value
        self._diag_shift = self._diag_shift_fn(0)

    @property
    def _needs_grad(self):
        return False

    def update_hyperparams(self, step):
        """Evaluate ``diag_shift`` and ``diag_scale`` for the step ``step``."""
        self._diag_shift = self._diag_shift_fn(step)
        self._diag_scale = self._diag_scale_fn(step)

    def __call__(self, parameters, t, *, intStep=None, **kwargs):
        """
        See ``AbstractOptimizer.__call__()``. ``intStep=0`` or ``None`` starts a new step, see \
        the class description.

        :meta public:
        """
        # Release the log-derivatives of the previous evaluation before computing new ones
        self._metric = None
        self._new_step = intStep is None or intStep == 0 or self._step is None

        return super().__call__(parameters, t, intStep=intStep, **kwargs)

    def _new_step_state(self):
        """Partition, keys of the holdout and of the block orders, and warm start of a step."""
        key_blocks, key_hold, key_order, self._key = jax.random.split(self._key, 4)
        n_padded = self._n_blocks * self._block_size
        blocks = jax.random.permutation(
            key_blocks, jnp.arange(n_padded)
        ).reshape((self._n_blocks, self._block_size))

        return dict(
            blocks=blocks, key_hold=key_hold, key_order=key_order,
            init=self._thetadot, n_sweeps=None
        )

    def get_update(self, objective_function_output: ObjectiveFunctionOutput):
        """
        Block Gauss-Seidel solution of the TDVP equation, see the class description.

        Args:
            * ``objective_function_output``: ``ObjectiveFunctionOutput``, without the force.

        Returns:
            The update, a real array in the layout of ``psi.parameters_flat``.
        """
        grad_log_psi = objective_function_output.grad_log_psi
        if isinstance(grad_log_psi, LazySampledObs):
            raise NotImplementedError(
                "BGSTDVP needs the materialized log-derivatives: "
                "use an objective function with batched_jacobian=False."
            )
        if self.psi.holomorphic:
            grad_log_psi.transform(self._remove_double_trans)

        if self._new_step:
            self._step = self._new_step_state()
        step = self._step

        O = grad_log_psi.observations
        eps = objective_function_output.o_loc.observations[:, 0]
        weights = grad_log_psi.weights
        holdout = self.holdout_fraction > 0

        if holdout:
            w_train, w_hold = _split_weights(weights, step["key_hold"], self.holdout_fraction)
        else:
            w_train, w_hold = weights, weights
        means, eps_means = _means(O, eps, jnp.stack([w_train, w_hold, weights]))
        sqrt_w_train, sqrt_w_hold = jnp.sqrt(w_train), jnp.sqrt(w_hold)
        e_train = -1j * sqrt_w_train * (eps - eps_means[0])
        if self.store_qgt:
            self._metric = (O, means[2], jnp.sqrt(weights))

        chol = _block_cholesky(
            O, means[0], sqrt_w_train, step["blocks"],
            self.diag_shift + self.value_penalty, self.diag_scale, self._real_params
        )

        n_padded = self._n_blocks * self._block_size
        if step["init"] is None:
            dtype = jnp.finfo(O.dtype).dtype if self._real_params else O.dtype
            thetadot = jnp.zeros(n_padded, dtype)
            r = e_train
        else:
            thetadot = jnp.pad(step["init"], (0, n_padded - self._num_params))
            r = e_train - _apply(O, means[0], sqrt_w_train, step["init"])

        def run_sweep(n, thetadot, r):
            order = jax.random.permutation(jax.random.fold_in(step["key_order"], n), self._n_blocks)
            return _sweep(
                O, means[0], sqrt_w_train, step["blocks"], chol, order, thetadot, r,
                self.value_penalty, self._real_params
            )

        if step["n_sweeps"] is None:
            # First evaluation of the step: the number of sweeps follows from the criterion
            e_hold = -1j * sqrt_w_hold * (eps - eps_means[1])
            best = None
            for n in range(self.max_sweeps):
                thetadot, r = run_sweep(n, thetadot, r)
                res_train = _relative_residual(r, e_train).item()
                if holdout:
                    r_hold = e_hold - _apply(O, means[1], sqrt_w_hold, thetadot[:self._num_params])
                    res = _relative_residual(r_hold, e_hold).item()
                else:
                    res = res_train

                if not math.isfinite(res):
                    raise FloatingPointError(
                        "The block solve gave a non-finite update: some S_kk + damping is not "
                        "positive definite. Increase diagonalShift."
                    )
                if best is not None and res >= best["res"]:
                    break
                converged = best is not None and res > best["res"] * (1 - self.tol)
                best = dict(res=res, res_train=res_train, thetadot=thetadot, n_sweeps=n + 1)
                if converged:
                    break

            thetadot = best["thetadot"][:self._num_params]
            step["n_sweeps"] = best["n_sweeps"]
            self._additional_info = dict(n_sweeps=best["n_sweeps"], tdvp_error=best["res_train"])
            if holdout:
                self._additional_info["holdout_tdvp_error"] = best["res"]
        else:
            # Further evaluations of the step: the same sweeps as the first one
            for n in range(step["n_sweeps"]):
                thetadot, r = run_sweep(n, thetadot, r)
            thetadot = thetadot[:self._num_params]

        self._thetadot = thetadot
        self.update = thetadot if self._real_params else self._make_real_fn(thetadot)

        return self.update

    def cross_validation(self, objective_function: AbstractObjectiveFunction, **objective_function_kwargs):
        """Does nothing: the holdout of the stopping criterion plays its role."""
        pass

    def _update_meta_data(self):
        self.meta_data = dict(**self._additional_info)
