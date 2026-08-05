import jax
import jax.numpy as jnp
import warnings
import numpy as np
from typing import Literal
import math
import jaxmg

from jVMC_exp.sharding_config import MESH, MESH_2D, DEVICE_SHARDING_2D

def _eigh_numpy(S):
    e, V = np.linalg.eigh(np.array(S))

    return jnp.array(e), jnp.array(V)

def diagonalize(
        A, 
        pad_size: int = 0, 
        mode: Literal["device", "distributed", "host"] = "device",
        T_A: int | None = None
    ):
    if not jnp.allclose(A, A.conjugate().T):
        raise ValueError(
            "The given matrix is not Hermitian, "
            "thus can not be diagonalized with this method"
        )

    if mode.lower() == "distributed" and jax.default_backend() != "gpu":
        if jax.process_index() == 0:
            warnings.warn(
                "mode='distributed' requires a real GPU backend, "
                f"got jax.default_backend()={jax.default_backend()}. "
                "Falling back to mode='device'."
            )
        mode = "device"

    elif mode.lower() == "distributed" and A.shape[0] % MESH.shape["devices"] != 0:
        if jax.process_index() == 0:
            warnings.warn(
                "mode=distributed requires the given matrix to "
                "have shape divisible by the number of devices."
                f"Got shape={A.shape}, n_devices={MESH.shape['devices']}."
                "Falling back to mode='device'."
            )
        mode = "device"

    elif mode.lower() == "distributed" and MESH.shape["devices"] == 1:
        if jax.process_index() == 0:
            warnings.warn(
                "Got mode=distributed but there is only one device. "
                "Falling back to mode='device'."
            )
        mode = "device"

    if pad_size !=0:
        # Frobenius norm >= spectral radius for Hermitian A
        # Used here to recognize the fake eigenvalues given by padding
        c = 2 * jnp.linalg.norm(A) + 1
        N = A.shape[0] - pad_size
        pad_idx = jnp.arange(N, A.shape[0])
        A = A.at[pad_idx, pad_idx].add(c)

    if mode.lower() == "device":
        try:
            ev, V = jnp.linalg.eigh(A)
        except ValueError:
            if jax.process_index() == 0:
                warnings.warn(
                    "jax.numpy.linalg.eigh raised an exception. Falling back to " 
                    "numpy.linalg.eigh for diagonalization.", RuntimeWarning
                )
            ev, V = _eigh_numpy(A)

    elif mode.lower() == "host":
        ev, V = _eigh_numpy(A)

    elif mode.lower() == "distributed":
        A = jax.device_put(A, DEVICE_SHARDING_2D, donate=True)

        if T_A is None:
            p_rows, p_cols = MESH_2D.shape["row"], MESH_2D.shape["col"]
            T_A = math.gcd((A.shape[0]) // p_rows, (A.shape[0]) // p_cols)

        ev, V = jaxmg.syevd(A, T_A, MESH_2D)

    else:
        raise ValueError(
            f"The available modes are 'device', 'host' or 'distributed' got {mode}."
        )

    if pad_size != 0:
        ev = ev[:N]
        V = V[:N, :N]

    return ev, V

    # self._Vtb = jnp.dot(jnp.transpose(jnp.conj(self._V)), b) # TODO: move it from here