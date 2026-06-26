from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp
import os
import copy

import jVMC_exp.global_defs as global_defs

allowed_dtypes_real = {
    "f64": jnp.float64,
    "f32": jnp.float32,
    "f16": jnp.float16,
    "bf16": jnp.bfloat16
}
allowed_dtypes_cpx = {
    "c128": jnp.complex128,
    "c64": jnp.complex64
}
allowed_dtypes = {**allowed_dtypes_real, **allowed_dtypes_cpx}

def read_dtype(dtype_str: str | None, real: bool | None):
    dtype_str = dtype_str.lower()

    if dtype_str == "none":
        return None
    
    if real is not None:
        dtype_real = allowed_dtypes_real.get(dtype_str)
        dtype_cpx = allowed_dtypes_cpx.get(dtype_str)

        if real:
            if dtype_real is None:
                if dtype_cpx is not None:
                    raise ValueError("Trying to set a real variable with a complex dtype.")
                
                allowed = ", ".join(k for k in allowed_dtypes_real.keys())
                raise ValueError(
                    f"Allowed real dtypes are {allowed}. "
                    f"Got {dtype_str}."
                )
            return dtype_real
                
        if dtype_cpx is None:
            if dtype_real is not None:
                raise ValueError("Trying to set a complex variable with a real dtype.")
            
            allowed = ", ".join(k for k in allowed_dtypes_cpx.keys())
            raise ValueError(
                f"Allowed complex dtypes are {allowed}. "
                f"Got {dtype_str}."
            )
        return dtype_cpx
    
    dtype = allowed_dtypes.get(dtype_str)
    if dtype is None:
        allowed = ", ".join(k for k in allowed_dtypes.keys())
        raise ValueError(
            f"Allowed dtypes are {allowed}. "
            f"Got {dtype_str}."
        )
    
    return dtype

DT_SAMPLES_CONT = jnp.float64
DT_SAMPLES = jnp.int32
DT_PARAMS_REAL = jnp.float64 # TODO: Might change name in general real precision
DT_PARAMS_CPX = jnp.complex128
DT_OUT_REAL = read_dtype(os.environ.get("JVMC_DT_OUT_REAL", "f64"), True)
DT_OUT_CPX = read_dtype(os.environ.get("JVMC_DT_OUT_CPX", "c128"), False)
DT_SAMPLER = read_dtype(os.environ.get("JVMC_DT_SAMPLER", "none"), None)
DT_EVAL = read_dtype(os.environ.get("JVMC_DT_EVAL", "none"), None)
DT_GRAD = read_dtype(os.environ.get("JVMC_DT_GRAD", "none"), None)
DT_OPERATORS_REAL = DT_OUT_REAL
DT_OPERATORS_CPX = DT_OUT_CPX

USE_DISTRIBUTED = os.environ.get('JVMC_USE_DISTRIBUTED', 'false').lower() == 'true'