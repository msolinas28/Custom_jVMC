from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp
import os
import copy

import jVMC_exp.global_defs as global_defs

allowed_dtyes_real = {
    "f64": jnp.float64,
    "f32": jnp.float32,
    "f16": jnp.float16,
    "bf16": jnp.bfloat16
}
allowed_dtyes_cpx = {
    "c128": jnp.complex128,
    "c64": jnp.complex64
}
allowed_dtyes = copy.deepcopy(allowed_dtyes_real)
allowed_dtyes.update(allowed_dtyes_cpx)

def read_dtype(dtype_str: str | None, real: bool | None):
    dtype_str = dtype_str.lower()

    if dtype_str == "none":
        return None
    
    if real is not None:
        dtype_real = allowed_dtyes_real.get(dtype_str)
        dtype_cpx = allowed_dtyes_cpx.get(dtype_str)

        if real:
            if dtype_real is None:
                if dtype_cpx is not None:
                    raise ValueError("Trying to set a real variable with a complex dtype.")
                
                allowed = ", ".join(k for k in allowed_dtyes_real.keys())
                raise ValueError(
                    f"Allowed real dtypes are {allowed}. "
                    f"Got {dtype_str}."
                )
            return dtype_real
                
        if dtype_cpx is None:
            if dtype_real is not None:
                raise ValueError("Trying to set a complex variable with a real dtype.")
            
            allowed = ", ".join(k for k in allowed_dtyes_cpx.keys())
            raise ValueError(
                f"Allowed complex dtypes are {allowed}. "
                f"Got {dtype_str}."
            )
        return dtype_cpx
    
    dtype = allowed_dtyes.get(dtype_str)
    if dtype is None:
        allowed = ", ".join(k for k in allowed_dtyes.keys())
        raise ValueError(
            f"Allowed dtypes are {allowed}. "
            f"Got {dtype_str}."
        )
    
    return dtype
     

USE_DISTRIBUTED = os.environ.get('JVMC_USE_DISTRIBUTED', 'false').lower() == 'true'

DT_SAMPLES_CONT = jnp.float64
DT_SAMPLES = jnp.int32
DT_PARAMS_REAL = jnp.float64
DT_PARAMS_CPX = jnp.complex128
DT_OUT_REAL = read_dtype(os.environ.get("JVMC_DT_OUT_REAL", "f64"), True)
DT_OUT_CPX = read_dtype(os.environ.get("JVMC_DT_OUT_CPX", "c128"), False)
DT_SAMPLER = read_dtype(os.environ.get("JVMC_DT_SAMPLER", "none"), None)
DT_EVAL = read_dtype(os.environ.get("JVMC_DT_EVAL", "none"), None)
DT_GRAD = read_dtype(os.environ.get("JVMC_DT_GRAD", "none"), None)
DT_OPERATORS_REAL = DT_OUT_REAL
DT_OPERATORS_CPX = DT_OUT_CPX

@dataclass(frozen=True)
class PrecisionDTypes:
    mode: str
    param_dtype: any
    param_complex_dtype: any
    compute_dtype: any
    observable_dtype: any
    complex_dtype: any

def precision_dtypes(mode: str) -> PrecisionDTypes:
    key = str(mode).strip().lower()
    if key == "mixed_fp32":
        return PrecisionDTypes(
            mode=key,
            param_dtype=jnp.float32,
            param_complex_dtype=jnp.complex64,
            compute_dtype=jnp.float32,
            observable_dtype=jnp.float64,
            complex_dtype=jnp.complex128,
        )
    if key == "full_fp64":
        return PrecisionDTypes(
            mode=key,
            param_dtype=jnp.float64,
            param_complex_dtype=jnp.complex128,
            compute_dtype=jnp.float64,
            observable_dtype=jnp.float64,
            complex_dtype=jnp.complex128,
        )
    raise ValueError("precision_mode must be 'mixed_fp32' or 'full_fp64'.")

def set_precision_mode(mode: str) -> PrecisionDTypes:
    dtypes = precision_dtypes(mode)
    global_defs.DT_PARAMS_REAL = dtypes.param_dtype
    global_defs.DT_PARAMS_CPX = dtypes.param_complex_dtype
    global_defs.DT_OPERATORS_REAL = dtypes.observable_dtype
    global_defs.DT_OPERATORS_CPX = dtypes.complex_dtype
    return dtypes
