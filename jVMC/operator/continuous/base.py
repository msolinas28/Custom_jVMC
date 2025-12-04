import jax 
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Callable
import inspect

from jVMC.geometry import AbstractGeometry
from jVMC.vqs import NQS, create_batches
import jVMC.global_defs as global_defs

def _flatten_ops(op):
    if isinstance(op, SumOperator):
        return op.ops
    else:
        return [op]

class Operator(ABC):
    def __init__(self, geometry: AbstractGeometry, is_diagonal: bool):
        if not isinstance(geometry, AbstractGeometry):
            raise TypeError(f"geometry must be an AbstractGeometry, got {type(geometry)}")
        self.geometry = geometry
        self._is_diagonal = is_diagonal
        self._init_pmap()
        
    def _init_pmap(self):
        self._get_O_loc_vmapd = jax.vmap(self._get_O_loc, in_axes=(0, None, None, None))
        self._get_O_loc_pmapd = global_defs.pmap_for_my_devices(
            self._get_O_loc_vmapd,
            static_broadcasted_argnums=(1, ),
            in_axes=(0, None, None, None)
        )
        self._get_O_loc_batched_pmapd = global_defs.pmap_for_my_devices(
            self._get_O_loc_batched,
            static_broadcasted_argnums=(1, 3),
            in_axes=(0, None, None, None, None)
        )

    @property
    def is_diagonal(self):
        return self._is_diagonal

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            right_op = _flatten_ops(ScaledOperator(IdentityOperator(self.geometry), other))
        elif isinstance(other, Operator):
            right_op = _flatten_ops(other)
        else:
            raise NotImplemented
        return SumOperator(_flatten_ops(self) + right_op)
        
    def __radd__(self, other):
        if isinstance(other, (int, float, complex)):
            left_ops = _flatten_ops(ScaledOperator(IdentityOperator(self.geometry), other))
            return SumOperator(left_ops + _flatten_ops(self))
        else:
            raise NotImplemented
            
    def __neg__(self):
        return ScaledOperator(self, -1)
    
    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            right_op = _flatten_ops(ScaledOperator(IdentityOperator(self.geometry), -other))
        elif isinstance(other, Operator):
            right_op = _flatten_ops(-other)
        else:
            raise NotImplemented
        return SumOperator(_flatten_ops(self) + right_op)
        
    def __rsub__(self, other):
        if isinstance(other, (int, float, complex)):
            left_ops = _flatten_ops(ScaledOperator(IdentityOperator(self.geometry), other))
            return SumOperator(left_ops + _flatten_ops(-self))
        else:
            raise NotImplemented
        
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return ScaledOperator(self, other)
        elif isinstance(other, Operator):
            if self.is_diagonal and other.is_diagonal:
                return MulOperator(self, other)
            else:
                raise NotImplementedError("Multiplication only implemented for diagonal operators.")
        else:
            raise NotImplemented
        
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return ScaledOperator(self, other)
        else:
            raise NotImplemented
        
    def make_dynamic(self, f_t: Callable):
        return TimeDependentOperator(self, f_t)

    def get_O_loc(self, s, psi: NQS, **kwargs):
        apply_fun = psi.net.apply
        parameters = psi.parameters

        if psi.batchSize is not None:
            return self._get_O_loc_batched_pmapd(s, apply_fun, parameters, psi.batchSize, kwargs)
        else:
            return self._get_O_loc_pmapd(s, apply_fun, parameters, kwargs)
        
    def _get_O_loc_batched(self, s, apply_fun, parameters, batch_size, kwargs):
        sb = create_batches(s, batch_size)
        def scan_fun(c, x):
            return c, self._get_O_loc_vmapd(x, apply_fun, parameters, kwargs)
        res = jax.lax.scan(scan_fun, None, jnp.array(sb))[1].reshape((-1,))

        return res[:s.shape[0]]

    @abstractmethod
    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        """
        Implement (O 𝜓)(s) / 𝜓(s) 
        """
        pass

class IdentityOperator(Operator):
    def __init__(self, geometry):
        super().__init__(geometry, is_diagonal=True)

    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        return jnp.asarray(1)
    
class SumOperator(Operator):
    def __init__(self, ops):
        if len(ops) == 0:
            raise ValueError("SumOperator requires at least one operator.")        
        for o in ops:
            if not isinstance(o, Operator):
                raise TypeError("All entries in ops must be Operator instances.")
            if o.geometry != ops[0].geometry:
                raise ValueError("All operators must share the same geometry.")
        super().__init__(ops[0].geometry, is_diagonal=all(o.is_diagonal for o in ops))
        
        flat = []
        for o in ops:
            if isinstance(o, SumOperator):
                flat.extend(o.ops)
            else:
                flat.append(o)
        self.ops = flat

    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        return jnp.asarray([O._get_O_loc(s, apply_fun, parameters, kwargs) for O in self.ops]).sum()
    
class ScaledOperator(Operator):
    def  __init__(self, O: Operator, scalar):
        if not isinstance(scalar, (int, float, complex)):
            raise ValueError('The second element has to be a scalar')
        super().__init__(O.geometry, O.is_diagonal)
        self.scalar = scalar
        self.O = O

    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        return self.scalar * self.O._get_O_loc(s, apply_fun, parameters, kwargs)
    
class MulOperator(Operator):
    def __init__(self, O_1: Operator, O_2: Operator):    
        if O_1.geometry != O_2.geometry:
            raise ValueError("Operators must share the same geometry.")

        super().__init__(O_1.geometry, is_diagonal=True)
        self.O_1 = O_1
        self.O_2 = O_2

    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        return self.O_1._get_O_loc(s, apply_fun, parameters, kwargs) * self.O_2._get_O_loc(s, apply_fun, parameters, kwargs)
    
class TimeDependentOperator(Operator):
    def __init__(self, O: Operator, f_t: Callable):
        if not callable(f_t):
            raise ValueError("The argument 'f_t' has to a be function.")
        if len(inspect.signature(f_t).parameters) != 1:
            raise ValueError(f"The function 'f_t' has to take only one argument, got {len(inspect.signature(f_t).parameters)}.")
        super().__init__(O.geometry, O.is_diagonal)

        self.O = O
        self.f_t = f_t

    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        if "t" not in kwargs:
            raise KeyError("'t' must be provided in kwargs for time-dependent operators.")
        
        return self.f_t(kwargs["t"]) * self.O._get_O_loc(s, apply_fun, parameters, kwargs)