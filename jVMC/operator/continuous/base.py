import jax 
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Callable
import inspect
from typing import Callable, List

from jVMC.geometry import AbstractGeometry
from jVMC.vqs import NQS, create_batches
import jVMC.global_defs as global_defs

class Operator(ABC):
    def __init__(self, geometry: AbstractGeometry, is_diagonal: bool):
        if not isinstance(geometry, AbstractGeometry):
            raise TypeError(f"geometry must be an AbstractGeometry, got {type(geometry)}")
        self.geometry = geometry
        self._is_diagonal = is_diagonal
        self._init_pmap()
        self._ops = [self]
        
    def _init_pmap(self):
        self._get_O_loc_vmapd = jax.vmap(self._get_O_loc, in_axes=(0, None, None, None))
        self._get_O_loc_pmapd = global_defs.pmap_for_my_devices(
            self._get_O_loc_vmapd,
            static_broadcasted_argnums=(1,),
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
            new_ops = _ScaledOperator(IdentityOperator(self.geometry), other)._ops
            is_diagonal = self.is_diagonal
        elif isinstance(other, Operator):
            if other.geometry != self.geometry:
                raise ValueError("Only two operators with the same geometry can be summed!")
            new_ops = other._ops
            is_diagonal = self.is_diagonal and other.is_diagonal
        else:
            raise NotImplemented
        return _SumOperator(self._ops + new_ops, is_diagonal)
        
    def __radd__(self, other):
        if isinstance(other, (int, float, complex)):
            left_ops = _ScaledOperator(IdentityOperator(self.geometry), other)._ops
            return _SumOperator(left_ops + self._ops, self.is_diagonal)
        else:
            raise NotImplemented
            
    def __neg__(self):
        return -1 * self 
    
    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            new_ops = _ScaledOperator(IdentityOperator(self.geometry), -other)._ops
            is_diagonal = self.is_diagonal
        elif isinstance(other, Operator):
            if other.geometry != self.geometry:
                raise ValueError("Only two operators with the same geometry can be summed!")
            new_ops = [-O for O in other._ops]
            is_diagonal = self.is_diagonal and other.is_diagonal
        else:
            raise NotImplemented
        return _SumOperator(self._ops + new_ops, is_diagonal)
        
    def __rsub__(self, other):
        if isinstance(other, (int, float, complex)):
            left_ops = _ScaledOperator(IdentityOperator(self.geometry), other)._ops
            return _SumOperator(left_ops + [-O for O in self._ops])
        else:
            raise NotImplemented
        
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            ops = [_ScaledOperator(O, other) for O in self._ops]            
            if len(ops) == 1:
                return ops[0]
            else:
                return _SumOperator(ops, self.is_diagonal) 
            #TODO Implement case for Muloperator
        elif isinstance(other, Operator):
            if self.is_diagonal and other.is_diagonal:
                return MulOperator(self, other)
            else:
                raise NotImplementedError("Multiplication only implemented for diagonal operators.")
        else:
            raise NotImplemented
        
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            ops = [_ScaledOperator(O, other) for O in self._ops]            
            if len(ops) == 1:
                return ops[0]
            else:
                return _SumOperator(ops, self.is_diagonal) 
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
    
class _SumOperator(Operator):
    def __init__(self, ops, is_diagonal):
        if len(ops) == 0:
            raise ValueError("SumOperator requires at least one operator.")
        super().__init__(ops[0].geometry, is_diagonal=is_diagonal)
        self._ops = ops

    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        return jnp.asarray([O._get_O_loc(s, apply_fun, parameters, kwargs) for O in self._ops]).sum()
    
class _ScaledOperator(Operator):
    def __init__(self, O: Operator, scalar):
        if len(O._ops) !=1:
            raise NotImplementedError
        if not isinstance(scalar, (int, float, complex)):
            raise ValueError('The second element has to be a scalar')
        super().__init__(O.geometry, O.is_diagonal)

        self.scalar = scalar
        if isinstance(O, _ScaledOperator):
            self.scalar *= O.scalar
            self.O = O.O
        else:
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