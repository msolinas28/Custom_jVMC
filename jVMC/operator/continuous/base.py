import jax 
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Callable

from jVMC.geometry import AbstractGeometry
from jVMC.vqs import NQS, create_batches
import jVMC.global_defs as global_defs

# TODO: might implement an "acting_on" property
class Operator(ABC):
    def __init__(self, geometry: AbstractGeometry, is_multiplicative: bool):
        if not isinstance(geometry, AbstractGeometry):
            raise TypeError(f"geometry must be an AbstractGeometry, got {type(geometry)}")
        self.geometry = geometry
        self._is_multiplicative = is_multiplicative
        self._len_args = -1

    @property
    def is_multiplicative(self):
        return self._is_multiplicative

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            return SumOperator(self, ScaledOperator(IdentityOperator(self.geometry), other))
        elif isinstance(other, Operator):
            return SumOperator(self, other)
        else:
            raise NotImplemented
        
    def __radd__(self, other):
        if isinstance(other, (int, float, complex)):
            return SumOperator(ScaledOperator(IdentityOperator(self.geometry), other), self)
        else:
            raise NotImplemented
        
    def __neg__(self):
        return ScaledOperator(self, -1)
    
    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            return SumOperator(self, ScaledOperator(IdentityOperator(self.geometry), -other))
        elif isinstance(other, Operator):
            return SumOperator(self, -other)
        else:
            raise NotImplemented
        
    def __rsub__(self, other):
        if isinstance(other, (int, float, complex)):
            return SumOperator(ScaledOperator(IdentityOperator(self.geometry), other), -self)
        else:
            raise NotImplemented
        
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return ScaledOperator(self, other)
        elif isinstance(other, Operator):
            if self.is_multiplicative and other.is_multiplicative:
                return MulOperator(self, other)
            else:
                raise NotImplementedError("Multiplication only implemented for multiplicative operators.")
        elif callable(other):
            return CallableScaledOperator(self, other)
        else:
            raise NotImplemented
        
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return ScaledOperator(self, other)
        elif callable(other):
            return CallableScaledOperator(self, other)
        else:
            raise NotImplemented
    
    def _init_get_O_loc(self, len_args):
        self._get_O_loc_vmapd = jax.vmap(self._get_O_loc, in_axes=(0, None, None) + (None, ) * len_args)
        self._get_O_loc_pmapd = global_defs.pmap_for_my_devices(
            self._get_O_loc_vmapd,
            static_broadcasted_argnums=(1, ),
            in_axes=(0, None, None) + (None, ) * len_args
        )
        self._get_O_loc_batched_pmapd = global_defs.pmap_for_my_devices(
            self._get_O_loc_batched,
            static_broadcasted_argnums=(1, 3),
            in_axes=(0, None, None, None) + (None, ) * len_args
        )
        self._initialized = True

    def get_O_loc(self, s, psi: NQS, *args):
        apply_fun = psi.net.apply
        parameters = psi.parameters
        
        if self._len_args != len(args):
            self._len_args = len(args)
            self._init_get_O_loc(self._len_args)

        if psi.batchSize is not None:
            return self._get_O_loc_batched_pmapd(s, apply_fun, parameters, psi.batchSize, *args)
        else:
            return self._get_O_loc_pmapd(s, apply_fun, parameters, *args)
        
    def _get_O_loc_batched(self, s, apply_fun, parameters, batch_size, *args):
        sb = create_batches(s, batch_size)
        def scan_fun(c, x):
            return c, self._get_O_loc_vmapd(x, apply_fun, parameters, *args)
        res = jax.lax.scan(scan_fun, None, jnp.array(sb))[1].reshape((-1,))

        return res[:s.shape[0]]

    @abstractmethod
    def _get_O_loc(self, s, apply_fun, parameters, *args):
        """
        Implement (O 𝜓)(s) / 𝜓(s) 
        """
        pass

class IdentityOperator(Operator):
    def __init__(self, geometry):
        super().__init__(geometry, is_multiplicative=True)

    def _get_O_loc(self, s, apply_fun, parameters, *args):
        return jnp.asarray(1)
    
class SumOperator(Operator):
    def __init__(self, O_1: Operator, O_2: Operator):    
        if O_1.geometry != O_2.geometry:
            raise ValueError("Operators must share the same geometry.")

        super().__init__(O_1.geometry, O_1.is_multiplicative and O_2.is_multiplicative)
        self.O_1 = O_1
        self.O_2 = O_2

    def _get_O_loc(self, s, apply_fun, parameters, *args):
        return self.O_1._get_O_loc(s, apply_fun, parameters, *args) + self.O_2._get_O_loc(s, apply_fun, parameters, *args)
    
class ScaledOperator(Operator):
    def  __init__(self, O: Operator, scalar):
        if not isinstance(scalar, (int, float, complex)):
            raise ValueError('The second element has to be a scalar')
        super().__init__(O.geometry, O.is_multiplicative)
        self.scalar = scalar
        self.O = O

    def _get_O_loc(self, s, apply_fun, parameters, *args):
        return self.scalar * self.O._get_O_loc(s, apply_fun, parameters, *args)
    
class CallableScaledOperator(Operator):
    def __init__(self, O: Operator, f: Callable):
        super().__init__(O.geometry, O.is_multiplicative)
        self.callable = f
        self.O = O 

    def _get_O_loc(self, s, apply_fun, parameters, *args):
        return self.callable(*args) * self.O._get_O_loc(s, apply_fun, parameters, *args)
    
class MulOperator(Operator):
    def __init__(self, O_1: Operator, O_2: Operator):    
        if O_1.geometry != O_2.geometry:
            raise ValueError("Operators must share the same geometry.")

        super().__init__(O_1.geometry, is_multiplicative=True)
        self.O_1 = O_1
        self.O_2 = O_2

    def _get_O_loc(self, s, apply_fun, parameters, *args):
        return self.O_1._get_O_loc(s, apply_fun, parameters, *args) * self.O_2._get_O_loc(s, apply_fun, parameters, *args)