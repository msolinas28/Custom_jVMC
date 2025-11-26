import jax 
import jax.numpy as jnp

from abc import ABC, abstractmethod
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

        # Understand if 0 is correct since s.shape[0] is usually the number of devices
        self._get_O_loc_vmapd = jax.vmap(self._get_O_loc, in_axes=(0, None, None))
        self._get_O_loc_pmapd = global_defs.pmap_for_my_devices(
            self._get_O_loc_vmapd,
            static_broadcasted_argnums=(1, ),
            in_axes=(0, None, None)
        )
        self._get_O_loc_batched_pmapd = global_defs.pmap_for_my_devices(
            self._get_O_loc_batched,
            static_broadcasted_argnums=(1, ),
            in_axes=(0, None, None, None)
        )

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
        else:
            raise NotImplemented
        
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return ScaledOperator(self, other)
        else:
            raise NotImplemented
    
    def get_O_loc(self, s, psi: NQS):
        apply_fun = psi.net.apply
        parameters = psi.parameters

        if psi.batchSize is not None:
            return self._get_O_loc_batched_pmapd(s, apply_fun, parameters, psi.batchSize)
        else:
            return self._get_O_loc_pmapd(s, apply_fun, parameters)
        
    def _get_O_loc_batched(self, s, apply_fun, parameters, batch_size):
        sb = create_batches(s, batch_size)
        def scan_fun(c, x):
            return c, self._get_O_loc_vmapd(x, apply_fun, parameters)
        res = jax.lax.scan(scan_fun, None, jnp.array(sb))[1].reshape((-1,))

        return res[:s.shape[0]]

    @abstractmethod
    def _get_O_loc(self, s, apply_fun, parameters):
        """
        Implement (O 𝜓)(s) / 𝜓(s) 
        """
        pass

class IdentityOperator(Operator):
    def __init__(self, geometry):
        super().__init__(geometry, is_multiplicative=True)

    def _get_O_loc(self, s, apply_fun, parameters):
        return jnp.asarray(1)
    
class SumOperator(Operator):
    def __init__(self, O_1: Operator, O_2: Operator):    
        if O_1.geometry != O_2.geometry:
            raise ValueError("Operators must share the same geometry.")

        super().__init__(O_1.geometry, O_1.is_multiplicative and O_2.is_multiplicative)
        self.O_1 = O_1
        self.O_2 = O_2

    def _get_O_loc(self, s, apply_fun, parameters):
        return self.O_1._get_O_loc(s, apply_fun, parameters) + self.O_2._get_O_loc(s, apply_fun, parameters)
    
class ScaledOperator(Operator):
    def  __init__(self, O: Operator, scalar):
        if not isinstance(scalar, (int, float, complex)):
            raise ValueError('The second element has to be a scalar')
        super().__init__(O.geometry, O.is_multiplicative)
        self.scalar = scalar
        self.O = O

    def _get_O_loc(self, s, apply_fun, parameters):
        return self.scalar * self.O._get_O_loc(s, apply_fun, parameters)
    
class MulOperator(Operator):
    def __init__(self, O_1: Operator, O_2: Operator):    
        if O_1.geometry != O_2.geometry:
            raise ValueError("Operators must share the same geometry.")

        super().__init__(O_1.geometry, is_multiplicative=True)
        self.O_1 = O_1
        self.O_2 = O_2

    def _get_O_loc(self, s, apply_fun, parameters):
        return self.O_1._get_O_loc(s, apply_fun, parameters) * self.O_2._get_O_loc(s, apply_fun, parameters)