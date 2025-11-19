import jax 
import jax.numpy as jnp

from abc import ABC, abstractmethod
from jVMC.geometry import AbstractGeometry

class Operator(ABC):
    def __init__(self, geometry: AbstractGeometry):
        if not isinstance(geometry, AbstractGeometry):
            raise TypeError(f"geometry must be an AbstractGeometry, got {type(geometry)}")
        self.geometry = geometry

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
            return NotImplemented
        
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
            return NotImplemented
        
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return ScaledOperator(self, other)
        elif isinstance(other, Operator):
            return MulOperator(self, other)
        else:
            raise NotImplemented
        
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return ScaledOperator(self, other)
        else:
            return NotImplemented
    
    def local_value(self, s, apply_fun, parameters):
        """
        Implement (O 𝜓)(s) / 𝜓(s) 
        """
        pass

class IdentityOperator(Operator):
    def __init__(self, geometry):
        super().__init__(geometry)

    def local_value(self, s, apply_fun, parameters):
        return jnp.asarray(1)

class SumOperator(Operator):
    def __init__(self, O_1: Operator, O_2: Operator):    
        if O_1.geometry != O_2.geometry:
            raise ValueError("Operators must share the same geometry.")

        super().__init__(O_1.geometry)
        self.O_1 = O_1
        self.O_2 = O_2

    def local_value(self, s, apply_fun, parameters):
        return self.O_1.local_value(s, apply_fun, parameters) + self.O_2.local_value(s, apply_fun, parameters)
    
class ScaledOperator(Operator):
    def  __init__(self, O: Operator, scalar):
        if not isinstance(scalar, (int, float, complex)):
            raise ValueError('The second element has to be a scalar')
        super().__init__(O.geometry)
        self.scalar = scalar
        self.O = O

    def local_value(self, s, apply_fun, parameters):
        return self.scalar * self.O.local_value(s, apply_fun, parameters)
    
class MulOperator(Operator):
    def __init__(self, O_1: Operator, O_2: Operator):    
        if O_1.geometry != O_2.geometry:
            raise ValueError("Operators must share the same geometry.")

        super().__init__(O_1.geometry)
        self.O_1 = O_1
        self.O_2 = O_2

    def local_value(self, s, apply_fun, parameters):
        NotImplementedError