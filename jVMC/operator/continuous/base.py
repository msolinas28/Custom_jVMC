import jax 
import jax.numpy as jnp

from abc import ABC, abstractmethod
from jVMC.geometry import AbstractGeometry

class Operator(ABC):
    def __init__(self, geometry: AbstractGeometry):
        if not isinstance(geometry, AbstractGeometry):
            raise TypeError(f"geometry must be an AbstractGeometry, got {type(geometry)}")
        self.geometry = geometry

    def __call__(self, s, apply_fun, parameters):
        """
        Implement (O 𝜓)(s).
        """

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            # TODO: implementation of sum_operator(identity * other, self)
            pass
        elif isinstance(other, Operator):
            # TODO: implement SumOperator(other, self)
            pass

    def local_value(self, s, apply_fun, parameters):
        """
        Implement (O 𝜓)(s) / 𝜓(s)
        """

class SumOperator(Operator):
    # Maybe implement here also the identity sum
    def __init__(self, O_1: Operator, O_2: Operator):
        if not O_1.geometry == O_2.geometry:
            raise NotImplementedError()
        super().__init__(O_1.geometry)
        self.O_1 = O_1
        self.O_2 = O_2

    def __call__(self, s, apply_fun, parameters):
        return self.O_1(s, apply_fun, parameters) + self.O_2(s, apply_fun, parameters)