from jVMC.operator.continuous.base import Operator

class PotentialOperator(Operator):
    def __init__(self, geometry, potential):
        super().__init__(geometry, is_multiplicative=True)

        if not callable(potential):
            raise ValueError("The property potential has to be a function.")
        self._potential = potential
    
    @property
    def potential(self):
        return self._potential
        
    def local_value(self, s, apply_fun, parameters):
        return self.potential(s)