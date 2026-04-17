from optax import adam

from jVMC_exp.optimizer.base import AbstractOptimizer
from jVMC_exp.objective_function.base import ObjectiveFunctionOutput

class Adam(AbstractOptimizer):
    def __init__(
            self, sampler, psi,
            learning_rate: float=1e-3, b1: float=0.9, b2: float=0.999, eps: float=1e-8
    ):
        super().__init__(sampler, psi, False)

        self._adam = adam(learning_rate, b1, b2, eps)
        self._opt_state = self._adam.init(self.psi.parameters_flat)

    def get_update(self, objective_function_out: ObjectiveFunctionOutput):
        grad = objective_function_out.grad.mean.squeeze()
        update, self._opt_state = self._adam.update(grad, self._opt_state, self.psi.parameters_flat)

        return update
    
    def _update_meta_data(self):
        pass

    def cross_validation(self):
        pass