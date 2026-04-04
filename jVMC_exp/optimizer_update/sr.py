from .base import Evolution
from jVMC_exp.solver.pinv_snr import PinvSNR

class SR(Evolution):
    def __init__(
            self, sampler, psi, stepper, 
            use_cross_valiadation=False, diagonalShift=0.001, solver=PinvSNR()
        ):
        super().__init__(
            sampler, psi, stepper, True, True, 
            use_cross_valiadation, diagonalShift, solver
        )