from jVMC_exp.util.input_manager import *
from jVMC_exp.util.output_manager import *
from jVMC_exp.util.util import *

from typing import TYPE_CHECKING

from .grads import *
from .key_gen import *

if TYPE_CHECKING:
    # Seen by static analyzers only: at runtime these names are imported by __getattr__
    from .ptvmc import PTVMC, AbstractPropagator, LevenbergMarquardt, ProductExpansion, Stage

_PTVMC_NAMES = ("PTVMC", "AbstractPropagator", "ProductExpansion", "Stage", "LevenbergMarquardt")


def __getattr__(name):
    # ptvmc depends on the objective functions, which import this package: it is imported on
    # first access
    if name in _PTVMC_NAMES:
        from . import ptvmc

        return getattr(ptvmc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
