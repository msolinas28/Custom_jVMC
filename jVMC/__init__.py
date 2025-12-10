from . import global_defs
from . import sharding_config
from . import nets
from . import operator
from . import util
from . import mpi_wrapper
from . import vqs
from . import vqs_sharding
from . import sampler
from . import sampler_sharding
from . import stats
from . import geometry
from . import propose

from .version import __version__
from .global_defs import set_pmap_devices

import jax
jax.config.update("jax_enable_x64", True)