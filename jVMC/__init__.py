from . import sharding_config
from . import nets
from . import operator
from . import util
from . import vqs
from . import vqs_sharding
from . import sampler
from . import sampler_sharding
from . import stats
from . import stats_sharding
from . import geometry
from . import geometry_sharding
from . import propose
from . import propose_sharding
from . import global_defs

try:
    from . import mpi_wrapper
except ImportError:
    pass

from .version import __version__
from .global_defs import set_pmap_devices

import jax
jax.config.update("jax_enable_x64", True)