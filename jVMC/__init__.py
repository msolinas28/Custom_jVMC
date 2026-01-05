from . import sharding_config
from . import nets
from . import util
from . import vqs_sharding
from . import sampler_sharding
from . import geometry_sharding
from . import propose_sharding
from . import global_defs

from .version import __version__
from .global_defs import set_pmap_devices

import jax
jax.config.update("jax_enable_x64", True)