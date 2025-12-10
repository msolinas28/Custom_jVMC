import jax
from jax.sharding import Mesh, NamedSharding
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P

try:
    jax.distributed.initialize()
    num_processes = jax.process_count()
    num_devices = jax.device_count() 
    print(f"JAX distributed initialized: {num_processes} processes and {num_devices} devices.")
except RuntimeError:
    pass

global_devices = mesh_utils.create_device_mesh((jax.device_count(),))
MESH = Mesh(global_devices, axis_names=("devices",))
DEVICE_SHARDING = NamedSharding(MESH, P("devices"))
REPLICATED_SHARDING = NamedSharding(MESH, P())
DEVICE_SPEC = P("devices")
REPLICATED_SPEC = P()