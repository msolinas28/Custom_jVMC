import jax
from jax.sharding import Mesh, NamedSharding
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P

def init_dist():
    try:
        jax.distributed.initialize()
    except RuntimeError:
        # already initialized or not in multi-host mode
        pass

    # Create mesh
    global_devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(global_devices, axis_names=("devices",))

    # Define shardings
    device_sharding = NamedSharding(mesh, P("devices"))
    replicated_sharding = NamedSharding(mesh, P())

    return mesh, device_sharding, replicated_sharding