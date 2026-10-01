import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import PartitionSpec as P
import functools
from functools import wraps
from dataclasses import dataclass
from typing import Iterator, ParamSpec, TypeVar, Callable
import math

from jVMC_exp import global_defs

if global_defs.USE_DISTRIBUTED:
    try:
        jax.distributed.initialize()
        num_processes = jax.process_count()
        num_devices = jax.device_count()
        if jax.process_index() == 0:
            print(f"JAX distributed initialized: {num_processes} processes and {num_devices} devices.")
    except Exception as e:
        if jax.process_index() == 0:
            print(f"Failed to initialize JAX distributed: {e}")
else:
    if jax.process_index() == 0:
        print("Running in single-node mode (JVMC_USE_DISTRIBUTED not set)")

global_devices = mesh_utils.create_device_mesh((jax.device_count(),))
MESH = Mesh(global_devices, axis_names=("devices",))
DEVICE_SHARDING = NamedSharding(MESH, P("devices"))
REPLICATED_SHARDING = NamedSharding(MESH, P())
DEVICE_SPEC = P("devices")
REPLICATED_SPEC = P()

def make_2d_mesh(axis_names=("row", "col")):
    num_devices = jax.device_count()

    p_rows = int(math.isqrt(num_devices))
    while num_devices % p_rows != 0:
        p_rows -= 1
    p_cols = num_devices // p_rows
    devices_2d = mesh_utils.create_device_mesh((p_rows, p_cols))
    
    return Mesh(devices_2d, axis_names=axis_names)

MESH_2D = make_2d_mesh(axis_names=("row", "col"))
DEVICE_SPEC_2D = P("row", "col")
DEVICE_SHARDING_2D = NamedSharding(MESH_2D, DEVICE_SPEC_2D)

@functools.lru_cache(maxsize=None)
def _local_batch_fns(n, b, n_batches, last):
    """
    Compiled helpers of a `BatchLayout` with ``n`` rows per device, ``b`` rows per device
    in a batch and ``last`` rows per device in the last batch. They only touch the local
    shard of each device, so no data moves between devices.
    """
    pad = n_batches * b - n

    def take(a, i):
        if pad:
            a = jnp.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1))
        return jax.lax.dynamic_slice_in_dim(a, i * b, b, axis=0)

    def trim_last(p):
        return p[:last]

    def put(out, p, i):
        return jax.lax.dynamic_update_slice_in_dim(out, p, i * b, axis=0)

    def put_last(out, p):
        return jax.lax.dynamic_update_slice_in_dim(out, p[:last], (n_batches - 1) * b, axis=0)

    def local(fn, in_specs):
        return jax.shard_map(fn, mesh=MESH, in_specs=in_specs, out_specs=DEVICE_SPEC)

    return {
        "take": jax.jit(local(take, (DEVICE_SPEC, REPLICATED_SPEC))),
        "trim_last": jax.jit(local(trim_last, (DEVICE_SPEC,))),
        "put": jax.jit(local(put, (DEVICE_SPEC, DEVICE_SPEC, REPLICATED_SPEC)), donate_argnums=0),
        "put_last": jax.jit(local(put_last, (DEVICE_SPEC, DEVICE_SPEC)), donate_argnums=0),
    }

@functools.lru_cache(maxsize=None)
def _alloc_fn(shape, dtype):
    return jax.jit(lambda: jnp.zeros(shape, dtype), out_shardings=DEVICE_SHARDING)

@dataclass(frozen=True)
class BatchLayout:
    """
    Splits ``num_samples`` samples, sharded across devices along the first axis, into
    batches of ``batch_size`` samples.

    Batch ``i`` is made of the ``i``-th block of ``batch_size // n_devices`` rows of every
    device's shard, so taking a batch out of an array, or writing the result of a batch
    back into one, never moves data between devices. The last batch holds the remaining
    rows of every device. Hence, on more than one device a batch does not consist of
    consecutive samples: any full-length array that has to line up with the batches
    must be split with `split` of the same layout.
    """
    num_samples: int
    batch_size: int

    def __post_init__(self):
        for name, value in (("number of samples", self.num_samples), ("batch size", self.batch_size)):
            if value <= 0 or value % MESH.size != 0:
                raise ValueError(
                    f"The {name} ({value}) has to be a positive multiple of the number of devices ({MESH.size})"
                )

    @property
    def n_batches(self):
        return math.ceil(self._local_size / self._local_batch_size)

    @property
    def _local_size(self):
        return self.num_samples // MESH.size

    @property
    def _local_batch_size(self):
        return self.batch_size // MESH.size

    @property
    def _last_local_size(self):
        return self._local_size - (self.n_batches - 1) * self._local_batch_size

    @property
    def _fns(self):
        return _local_batch_fns(self._local_size, self._local_batch_size, self.n_batches, self._last_local_size)

    def _is_padded(self, i):
        return i == self.n_batches - 1 and self._last_local_size < self._local_batch_size

    def take(self, x, i):
        """
        Batch ``i`` of the row-sharded array ``x``, with ``batch_size`` rows.
        The last batch is zero padded if it is not full.
        """
        if self._local_size == self._local_batch_size:
            return x
        return self._fns["take"](x, i)

    def trim(self, piece, i):
        """
        Removes the padding rows from the result of batch ``i``.
        """
        return self._fns["trim_last"](piece) if self._is_padded(i) else piece

    def split(self, x):
        """
        All batches of ``x``, the last one without padding.
        """
        if x.shape[0] != self.num_samples:
            raise ValueError(
                f"Expected an array with {self.num_samples} rows to split, got shape {x.shape}"
            )
        x = jax.device_put(x, DEVICE_SHARDING)

        return [self.trim(self.take(x, i), i) for i in range(self.n_batches)]

    def alloc(self, piece):
        """
        Row-sharded zeros with ``num_samples`` rows and the trailing shape and dtype of ``piece``.
        """
        return _alloc_fn((self.num_samples,) + tuple(piece.shape[1:]), piece.dtype)()

    def put(self, out, piece, i):
        """
        Writes the result of batch ``i`` (with or without padding) into ``out``, in place:
        ``out`` is donated and must not be used afterwards.
        """
        if i == self.n_batches - 1:
            return self._fns["put_last"](out, piece)
        return self._fns["put"](out, piece, i)

    def scaled(self, k):
        """
        Layout of arrays holding ``k`` consecutive rows per sample.
        """
        return BatchLayout(k * self.num_samples, k * self.batch_size)

    def batch_positions(self):
        """
        For every sample, its position in the concatenation of all batches.
        """
        b = self._local_batch_size
        local_rows = np.arange(self.num_samples).reshape(MESH.size, self._local_size)
        order = np.concatenate([local_rows[:, i * b:(i + 1) * b].reshape(-1) for i in range(self.n_batches)])
        positions = np.empty_like(order)
        positions[order] = np.arange(self.num_samples)

        return positions

@dataclass
class SizedIterable:
    reusable_iterable: Callable
    n_iterations: int
    batch_size: int
    layout: BatchLayout | None = None

    def __len__(self):
        return self.n_iterations

    def __iter__(self) -> Iterator:
        return self.reusable_iterable()

P = ParamSpec('P')
R = TypeVar('R')

def is_on_device(args, target_sharding=DEVICE_SHARDING):
    return any(jax.tree_util.tree_map(lambda x: x.sharding == target_sharding, args))

def _row_sharded(x):
    # No-op if x is already sharded along its first axis
    return jax.device_put(x, DEVICE_SHARDING)

def distribute(global_size: int, label: str | None=None):
    """
    Adjust a global array size to be compatible with device sharding.

    This helper ensures that a quantity intended to be sharded across the
    JAX device mesh is compatible with the number of available devices.
    In particular, it enforces that the global size is at least the number
    of devices and divisible by it, so that each device receives an equal
    shard.

    Parameters
    ----------
    global_size : int
        Total number of elements before sharding (e.g. number of chains,
        walkers, or other per-device replicated objects).
    label : str
        Human-readable label used in warning messages to identify the
        adjusted quantity.

    Returns
    -------
    int
        A possibly increased size that is:
        - greater than or equal to the number of devices, and
        - exactly divisible by the number of devices.

    Notes
    -----
    If `global_size` is smaller than the number of devices, it is increased
    to match the device count. If it is not divisible by the device count,
    it is rounded up to the next multiple. In both cases, a warning is
    printed to notify the user of the adjustment.
    """
    total_devices = MESH.shape["devices"]

    if global_size < total_devices:
        if label is not None:
            print(f"WARNING: Number of {label} ({global_size}) is smaller than the total number of devices ({total_devices}).")
            print(f"         Increased to: {total_devices}")
        return total_devices
    if global_size % total_devices != 0:
        adjusted_size = ((global_size + total_devices - 1) // total_devices) * total_devices
        if label is not None:
            print(f"WARNING: Number of {label} ({global_size}) is not divisible by the number of devices ({total_devices}).")
            print(f"         Increased to: {adjusted_size}")
        return adjusted_size

    return global_size

def broadcast_split_key(key, n_out_keys: int):
    """
    Split a PRNG key on a single host and broadcast the result to all hosts.

    This function ensures that all JAX processes receive the same set of
    PRNG subkeys in a multi-host setting. Only process 0 performs the key
    splitting, while all other processes allocate a dummy array of the
    correct shape. The resulting array of subkeys is then broadcast from
    process 0 to all other processes.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Base PRNG key used to generate subkeys. The key is only consumed
        on process 0.
    n_out_keys : int
        Number of PRNG subkeys to generate.

    Returns
    -------
    jax.Array
        Array of shape `(n_out_keys, 2)` containing PRNG subkeys, identical
        across all processes and suitable for further splitting or sharding.
    """
    if jax.process_index() == 0:
        # Only process 0 generates the keys
        out_keys = jax.random.split(key, n_out_keys)
    else:
        # Other processes create dummy data with the correct shape
        out_keys = jnp.zeros((n_out_keys, 2), dtype=jnp.uint32)

    # Broadcast from process 0 to all processes
    out_keys = multihost_utils.broadcast_one_to_all(out_keys)

    return out_keys.astype(jnp.uint32)

def create_batches(configs, b):
    append = b * ((configs.shape[0] + b - 1) // b) - configs.shape[0]
    pads = [(0, append), ] + [(0, 0)] * (len(configs.shape) - 1)

    return jnp.pad(configs, pads).reshape((-1, b) + configs.shape[1:])
    
class sharded:
    """Decorator to automatically create sharded versions of methods."""
    def __init__(
            self,
            static_argnums=None,
            static_kwarg_names=(),
            use_vmap=True, vmap_in_axes=None, # If None, default to (0,) * num_args
            in_specs=None,                    # If None, default to (DEVICE_SPEC,) * num_args
            out_specs=DEVICE_SPEC,
            automatic_sharding=False,
            donate_argnums=None,
            yield_iter=False,
    ):
        self.static_argnums = static_argnums
        self.static_kwarg_names = set(static_kwarg_names + ('batch_size',))
        self.use_vmap = use_vmap
        self.vmap_in_axes = vmap_in_axes
        self.in_specs = in_specs
        self.in_sharding = None
        self.out_specs = out_specs
        self.automatic_sharding = automatic_sharding
        self.donate_argnums = donate_argnums
        self.yield_iter = yield_iter

    def __call__(self, method: Callable[P, R]) -> Callable[P, R]:
        @wraps(method)
        def wrapper(instance, *args, **kwargs):
            jsh_fn = self._get_jsh(instance, method, args, kwargs)
            batch_size, kwargs = self._split_batch_size(kwargs)
            num_samples = args[0].shape[0]

            if batch_size is None and not self.yield_iter:
                args = tuple(jax.device_put(a, self.in_sharding[i]) for i, a in enumerate(args))
                return jsh_fn(kwargs, *args)

            if num_samples % MESH.size != 0:
                if self.yield_iter:
                    raise ValueError(
                        f"Lazy evaluation requires the number of samples ({num_samples}) "
                        f"to be divisible by the number of devices ({MESH.size})"
                    )
                return self._call_contiguous(batch_size, kwargs, args, jsh_fn)

            if REPLICATED_SPEC in self.in_specs:
                raise NotImplementedError("Batched evaluation of replicated arguments is not supported")

            layout = BatchLayout(num_samples, num_samples if batch_size is None else batch_size)
            args = tuple(jax.device_put(a, DEVICE_SHARDING) for a in args)
            if self.yield_iter:
                return SizedIterable(
                    reusable_iterable=lambda: self._iter_local_batches(layout, kwargs, args, jsh_fn),
                    n_iterations=layout.n_batches,
                    batch_size=layout.batch_size,
                    layout=layout
                )

            return self._call_local(layout, kwargs, args, jsh_fn)

        return wrapper

    def _split_batch_size(self, kwargs):
        batch_size = kwargs['batch_size']
        kwargs = {k: v for k, v in kwargs.items() if k not in self.static_kwarg_names}
        
        return batch_size, kwargs

    def _get_jsh(self, instance, method, args, kwargs):
        if not hasattr(instance, '_sharded_cache'):
            instance._sharded_cache = {}

        # Static kwargs are baked into the compiled function, so they are part of the key
        static_kwargs = {k: v for k, v in kwargs.items() if k in self.static_kwarg_names}
        cache_key = (method.__name__, tuple(sorted(static_kwargs.items())))

        if cache_key not in instance._sharded_cache:
            if self.in_specs is None:
                self.in_specs = (DEVICE_SPEC,) * len(args)
            elif len(self.in_specs) != len(args):
                raise ValueError(f"in_specs length ({len(self.in_specs)}) must match "
                                 f"number of args ({len(args)})")
            self.in_sharding = tuple(
                REPLICATED_SHARDING if REPLICATED_SPEC == s else DEVICE_SHARDING for s in self.in_specs
            )

            if self.vmap_in_axes is None:
                self.vmap_in_axes = (0,) * len(args)
            elif len(self.vmap_in_axes) != len(args):
                raise ValueError(f"vmap_in_axes length ({len(self.vmap_in_axes)}) must match "
                                 f"number of args ({len(args)})")

            if kwargs['batch_size'] is not None and kwargs['batch_size'] % MESH.size != 0:
                raise ValueError(f"The batch size ({kwargs['batch_size']}) "
                                 f"has to be divisible by the number of devices ({MESH.size})")

            base_fn = lambda kw, *a: method(instance, *a, **kw, **static_kwargs)
            instance._sharded_cache[cache_key] = self._create_sharded_versions(base_fn)

        return instance._sharded_cache[cache_key]['jsh']

    def _create_sharded_versions(self, base_fn):
        vmapd_fn = jax.vmap(
            base_fn, in_axes=(None,) + self.vmap_in_axes
        ) if self.use_vmap else base_fn

        if self.automatic_sharding:
            jsh_fn = jax.jit(
                vmapd_fn, 
                static_argnums=self.static_argnums, 
                donate_argnums=self.donate_argnums
            )
        else:
            jsh_fn = jax.jit(
                jax.shard_map(
                    vmapd_fn,
                    mesh=MESH,
                    in_specs=(REPLICATED_SPEC,) + self.in_specs,
                    out_specs=self.out_specs
                ),
                static_argnums=self.static_argnums,
                donate_argnums=self.donate_argnums
            )

        return {'single': base_fn, 'vmapd': vmapd_fn, 'jsh': jsh_fn}

    def _iter_local_batches(self, layout, kwargs, args, jsh_fn):
        """
        Generator yielding the result of one batch of ``layout`` at a time, without padding.
        """
        for i in range(layout.n_batches):
            result = jsh_fn(kwargs, *(layout.take(a, i) for a in args))

            yield jax.tree_util.tree_map(lambda x: layout.trim(_row_sharded(x), i), result)

    def _call_local(self, layout, kwargs, args, jsh_fn):
        """
        Evaluates all batches of ``layout`` and writes their results in place into
        preallocated outputs, which keep the order of the input samples.
        """
        if layout.n_batches == 1:
            return next(self._iter_local_batches(layout, kwargs, args, jsh_fn))

        out = None
        for i in range(layout.n_batches):
            result = jsh_fn(kwargs, *(layout.take(a, i) for a in args))
            result = jax.tree_util.tree_map(_row_sharded, result)
            if out is None:
                out = jax.tree_util.tree_map(layout.alloc, result)
            out = jax.tree_util.tree_map(lambda o, r: layout.put(o, r, i), out, result)

        return out

    def _call_contiguous(self, batch_size, kwargs, args, jsh_fn):
        """
        Fallback for a number of samples that is not divisible by the number of devices:
        batches of consecutive samples, concatenated at the end. Unlike the local batches,
        this moves data between devices.
        """
        results = list(self._iter_contiguous_batches(batch_size, kwargs, *args, jsh_fn=jsh_fn))
        if len(results) == 1:
            return results[0]

        return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *results)

    def _iter_contiguous_batches(self, batch_size, kwargs, *args, jsh_fn):
        """
        Generator yielding one (trimmed) chunk result at a time.
        Assumes batch_size is divisible by number of devices.
        """
        num_samples = args[0].shape[0]
        append = (-num_samples) % batch_size
        total_samples = num_samples + append

        if (total_samples > batch_size) and is_on_device(args):
            args = tuple(jax.device_put(a, REPLICATED_SHARDING) for a in args)

        num_batches = total_samples // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            batch = tuple(a[start:start + batch_size] for a in args)
            if batch_idx == num_batches - 1 and append:
                batch = tuple(
                    jnp.pad(b, [(0, append),] + [(0, 0)] * (b.ndim - 1)) 
                    for b in batch
                )

            result = jsh_fn(
                kwargs,
                *tuple(
                    jax.device_put(b, self.in_sharding[arg_idx]) 
                    for arg_idx, b in enumerate(batch)
                )
            )

            if batch_idx == num_batches - 1 and append:
                trim = batch_size - append
                result = jax.tree_util.tree_map(lambda x: x[:trim], result)

            yield result