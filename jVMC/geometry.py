import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod
import jVMC

class AbstractGeometry(ABC):
    def __init__(self, n_particles, n_dim, PBC=True, extent=None):
        self._n_particles = n_particles
        self._n_dim = n_dim 

        if not isinstance(PBC, tuple):
            PBC = (PBC, ) * self.n_dim
        self._PBC = PBC

        if extent is None:
            extent = (1,) * self.n_dim
        elif not isinstance(extent, tuple):
            raise ValueError(f'Extent has to be a tuple, got {extent}.')
        self._extent = extent

    @property
    def n_particles(self):
        return self._n_particles
    
    @property
    def n_dim(self):
        return self._n_dim
    
    @property
    def PBC(self):
        return self._PBC
    
    @property
    def extent(self):
        return self._extent
    
    @abstractmethod
    def domain(self):
        """
        Return the domain of the geometry.
        """
        pass
    
    @abstractmethod
    def apply_PBC(self, x):
        """
        Apply PBC where needed given the geometry of the system.
        """
        pass

    @abstractmethod
    def uniform_populate(self, key):
        """
        Generate configuration of the particles with positions sampled from a uniform distribution.
        """
        pass

class HyperRectangle(AbstractGeometry):
    @property
    def domain(self):
        ext = jnp.array(self.extent)

        return jnp.stack([-ext / 2, ext / 2], axis=1)
    
    def apply_PBC(self, x):
        extent = jnp.array(self.extent)
        PBC = jnp.array(self.PBC, dtype=jnp.bool_)

        return jnp.where(PBC, ((x + extent/2) % extent) - extent/2, x)
    
    def uniform_populate(self, key, dtype=jnp.float64):
        low = self.domain[:, 0]
        shape = (self.n_particles, self.n_dim)
        samples = jax.random.uniform(jVMC.util.key_gen.format_key(key), shape, dtype=dtype)
        
        return low + samples * jnp.array(self.extent)