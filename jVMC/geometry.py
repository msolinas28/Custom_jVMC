import jax.numpy as jnp

class Geometry():
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
    
    @property
    def modulus(self):
        return jnp.where(jnp.array(self.PBC), jnp.array(self.extent), jnp.inf)