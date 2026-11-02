import jax
import jax.numpy as jnp
import numpy as np

def format_key(key):
    if isinstance(key, jax.Array):
        if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
            return jax.random.key_data(key)
        return key
    elif key is None:
        return jax.random.PRNGKey(generate_seed())
    else:
        return jax.random.PRNGKey(int(key))
    
def generate_seed():
    return np.random.SeedSequence().entropy % int(2 ** 32)