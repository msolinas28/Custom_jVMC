import jax
import time

def format_key(key):
    if isinstance(key, jax.Array):
        return key
    elif key is None:
        return jax.random.PRNGKey(time.time())
    else:
        return jax.random.PRNGKey(int(key))