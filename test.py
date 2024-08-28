import os

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".10"
#os.environ["JAX_LOG_LEVEL"] = "debug"

#import logging
#logging.basicConfig(level=logging.DEBUG)

import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000, 1000))
y = jnp.dot(x, x.T)
print(y)

print(jax.devices())