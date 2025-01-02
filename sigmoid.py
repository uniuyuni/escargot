
import numpy as np
import jax.numpy as jnp
from jax import jit

@jit
def __naive_sigmoid(x, gain, mid):    
    return 1.0 / (1.0 + jnp.exp((mid - x) * gain))

def naive_sigmoid(x, gain, mid):
    return np.array((__naive_sigmoid(x, gain, mid)).block_until_ready())

@jit
def __scaled_sigmoid(x, gain, mid):
    min = __naive_sigmoid(0.0, gain, mid)
    max = __naive_sigmoid(1.0, gain, mid)
    s = __naive_sigmoid(x, gain, mid)
    return (s - min) / (max - min)

def scaled_sigmoid(x, gain, mid):
    return np.array((__scaled_sigmoid(x, gain, mid)).block_until_ready())

@jit
def __naive_inverse_sigmoid(x, gain, mid):
    min = __naive_sigmoid(0.0, gain, mid)
    max = __naive_sigmoid(1.0, gain, mid)
    #s = __naive_sigmoid(jnp.clip(x, 1e-7, 1 - 1e-7), gain, mid)
    a = (max - min) * x + min
    return jnp.log(1.0 / a - 1.0)

def naive_inverse_sigmoid(x, gain, mid):
    return np.array((__naive_inverse_sigmoid(x, gain, mid)).block_until_ready())

@jit
def __scaled_inverse_sigmoid(x, gain, mid):
    min = __naive_inverse_sigmoid(0.0, gain, mid)
    max = __naive_inverse_sigmoid(1.0, gain, mid)
    s = __naive_inverse_sigmoid(x, gain, mid)
    return ((s - min) / (max - min))

def scaled_inverse_sigmoid(x, gain, mid):
    return np.array((__scaled_inverse_sigmoid(x, gain, mid)).block_until_ready())
