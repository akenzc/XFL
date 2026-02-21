import pytest
import jax
import jax.numpy as jnp
from algorithm.model.vgg_jax import vggjax

def test_vgg11_jax():
    model = vggjax(num_classes=10, layers=11)
    key = jax.random.PRNGKey(0)
    x = jnp.ones((2, 224, 224, 3))
    
    variables = model.init(key, x, train=False)
    assert variables is not None
    
    outputs = model.apply(variables, x, train=False)
    assert outputs.shape == (2, 10)