# sdtw-jax
Soft-DTW loss (with warp penalty) in JAX.

Usage:

```python
x = jax.random.normal(jax.random.PRNGKey(42), (4, 800, 80))
y = jnp.roll(x, 4, axis=1)
batched_sdtw(x, y, warp_penalty=1.0, temperature=0.01)
# DeviceArray([16.947954, 16.809141, 16.411541, 17.066374], dtype=float32)
```

Source:

```python
# Reference: https://arxiv.org/abs/2006.03575

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


def soft_minimum(values, temperature):
    values = jnp.stack(values, axis=-1)
    return -temperature * jax.nn.logsumexp(-values / temperature, axis=-1)


def skew_matrix(x):
    """Skew a matrix so that the diagonals become the rows."""
    clip = lambda x, a, b: min(max(x, a), b)
    height, width = x.shape
    ids = np.empty((height + width - 1, width), dtype=np.int32)
    for i in range(height + width - 1):
        for j in range(width):  # Shift each column j down by j steps.
            ids[i, j] = clip(i - j, 0, height - 1)
    x = jnp.take_along_axis(x, ids, axis=0)
    return x


def kernel_dist(kernel, xs, ys):
    """
    Returns:
    A 2d array `a` such that `a[i, j] = kernel(xs[i], ys[j])`.
    """
    return jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(ys))(xs)


@partial(jax.jit, static_argnums=(2, 3, 4))
def sdtw(a, b, warp_penalty=1.0, temperature=0.01, INFINITY=1e10):
    N, D1 = a.shape
    M, D2 = b.shape
    assert D1 == D2
    dist_fn = lambda x, y: jnp.mean(jnp.abs(x - y), axis=-1)
    cost = kernel_dist(dist_fn, a, b)
    size = cost.shape[-1]
    path_cost = INFINITY * jnp.ones((size + 1,))
    path_cost_prev = INFINITY * jnp.ones((size,))
    path_cost_prev = jnp.pad(path_cost_prev, (1, 0), constant_values=0.0)
    cost = skew_matrix(cost)

    def scan_fn(prev, inputs):
        path_cost_prev, path_cost = prev
        cost_i = inputs
        penalty_cost = path_cost + warp_penalty
        directions = [path_cost_prev[:-1], penalty_cost[1:], penalty_cost[:-1]]
        path_cost_next = cost_i + soft_minimum(directions, temperature)
        path_cost_next = jnp.pad(path_cost_next, (1, 0), constant_values=INFINITY)
        path_cost, path_cost_prev = path_cost_next, path_cost
        return (path_cost_prev, path_cost), None

    (path_cost_prev, path_cost), _ = jax.lax.scan(
        scan_fn, (path_cost_prev, path_cost), cost
    )
    return path_cost[-1]


@partial(jax.jit, static_argnums=(2, 3, 4))
def batched_sdtw(a, b, warp_penalty=1.0, temperature=0.01, INFINITY=1e8):
    dtw = partial(
        sdtw,
        warp_penalty=warp_penalty,
        temperature=temperature,
        INFINITY=INFINITY,
    )
    return jax.vmap(dtw)(a, b)
```
