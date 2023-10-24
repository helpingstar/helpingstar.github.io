---
layout: single
title: "JAX example, Class method"
date: 2023-10-24 11:10:32
lastmod : 2023-10-24 11:10:32
categories: jax
tag: [jax]
use_math: false
---

JAX의 코드 작성방식을 다양한 예시를 통해 학습한다. 

## 1

* 출처 : [https://github.com/google/jax/discussions/10598](https://github.com/google/jax/discussions/10598)



```python
import jax
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
jax.config.update("jax_enable_x64", True)


class A():
    def __init__(self, a: jnp.array)->None:
        self.a = a
        self.Init()
        
    def Init(self)->None:
        self.b = None
        
    def set_b(self, x):
        self.b = x
    
    @partial(jit, static_argnums=(0,))
    def f(self, var: float)->float:
        b = self.a * var
        return b

objA = A(jnp.array([2.0]))
print("1)",objA.a, objA.b)

b = objA.f(10.)
print("2)",b)
objA.set_b(b)
print("3)",objA.a, objA.b)

new_objA = A(jnp.array([3.0]))
print("4)",objA.a, objA.b)
print("5)",new_objA.a, new_objA.b)

new_b = new_objA.f(20.)
new_objA.set_b(new_b)
print("6)",objA.a, objA.b)
print("7)",new_objA.a, new_objA.b)
```
```python
1) [2.] None
2) [20.]
3) [2.] [20.]
4) [2.] [20.]
5) [3.] None
6) [2.] [20.]
7) [3.] [60.]
```

문제

```python
obj = A(1)
print(obj.f(2))
# 2

obj.a = 2
print(obj.f(2))  # should print 4, but prints 2
# 2
```

답변

```python
class A():
    def __init__(self, a: jnp.array)->None:
        self.a = a
        self.Init()
        
    def Init(self)->None:
        self.b = None
        
    def set_b(self, x):
        self.b = x
    
    @jit
    def f(self, var: float)->float:
        b = self.a * var
        return b

    def _tree_flatten(self):
      # You might also want to store self.b in either the first group
      # (if it's not hashable) or the second group (if it's hashable)
      return (self.a,), ()

    @classmethod
    def _tree_unflatten(cls, aux, children):
      return cls(*children)

tree_util.register_pytree_node(A, A._tree_flatten, A._tree_unflatten)

obj = A(1)
print(obj.f(2))
# 2

obj.a = 2
print(obj.f(2))
# 4
```

# 2

* 출처 : [https://github.com/google/jax/issues/1567](https://github.com/google/jax/issues/1567)

질문

```python
import jax.numpy as np
from jax import jit
from functools import partial


class World:
    def __init__(self, p, v):
        self.p = p
        self.v = v

    @partial(jit, static_argnums=(0,))
    def step(self, dt):
        a = - 9.8
        self.v += a * dt
        self.p += self.v *dt


world = World(np.array([0, 0]), np.array([1, 1]))

for i in range(1000):
    world.step(0.01)
print(world.p)
```

답변

```python
import jax.numpy as np
from jax import jit
from collections import namedtuple

World = namedtuple("World", ["p", "v"])

@jit
def step(world, dt):
  a = -9.8
  new_v = world.v + a * dt
  new_p = world.p + new_v * dt
  return World(new_p, new_v)

world = World(np.array([0, 0]), np.array([1, 1]))

for i in range(1000):
  world = step(world, 0.01)
print(world.p)
```

```python
from jax.tree_util import register_pytree_node
from functools import partial

class World:
  def __init__(self, p, v):
    self.p = p
    self.v = v

  @jit
  def step(self, dt):
    a = -9.8
    new_v = self.v + a * dt
    new_p = self.p + new_v * dt
    return World(new_p, new_v)

# By registering 'World' as a pytree, it turns into a transparent container and
# can be used as an argument to any JAX-transformed functions.
register_pytree_node(World,
                     lambda x: ((x.p, x.v), None),
                     lambda _, tup: World(tup[0], tup[1]))


world = World(np.array([0, 0]), np.array([1, 1]))

for i in range(1000):
  world = world.step(0.01)
print(world.p)
```

