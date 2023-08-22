---
layout: single
title: "tensorflow ↔ pytorch"
date: 2023-08-21 17:24:57
lastmod : 2023-08-22 17:04:15
categories: AI
tag: [Pytorch, Tensorflow]
toc: true
toc_sticky: true
---

## Weight initialization

### Pytorch

```python
import torch
import torch.nn as nn

def layer_init(layer, w_a=0, w_b=1, bias_const=-1):
    torch.nn.init.uniform_(layer.weight, a=w_a, b=w_b)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

test1 = layer_init1(nn.Linear(3, 4))
```

```python
>>> test1._parameters
OrderedDict([('weight',
              Parameter containing:
              tensor([[0.4316, 0.7005, 0.3997],
                      [0.0089, 0.4746, 0.2912],
                      [0.1438, 0.6648, 0.1226],
                      [0.4291, 0.0352, 0.6135]], requires_grad=True)),
             ('bias',
              Parameter containing:
              tensor([-1., -1., -1., -1.], requires_grad=True))])
```

### Tensorflow

```python
import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform, Constant

test2 = tf.keras.models.Sequential()
test2.add(tf.keras.Input(shape=(3,)))
test2.add(
    tf.keras.layers.Dense(
        4,
        activation="relu",
        kernel_initializer=RandomUniform(0, 1),
        bias_initializer=Constant(-1),
    )
)
```

```python
>>> test2.weights
[<tf.Variable 'dense/kernel:0' shape=(3, 4) dtype=float32, numpy=
 array([[0.792158  , 0.76683366, 0.71274936, 0.3051616 ],
        [0.22424495, 0.65580904, 0.06704581, 0.2954831 ],
        [0.7401217 , 0.24613738, 0.8886342 , 0.7413529 ]], dtype=float32)>,
 <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([-1., -1., -1., -1.], dtype=float32)>]
```

## Gradient

### Pytorch

```python
x = torch.tensor(0.0, requires_grad=True)
y = 2 * x + 3

y.backward()
grad_of_y_wrt_x = x.grad

print(grad_of_y_wrt_x)
# tensor(2.)
```

```python
x = torch.tensor(0., requires_grad=True)

y = 2 * x + 3
grad_of_y_wrt_x = torch.autograd.grad(y, x)[0]

print(grad_of_y_wrt_x)
# tensor(2.)
```

### Tensorflow

```python
import tensorflow as tf

x = tf.Variable(0.)
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)

print(grad_of_y_wrt_x)
# tf.Tensor(2.0, shape=(), dtype=float32)
```
