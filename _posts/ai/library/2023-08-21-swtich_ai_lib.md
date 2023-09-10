---
layout: single
title: "tensorflow ↔ pytorch"
date: 2023-08-21 17:24:57
lastmod : 2023-09-10 18:14:00
categories: AI
tag: [Pytorch, Tensorflow]
toc: true
toc_sticky: true
---

## Tensor

### 1

```python
test1_pth = torch.tensor([[1., 2.], [3., 4.]])
```

```python
>>> test1_pth
tensor([[1., 2.],
        [3., 4.]])
```

```python
test1_tf = tf.constant([[1., 2.], [3., 4.]])
```

```python
>>> test1_tf
tf.Tensor([2. 2.], shape=(2,), dtype=float32)
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[1., 2.],
       [3., 4.]], dtype=float32)>
```

### 2

```python
test2_pth = torch.tensor([[1., 2.], [3., 4.]])
test2_pth[0][0] = 9.
```

```python
>>> test2_pth
tensor([[9., 2.],
        [3., 4.]])
```

```python
test2_tf = tf.constant([[1., 2.], [3., 4.]])
test2_tf = tf.Variable(test2_tf)
test2_tf[0, 0].assign(9.)
```

```python
>>> test2_tf
<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
array([[9., 2.],
       [3., 4.]], dtype=float32)>
```

### 3

```python
test3_pth = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
```

```python
>>> test3_pth
tensor([[1., 2.],
        [3., 4.]], requires_grad=True)
```

```python
test3_tf = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
test3_tf = tf.Variable(test3_tf, trainable=True)
```

```python
>>> test3_tf
<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
array([[1., 2.],
       [3., 4.]], dtype=float32)>
```

## Weight initialization

### 1.

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

### 1

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

```python
import tensorflow as tf

x = tf.Variable(0.)
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)

print(grad_of_y_wrt_x)
# tf.Tensor(2.0, shape=(), dtype=float32)
```

### 2

```python
W = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
b = torch.tensor([[2., 1.], [1., 2.]], requires_grad=True)
x = torch.tensor([[4., 5.], [6., 7.]], requires_grad=True)

y = x.matmul(W) + b

y.backward(torch.ones_like(y))
```

```python
>>> b.grad, W.grad
(tensor([[1., 1.],
         [1., 1.]]),
 tensor([[10., 10.],
         [12., 12.]]))
```

```python
W = tf.Variable(tf.constant([[1., 2.], [3., 4.]]))
b = tf.Variable(tf.constant([[2., 1.], [1., 2.]]))
x = tf.Variable(tf.constant([[4., 5.], [6., 7.]]))

with tf.GradientTape() as tape:
    y = tf.matmul(x, W) + b

grad_of_y_wrt_W_and_b = tape.gradient(y, [b, W])
```

```python
>>> grad_of_y_wrt_W_and_b
[<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[1., 1.],
        [1., 1.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[10., 10.],
        [12., 12.]], dtype=float32)>]x
```

### 3

```python
import torch

t_c = torch.tensor([4])
t_u = torch.tensor([3])
learning_rate = 0.01

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

params = torch.tensor([2.0, 7.0], requires_grad=True)
loss = loss_fn(model(t_u, *params), t_c)
loss.backward()

print(params.grad)
# tensor([54., 18.])
```

```python
import tensorflow as tf

t_c = tf.constant([4.0])
t_u = tf.constant([3.0])
learning_rate = 0.01

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = tf.square(t_p - t_c)
    return tf.reduce_mean(squared_diffs)

params = tf.Variable([2.0, 7.0], dtype=tf.float32)

with tf.GradientTape() as tape:
    t_p = model(t_u, *params)
    loss = loss_fn(t_p, t_c)

grads = tape.gradient(loss, params)
print(grads)
# tf.Tensor([54. 18.], shape=(2,), dtype=float32)
```

## Model

### 1

```python
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)
```

```python
import tensorflow as tf

class QNetwork(tf.keras.Model):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        self.network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(np.array(env.single_observation_space.shape).prod(),)),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(env.single_action_space.n)
        ])

    def call(self, x):
        return self.network(x)
```

```python
import tensorflow as tf

def create_q_network(env):
    input_shape = (np.array(env.single_observation_space.shape).prod(),)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(120, activation='relu')(inputs)
    x = tf.keras.layers.Dense(84, activation='relu')(x)
    outputs = tf.keras.layers.Dense(env.single_action_space.n)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
```

### 2

```python
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias
```

```python
import tensorflow as tf

class Actor(tf.keras.Model):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc_mu = tf.keras.layers.Dense(np.prod(env.single_action_space.shape), activation='tanh')

        # action rescaling
        action_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_bias = (env.action_space.high + env.action_space.low) / 2.0
        self.action_scale = self.add_weight(name='action_scale', shape=action_scale.shape, initializer=tf.constant_initializer(action_scale), trainable=False)
        self.action_bias = self.add_weight(name='action_bias', shape=action_bias.shape, initializer=tf.constant_initializer(action_bias), trainable=False)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_mu(x)
        actions = x * self.action_scale + self.action_bias
        return actions

```

## BackPropagation

### 1

```python
import torch
import torch.optim as optim

learning_rate = 0.1

x = torch.tensor([2.0])
y = torch.tensor([3.0])

params = torch.tensor([1.0, 3.0], requires_grad=True)

optimizer = optim.SGD([params], lr=learning_rate)

result = x * params[0] + params[1]

loss = (result - y) ** 2

print(f"loss: {loss}")

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"params: {params}")
print(f"params.grad: {params.grad}")
```

```text
loss: tensor([4.], grad_fn=<PowBackward0>)
params: tensor([0.2000, 2.6000], requires_grad=True)
params.grad: tensor([8., 4.])
```

```python
import tensorflow as tf

learning_rate = 0.1

x = tf.constant([2.0])
y = tf.constant([3.0])

params = tf.Variable([1.0, 3.0], trainable=True)

optimizer = tf.optimizers.SGD(learning_rate)

with tf.GradientTape() as tape:
    result = x * params[0] + params[1]
    loss = tf.reduce_mean((result - y) ** 2)

print(f"loss: {loss.numpy()}")

gradients = tape.gradient(loss, [params])
optimizer.apply_gradients(zip(gradients, [params]))

print(f"params: {params.numpy()}")
print(f"params_grad: {gradients[0].numpy()}")
```

```text
loss: 4.0
params: [0.19999999 2.6       ]
params_grad: [8. 4.]
```

## gather

### 1.


```python
import torch

test = torch.tensor([[1, 2], [3, 4], [5, 6]])
a = torch.tensor([[0], [0], [1]])

print(test.gather(1, a))
```

```
tensor([[1],
        [3],
        [6]])
```

```python
import tensorflow as tf

test = tf.constant([[1, 2], [3, 4], [5, 6]])
a = tf.constant([[0], [0], [1]])

print(tf.experimental.numpy.take_along_axis(test, a, axis=1))
```

```
tf.Tensor(
[[1]
 [3]
 [6]], shape=(3, 1), dtype=int32)
```

<!--
###

```python

```

```python
>>>

```

```python

```

```python
>>>

```
 -->
