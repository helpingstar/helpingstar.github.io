---
layout: single
title: "자동 미분 계산 그래프 연습"
date: 2023-09-19 13:30:30
lastmod : 2023-09-19 13:30:30
categories: dl
tag: [auto diffrentiation, computationn graph]
toc: true
toc_sticky: true
---

자동 미분 계산 그래프를 연습한다.

## 1.

```py
learning_rate = 0.1

x = torch.tensor([2.0])
y = torch.tensor([3.0])

params = torch.tensor([1.0, 3.0], requires_grad=True)

result = x * params[0] + params[1]

loss = (result - y) ** 2

loss.backward()

print(f"loss: {loss}")
print(f"params.grad: {params.grad}")
```

```text
loss: tensor([4.], grad_fn=<PowBackward0>)
params.grad: tensor([8., 4.])
```

![auto_diff_1](../../assets/images/ai/auto_diff_1.jpg)

## 2.

```py
learning_rate = 0.1

x = torch.tensor([2.0])
y = torch.tensor([3.0])

params = torch.tensor([3.0], requires_grad=True)

a = x * params
b = y * params

loss = a * b

loss.backward()

print(f"loss: {loss}")
print(f"params.grad: {params.grad}")
```

```text
loss: tensor([54.], grad_fn=<MulBackward0>)
params.grad: tensor([36.])
```

```py
learning_rate = 0.1

x = torch.tensor([2.0])
y = torch.tensor([3.0])

params = torch.tensor([3.0], requires_grad=True)

a = x * params
b = y * params.detach()

loss = a * b

loss.backward()

print(f"loss: {loss}")
print(f"params.grad: {params.grad}")
```

```text
loss: tensor([54.], grad_fn=<MulBackward0>)
params.grad: tensor([18.])
```

![auto_diff_2](../../assets/images/ai/auto_diff_2.jpg)

## 3.

```py
learning_rate = 0.1

x = torch.tensor([7.0])

params = torch.tensor([2.0, 3.0, 5.0, 4.0], requires_grad=True)

a = x * params[0]
b = a * params[1]
r = b * params[2]
s = b * params[3]
t = r + s

t.backward()

print(f"r: {r}, s: {s}")
print(f"params.grad: {params.grad}")
```

```text
r: tensor([210.], grad_fn=<MulBackward0>), s: tensor([168.], grad_fn=<MulBackward0>)
params.grad: tensor([189., 126.,  42.,  42.])
```

![auto_diff_3](../../assets/images/ai/auto_diff_3.jpg)