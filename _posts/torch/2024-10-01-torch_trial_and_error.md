---
layout: single
title: "torch 시행착오/패턴"
date: 2024-07-11 13:54:14
lastmod : 2024-07-11 13:54:14
categories: torch
tag: [torch]
use_math: true
published: false
---

## 1. `torch.tenosr`는 `grad` 를 제거한다.


```python
import torch

tensor_list = [
    torch.tensor([1.0], requires_grad=True),
    torch.tensor([2.0], requires_grad=True),
    torch.tensor([3.0], requires_grad=True),
]

r1 = torch.stack(tensor_list)
r2 = torch.tensor(tensor_list).reshape(3, 1)
r3 = torch.Tensor(tensor_list).reshape(3, 1)

sr1 = torch.sum(r1)
sr2 = torch.sum(r2)
sr3 = torch.sum(r3)

print(sr1)  # tensor(6., grad_fn=<SumBackward0>)
print(sr2)  # tensor(6.)
print(sr3)  # tensor(6.)
```