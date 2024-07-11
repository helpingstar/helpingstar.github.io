---
layout: single
title: "JAX 시행착오"
date: 2024-07-11 13:54:14
lastmod : 2024-07-11 13:54:14
categories: jax
tag: [jax, flax]
use_math: false
---

## 1.

`jnp.inner`과 `vmap`이 같이 활용되다보니 헷갈린 예시 (아래 예시는 최대한 간단히 재구성함)

`ndim`이 2인 두 행렬을 계산한다고 가정하자 예를 들어 아래와 같다.

```python
test1 = jnp.arange(0, 6).reshape(3, 2)  # (3, 2)
test2 = jnp.arange(1, 7).reshape(3, 2)  # (3, 2)
```

여기서 아래 두 연산은 `test1`, `test2`에 대해 동작한다.

1. `jnp.inner`
2. `jax.vmap(jnp.inner)`

하지만 결과는 각각 다르다.

```python
print(jnp.inner(test1, test2).shape)  # (3, 3)
print(jax.vmap(jnp.inner)(test1, test2).shape)  # (3,)
```

두번째 `jax.vmap(jnp.inner)`은 `axis`를 따로 설정하지 않을 경우 `axis=0`을 배치라고 생각하고 연산하기 때문에 `test1`, `test2`는 `axis=0` 축의 같은 인덱스에 있는 벡터끼리 `jnp.inner` 연산을 하는 것과 같다.

때문에 2번째 연산의 경우 `[ 2, 18, 50]` 의 결과가 나오는데 이것은

```
 test1   test2
[[0 1]  [[1 2]
 [2 3]   [3 4]
 [4 5]]  [5 6]]
```

위와 같은 연산에서

`[0*1 + 1*2, 2*3 + 3*4, 4*5 + 5 * 6]` 과 같다.

Einstein Summation Convention 으로 표현하면 각 연산은 다음과 같다.

아래 연산은 `a.shape == b.shape == (3, 2)` 를 기준으로 한다.
1. `jnp.inner` : `'ij,kj->ik'`
2. `jax.vmap(jnp.inner)` : `'ij,ij->i'`