---
layout: single
title: "JAX 라이브러리 공부 메모"
date: 2024-04-24 18:45:21
lastmod : 2024-04-24 18:45:21
categories: jax
tag: [jax, flax]
use_math: false
---

## TrainState

`...` 표기한 부분은 코드 일부를 생략했다는 뜻이다.

```python
class TrainState(struct.PyTreeNode):

  ...

  step: Union[int, jax.Array]
  apply_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads, **kwargs):
    
    ...

    else:
      grads_with_opt = grads
      params_with_opt = self.params

    updates, new_opt_state = self.tx.update(
      grads_with_opt, self.opt_state, params_with_opt
    )
    new_params_with_opt = optax.apply_updates(params_with_opt, updates)

    ...

    else:
      new_params = new_params_with_opt
    return self.replace(
      step=self.step + 1,
      params=new_params,
      opt_state=new_opt_state,
      **kwargs,
    )


  @classmethod
  def create(cls, *, apply_fn, params, tx, **kwargs):
    """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
    # We exclude OWG params when present because they do not need opt states.
    params_with_opt = (
      ... else params
    )
    opt_state = tx.init(params_with_opt)
    return cls(
      step=0,
      apply_fn=apply_fn,
      params=params,
      tx=tx,
      opt_state=opt_state,
      **kwargs,
    )
```
* `apply_gradients`
  * `apply_gradient`를 호출하면 새로운 파라미터와(`new_params`), optimizer state(`new_opt_state`)를 얻는다 그리고 `step`을 1 증가시킨 후에 `step`, `params`, `opt_state`를 `replace` 한다.
* `create`
  * `tx.init`에서 반환된 optimizer state(`opt_state`)를 저장한다.
  * 클래스 변수 `step`을 0으로 저장하고, 클래스 변수 `opt_state`에 `tx.init`로부터 얻은 `opt_state`를 저장하고 각 클래스 변수에 각 인자를 대입한 `TrainState`를 반환한다.

<!-- TODO : struct.field(pytree_node=True)??? -->
<!-- TODO : update -> apply_updates 원리 -->