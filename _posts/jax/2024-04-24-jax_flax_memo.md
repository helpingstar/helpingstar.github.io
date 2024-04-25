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
* `optax.inject_hyperparams`를 통해 `GradientTransformation`(ex. `optax.adam`)를 Wrapping 하면 하이퍼파라미터에 직접 접근이 가능해진다. 그런 optimizer state 정보는 `TrainState.opt_state`에 저장되기 때문에 `TrainState.opt_state.hyperparams[<name of parameter>]`를 통해 optimizer의 state의 hyperparameter에 접근이 가능하다. 
  * `TrainState.tx` 가 `optax.chain`을 통해 여러 개의 `GradientTransformation`으로 이루어져 있다면 인덱스로 접근 후 `.hyperparams`를 통해 접근해야 한다.
    * ex. `agent_state.opt_state[1].hyperparams["learning_rate"]`
  * `TrainState.tx.opt_state`로 접근할 수 있다고 생각하기보다는 `TrainState`는 `apply_gradient`, `init`의 과정에서 `opt_state`를 자동으로 내부에 저장하는데 그것에 접근하는 것이라 생각하는 것이 좋을 것 같다.

ex. `TrainState`의 `optimizer`(`tx`)의 learning_rate를 얻기

```python
# learning_rate를 얻으려는 경우 learning_rate는 schedule일 확률이 높다.
tx = optax.inject_hyperparams(optax.adam)(learning_rate=scalar_or_schedule)

train_state = CustomTrainState.create(
  ...
  tx=tx,
)

...

lr_adam = train_state.opt_state.hyperparams["learning_rate"]
```

<!-- TODO : struct.field(pytree_node=True)??? -->
<!-- TODO : update -> apply_updates 원리 -->