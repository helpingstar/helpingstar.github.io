---
layout: single
title: "Android Compose UI"
date: 2025-12-26 23:00:00
lastmod : 2025-12-26 23:00:00
categories: android
tag: [Android, Compose]
toc: true
toc_sticky: true
published: true
---



## `remember`

Composable function은 `remember` API를 사용하여 객체를 메모리에 저장할 수 있다. `remember`로 계산된 값은 initial composition 동안 Composition에 저장되며, recomposition시에는 저장되어 있던 값이 다시 반환된다. `remember`는 mutable 객체와 immutable 객체 모두를 저장하는 데 사용할 수 있다.

> [!NOTE]
> `remember`를 호출한 composable이 Composition에서 제거되면 해당 객체(`remember`의 계산 결과) 를 잊어버린다.

`mutableStateOf`는 관찰 가능한 `MutableState<T>`를 생성하며, 이는 Compose runtime과 통합된 observable type이다.

**Compose runtime과 통합되었다는 것의 의미**
- Compose의 Snapshot 시스템에 등록된 값
- runtime이 읽기/쓰기를 추적 가능
- recomposition의 기준이 된다.

```kotlin
interface MutableState<T> : State<T> {
    override var value: T
}
```

`value`에 어떤 변경이든 발생하면, `value`를 읽는 모든 composable function의 recomposition이 스케줄된다.

composable에서 `MutableState` 객체를 선언하는 방법은 세 가지가 있다.

```kotlin
val mutableState = remember { mutableStateof(default) }
var value by remember { mutableStateOf(default) }
val (value, setValue) = remember { mutableStateOf(default) }
```

이 선언들은 모두 equivalent하다.

`by` delegate syntax를 사용하려면 다음 import가 필요하다.

```kotlin
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
```

`remember`로 저장된 값을 다른 composable의 parameter로 사용할 수도 있고, 어떤 composable을 표시할지 결정하는 로직으로도 사용할 수도 있다.

`remember`는 recomposition 간에 state를 유지하는 데 도움을 주지만, configuration change가 발생하면 state는 유지되지 않는다. 이를 위해서는 `rememberSaveable`을 사용해야 한다.

`remember`를 사용해 객체를 저장하는 composable은 internal state를 가지게 되며 이로 인해 해당 composable은 stateful이 된다.

이 방식은 호출자가 state를 직접 제어할 필요가 없고, state 관리를 신경 쓰지 않아도 되는 상황에서는 유용할 수 있다. 하지만 internal state를 가진 composable은 재사용성이 떨어지고 테스트하기가 더 어려운 경향이 있다.

stateless composable은 어떤 state도 보유하지 않는 composable을 의미한다. stateless composable을 만드는 가장 쉬운 방법 중 하나는 state hoisting을 사용하는 것이다.

## `rememberSaveable`

`rememberSaveable` API는 recomposition 간에 state를 유지한다는 점에서 `remember`와 유사하지만, saved instance state 메커니즘을 사용하여 activity 또는 process가 재생성되는 경우에도 state를 유지한다는 점이 다르다.

> [!NOTE]
> `rememberSaveable`은 activity가 사용자에 의해 완전히 종료(dismissed) 된 경우에는 state를 유지하지 않는다.

```kotlin
// android/nowinandroid
@Composable
fun NiaApp( ... ) {
    var showSettingsDialog by rememberSaveable { mutableStateOf(false) }
}
```

- `showSeettingsDialog`를 읽고 쓸 때마다 내부의 `MutableState.value`에 접근한다. `rememberSaveable`이 이 `MutableState`를 Configuration Change에서도 보존한다.
- `remember`를 쓰게 된다면 사용자가 setting버튼을 눌러서 `showSettingsDialog`를 true로 만든 후에 화면 회전이 일어나면 새 Activity가 생성되어 새로운 Composition이 시작되고 `remember { mutableStateOf(false) }` 코드가 다시 실행되어 초기값인 `false`로 시작한다.

### `key`

```kotlin
var name by remember { mutableStateOf("") }
```

일반적으로 `remember`는 `calculation` lambda parameter를 받는다. `remember`가 처음 실행될 때는 `calculation` lambda를 호출하고, 그 결과를 저장한다. 이후 recomposition 동안에는, `remember`는 마지막으로 저장된 값을 반환한다.

`state`를 캐싱하는 것 외에도, `remember`는 초기화하거나 계산하는 비용이 큰 객체나 연산 결과를 Composition에 저장하는 데 사용할 수 있다. 이러한 계산을 매 recomposition 마다 반복하고 싶지 않은 경우 유용하다.

`remember` API는 `key` 또는 `keys` parameter를 함께 받을 수 있다. 이 key들 중 하나라도 변경되면, 다음에 function이 recomposition될 때 `remember`는 기존 캐시를 무효화(invalidate) 하고 `calculation` lambda 블록을 다시 실행한다.

이 메커니즘을 통해 Composition 안에서 객체의 수명(liftime)을 제어할 수 있다. 즉, 계산 결과는 remembered value가 Composition에서 제거될 때까지가 아니라, 입력값이 변경될 때까지 유효하게 유지된다.

```kotlin
MainActivity {
    rememberNiaAppState {
        rememberNavigationState

    }
}
```

## `derivedStateOf`

Compose에서는 관찰 중인 state 객체나 composable input이 변경될 때마다 recomposition이 발생한다. 그런데 state객체나 input이 실제로 UI 업데이트가 필요하지 않은 빈도로 결정되는 경우가 있으며, 이로 인해 불필요한 recomposition이 발생할 수 있다.

composable에 전달되는 input이 recomposition이 필요한 빈도보다 더 자주 변경되는 경우에는 `derivedStateOf` 함수를 사용해야 한다.

`derivedStateOf`는 비용이 큰 연산이므로, 결과가 변경되지 않았을 때 불필요한 recomposition을 피하기 위한 경우에만 사용해야 한다.


## 참조
- https://developer.android.com/develop/ui/compose/side-effects


<!-- TODO
- https://medium.com/androiddevelopers/jetpack-compose-when-should-i-use-derivedstateof-63ce7954c11b
- https://developer.android.com/topic/architecture/ui-layer/stateholders#choose_between_a_viewmodel_and_plain_class_for_a_state_holder
- https://developer.android.com/develop/ui/compose/state-hoisting#plain-state 
- `State` 객체란?

import androidx.compose.foundation.layout.WindowInsets

consumeWindowInsets
-->