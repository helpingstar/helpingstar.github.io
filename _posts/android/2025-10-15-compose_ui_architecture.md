---
layout: single
title: "Compose Ui Architecture"
date: 2025-10-10 11:00:00
lastmod : 2025-10-10 11:00:00
categories: android
tag: [android, coroutine]
toc: true
toc_sticky: true
published: false
---


# Side-effects in Compose

A **side-effect** is a change to the state of the app that happens outside the scope of a composable function. Due to composables' lifecycle and properties such as unpredictable recompositions, executing recompositions of composables in different orders, or recompositions that can be discarded, composables [should ideally be side-effect free](https://developer.android.com/develop/ui/compose/mental-model).

However, sometimes side-effects are necessary, for example, to trigger a one-off event such as showing a snackbar or navigate to another screen given a certain state condition. These actions should be called from a controlled environment that is aware of the lifecycle of the composable. In this page, you'll learn about the different side-effect APIs Jetpack Compose offers.

## State and effect use cases

As covered in the [Thinking in Compose](https://developer.android.com/develop/ui/compose/mental-model) documentation, composables should be side-effect free. When you need to make changes to the state of the app, **you should use the Effect APIs so that those side effects are executed in a predictable manner**.

> **Key Term:** An **effect** is a composable function that doesn't emit UI and causes side effects to run when a composition completes.

Due to the different possibilities effects open up in Compose, they can be easily overused. Make sure that the work you do in them is UI related and doesn't break **unidirectional data flow**.

> **Note:** A responsive UI is inherently asynchronous, and Jetpack Compose solves this by embracing coroutines at the API level instead of using callbacks.

### `LaunchedEffect`: run suspend functions in the scope of a composable

To perform work over the life of a composable and have the ability to call suspend functions, use the [`LaunchedEffect`](https://developer.android.com/reference/kotlin/androidx/compose/runtime/package-summary#LaunchedEffect\(kotlin.Any,kotlin.coroutines.SuspendFunction1\)) composable. When `LaunchedEffect` enters the Composition, it launches a coroutine with the block of code passed as a parameter. The coroutine will be cancelled if `LaunchedEffect` leaves the composition. If `LaunchedEffect` is recomposed with different keys, the existing coroutine will be cancelled and the new suspend function will be launched in a new coroutine.

For example, here is an animation that pulses the alpha value with a configurable delay:

```kotlin
// Allow the pulse rate to be configured, so it can be sped up if the user is running
// out of time
// 사용자가 시간이 부족할 때 이 값을 줄여서 더 빠르게 깜빡이도록 할 수 있다
var pulseRateMs by remember { mutableLongStateOf(3000L) }
val alpha = remember { Animatable(1f) }
LaunchedEffect(pulseRateMs) { // Restart the effect when the pulse rate changes
    while (isActive) {
        delay(pulseRateMs) // Pulse the alpha every pulseRateMs to alert the user
        alpha.animateTo(0f)
        alpha.animateTo(1f)
    }
}
```

In the code above, the animation uses the suspending function [`delay`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/delay.html) to wait the set amount of time. Then, it sequentially animates the alpha to zero and back again using [`animateTo`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/Animatable#animateTo\(kotlin.Any,androidx.compose.animation.core.AnimationSpec,kotlin.Any,kotlin.Function1\)). This will repeat for the life of the composable.

### `rememberCoroutineScope`: obtain a composition-aware scope to launch a coroutine outside a composable

- `LaunchedEffect`는 composable 함수이기 때문에, 다른 composable 함수 내부에서만 사용가능
- composable 외부에서 코루틴을 실행하되, composition을 벗어나면 자동으로 취소되도록 범위를 지정할 때 사용
- 하나 이상의 코루틴의 생명주기를 수동으로 제어해야 할 때 사용
  - ex) 사용자 이벤트가 발생했을 때 애니메이션을 취소하는 경우
- composable 함수로 그것이 호출된 부분에 바인딩된 `CoroutineScope`를 반환.
  - 이 scope는 호출이 Composition을 벗어날때 취소.

Following the previous example, you could use this code to show a `Snackbar` when the user taps on a `Button`:

```kotlin
@Composable
fun MoviesScreen(snackbarHostState: SnackbarHostState) {

    // Creates a CoroutineScope bound to the MoviesScreen's lifecycle
    val scope = rememberCoroutineScope()

    Scaffold(
        snackbarHost = {
            SnackbarHost(hostState = snackbarHostState)
        }
    ) { contentPadding ->
        Column(Modifier.padding(contentPadding)) {
            Button(
                onClick = {
                    // Create a new coroutine in the event handler to show a snackbar
                    scope.launch {
                        snackbarHostState.showSnackbar("Something happened!")
                    }
                }
            ) {
                Text("Press me")
            }
        }
    }
}
```

### `rememberUpdatedState`: reference a value in an effect that shouldn't restart if the value changes

- 일부 상황에서 effect 내에서 값을 캡처하되, 그 값이 변경되더라도 effect가 재시작되지 않기를 원할 수 있습니다. 
- 이를 위해서는 `rememberUpdatedState`를 사용하여 캡처되고 업데이트될 수 있는 이 값에 대한 참조를 생성해야 합니다.
- 재생성하고 재시작하기에 비용이 많이 들거나 불가능한 장기 실행 작업을 포함하는 effect에 유용

예를 들어, 앱에 일정 시간 후 사라지는 `LandingScreen`이 있다고 가정해봅시다. `LandingScreen`이 재구성되더라도, 일정 시간을 기다리고 시간이 경과했음을 알리는 effect는 재시작되지 않아야 합니다:

```kotlin
@Composable
fun LandingScreen(onTimeout: () -> Unit) {

    // This will always refer to the latest onTimeout function that
    // LandingScreen was recomposed with
    val currentOnTimeout by rememberUpdatedState(onTimeout)

    // Create an effect that matches the lifecycle of LandingScreen.
    // If LandingScreen recomposes, the delay shouldn't start again.
    LaunchedEffect(true) {
        delay(SplashWaitTimeMillis)
        currentOnTimeout()
    }

    /* Landing screen content */
}
```

1. `rememberUpdatedState(onTimeout)`
   - `onTimeout` 파라미터의 최신 값을 항상 추적하는 State를 생성
   - 재구성 시 `onTimeout`이 변경되면, `currentOnTimeout`의 값만 업데이트됩니다
   - `Effect`는 재시작되지 않습니다
2. `LaunchedEffect(true)`
   - key가 `true` (불변 값) → 절대 재시작되지 않음
   - `LandingScreen`이 처음 Composition에 진입할 때 한 번만 실행됩니다
   - `LandingScreen`이 Composition에서 제거될 때만 취소됩니다

- `LaunchedEffect(true)`에서 `onTimeout`를 쓰지 않는 이유
  - 그렇지 않으면 `LaunchedEffect`가 시작될때의 값으로 고정된다.
  - 코루틴은 시작 시점에 주변 변수를 캡처

To create an effect that matches the lifecycle of the call site, a never-changing constant like `Unit` or `true` is passed as a parameter. In the code above, `LaunchedEffect(true)` is used. To make sure that the `onTimeout` lambda _always_ contains the latest value that `LandingScreen` was recomposed with, `onTimeout` needs to be wrapped with the `rememberUpdatedState` function. The returned `State`, `currentOnTimeout` in the code, should be used in the effect.

호출 지점의 생명주기와 일치하는 effect를 생성하려면, `Unit` 또는 `true`와 같이 절대 변경되지 않는 상수를 파라미터로 전달합니다. 위 코드에서는 `LaunchedEffect(true)`가 사용되었습니다. `onTimeout` 람다가 _항상_ `LandingScreen`이 재구성된 최신 값을 포함하도록 하려면, `onTimeout`을 `rememberUpdatedState` 함수로 감싸야 합니다. 반환된 `State`, 즉 코드에서 `currentOnTimeout`을 effect 내에서 사용해야 합니다.

> **Warning:** `LaunchedEffect(true)` is as suspicious as a `while(true)`. Even though there are valid use cases for it, _always_ pause and make sure that's what you need.

### `DisposableEffect`: effects that require cleanup

키가 변경된 후 또는 composable이 Composition을 벗어날 때 정리가 필요한 side effect의 경우, `DisposableEffect`를 사용하세요. `DisposableEffect` 키가 변경되면, composable은 현재 effect를 dispose(정리)하고, effect를 다시 호출하여 재설정해야 합니다.

As an example, you might want to send analytics events based on [`Lifecycle` events](https://developer.android.com/topic/libraries/architecture/lifecycle#lc) by using a [`LifecycleObserver`](https://developer.android.com/reference/androidx/lifecycle/LifecycleObserver). To listen for those events in Compose, use a `DisposableEffect` to register and unregister the observer when needed.

```kotlin
@Composable
fun HomeScreen(
    lifecycleOwner: LifecycleOwner = LocalLifecycleOwner.current,
    onStart: () -> Unit, // Send the 'started' analytics event
    onStop: () -> Unit // Send the 'stopped' analytics event
) {
    // Safely update the current lambdas when a new one is provided
    val currentOnStart by rememberUpdatedState(onStart)
    val currentOnStop by rememberUpdatedState(onStop)

    // If `lifecycleOwner` changes, dispose and reset the effect
    DisposableEffect(lifecycleOwner) {
        // Create an observer that triggers our remembered callbacks
        // for sending analytics events
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_START) {
                currentOnStart()
            } else if (event == Lifecycle.Event.ON_STOP) {
                currentOnStop()
            }
        }

        // Add the observer to the lifecycle
        lifecycleOwner.lifecycle.addObserver(observer)

        // When the effect leaves the Composition, remove the observer
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    /* Home screen content */
}
```

In the code above, the effect will add the `observer` to the `lifecycleOwner`. If `lifecycleOwner` changes, the effect is disposed and restarted with the new `lifecycleOwner`.

A `DisposableEffect` must include an `onDispose` clause as the final statement in its block of code.

**Note:** Having an empty block in `onDispose` is not a good practice. Always reconsider to see if there's an effect that fits your use case better.

### `SideEffect`: publish Compose state to non-Compose code

To share Compose state with objects not managed by compose, use the [`SideEffect`](https://developer.android.com/reference/kotlin/androidx/compose/runtime/package-summary#SideEffect\(kotlin.Function0\)) composable. Using a `SideEffect` guarantees that the effect executes after every successful recomposition. On the other hand, it is incorrect to perform an effect before a successful recomposition is guaranteed, which is the case when writing the effect directly in a composable.

Compose는 recomposition을 시작했다가 중간에 취소할 수 있는데, 재구성이 끝까지 완료되고 UI에 실제로 반영되었을 때만 successful 이라고 한다.

다음 예시를 보자

```kotlin
@Composable
fun MyScreen(analytics: AnalyticsService) {
    var count by remember { mutableStateOf(0) }
    
    // ❌ 잘못된 방법: 직접 작성
    analytics.logEvent("count_changed", count)  
    
    Button(onClick = { count++ }) {
        Text("Count: $count")
    }
}
```

위 같은 경우 `analytics.logEvent`가 recomposition 중에 즉시 실행된다. 하지만 recomposition이 중간에 취소될 경우 로그는 이미 전송되었지만 UI는 실제로 업데이트되지 않아 데이터 불일치가 발생할 수 있다. 이럴 때 다음과 같이 `SideEffect`를 사용하여 해결할 수 있다.

```kotlin
@Composable
fun MyScreen(analytics: AnalyticsService) {
    var count by remember { mutableStateOf(0) }
    
    // ✅ 올바른 방법: SideEffect 사용
    SideEffect {
        analytics.logEvent("count_changed", count)
    }
    
    Button(onClick = { count++ }) {
        Text("Count: $count")
    }
}
```

1. Recomposition 시작
2. Composable 함수들이 실행됨
3. 모든 Recomposition이 성공적으로 완료됨
4. `SideEffect` 블록이 실행됨
5. UI가 화면에 반영됨

### `produceState`: convert non-Compose state into Compose state

`produceState`는 
- 반환된 `State`에 값을 푸시할 수 있는 Composition으로 범위가 지정된 코루틴을 실행
  - non-Compose state를 Compose state로 변환 
  - 예를 들어 `Flow`, `LiveData` 또는 `RxJava`와 같은 외부 구독 기반 state를 Composition으로 가져올 수 있다.
- producer는 
  - `produceState`가 Composition에 진입할 때 실행
  - Composition을 벗어날 때 취소
  - 반환된 `State`는 병합(conflated) : 동일한 값을 설정해도 recomposition이 트리거되지 않음

`produceState`가 코루틴을 생성하긴 하지만, non-suspending source of data를 관찰하는 데에도 사용할 수 있습니다. 해당 소스에 대한 구독을 제거하려면 `awaitDispose` 함수를 사용하세요.

다음 예시는 `produceState`를 사용하여 네트워크에서 이미지를 로드하는 방법을 보여줍니다. `loadNetworkImage` composable 함수는 다른 composable에서 사용할 수 있는 `State`를 반환합니다.

```kotlin
@Composable
fun loadNetworkImage(
    url: String,
    imageRepository: ImageRepository = ImageRepository()
): State<Result<Image>> {
    // Creates a State<T> with Result.Loading as initial value
    // If either `url` or `imageRepository` changes, the running producer
    // will cancel and will be re-launched with the new inputs.
    return produceState<Result<Image>>(initialValue = Result.Loading, url, imageRepository) {
        // In a coroutine, can make suspend calls
        val image = imageRepository.load(url)

        // Update State with either an Error or Success result.
        // This will trigger a recomposition where this State is read
        value = if (image == null) {
            Result.Error
        } else {
            Result.Success(image)
        }
    }
}
```

```kotlin
// 초기화
produceState<Result<Image>>(initialValue = Result.Loading, url, imageRepository) { ... }
```
- `Result.Loading` 상태를 가진 `State` 객체를 즉시 반환, 화면에는 로딩 상태가 먼저 표시됨
- `url` : key 1
- `imageRepository` : key 2

```kotlin
// 코루틴 실행
{
    val image = imageRepository.load(url) // 비동기 네트워크 호출
    value = ... // State 업데이트
}
```
- Composition에 진입하면 코루틴이 시작됨
- `suspend` 함수 호출 가능
- `value`를 변경하면 `State`가 업데이트되고 Compose에서 recomposition 트리거

```kotlin
// Key 변경 시 재시작
produceState(..., url, imageRepository)
```

- `url`이나 `imageRepository`가 변경되면 기존 코루틴 취소 및 새로운 코루틴 시작
- Composable이 화면에서 사라지면 코루틴 자동 취소

사용 예시

```kotlin
@Composable
fun ImageScreen(imageUrl: String) {
    // State를 받아옴
    val imageState by loadNetworkImage(imageUrl)
    
    // State에 따라 다른 UI 표시
    when (imageState) {
        Result.Loading -> {
            CircularProgressIndicator()
        }
        Result.Error -> {
            Text("Failed to load image")
        }
        is Result.Success -> {
            Image(bitmap = imageState.image.asBitmap())
        }
    }
}
```

1. `ImageScreen` 표시 -> `loadNetworkImage` 호출
2. `Result.Loading` (초기값) 반환 -> 로딩 표시
3. 백그라운드에서 이미지 로딩 시작
4. 로딩 완료 -> `value` 업데이트 -> recomposition 트리거
5. `Result.Success` -> 이미지 표시


---

- `loadNetworkImage`는 State 값만 변경 (간접적)
- `ImageScreen`이 State를 읽고 있기 때문에 재구성됨 (직접적 구독자)
- Compose의 반응형 시스템이 중간에서 연결해줌
**Note:** Composables with a return type should be named the way you'd name a normal Kotlin function, starting with a lowercase letter.

**Key Point:** Under the hood, `produceState` makes use of other effects! It holds a `result` variable using `remember { mutableStateOf(initialValue) }`, and triggers the `producer` block in a `LaunchedEffect`. Whenever `value` is updated in the `producer` block, the `result` state is updated to the new value.

You can easily create your own effects building on top of the existing APIs.

### `derivedStateOf`: convert one or multiple state objects into another state

In Compose, [recomposition](https://developer.android.com/develop/ui/compose/mental-model#recomposition) occurs each time an observed state object or composable input changes. A state object or input may be changing more often than the UI actually needs to update, leading to unnecessary recomposition.

You should use the `derivedStateOf` function when your inputs to a composable are changing more often than you need to recompose. This often occurs when something is frequently changing, such as a scroll position, but the composable only needs to react to it once it crosses a certain threshold. `derivedStateOf` creates a new Compose state object you can observe that only updates as much as you need. In this way, it acts similarly to the Kotlin Flows [`distinctUntilChanged()`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/distinct-until-changed.html#:%7E:text=Returns%20flow%20where%20all%20subsequent,a%20StateFlow%20has%20no%20effect.) operator.

> **Caution:** `derivedStateOf` is expensive, and you should only use it to avoid unnecessary recomposition when a result hasn't changed.

#### Correct usage

The following snippet shows an appropriate use case for `derivedStateOf`:

```kotlin
@Composable
// When the messages parameter changes, the MessageList
// composable recomposes. derivedStateOf does not
// affect this recomposition.
fun MessageList(messages: List<Message>) {
    Box {
        val listState = rememberLazyListState()

        LazyColumn(state = listState) {
            // ...
        }

        // Show the button if the first visible item is past
        // the first item. We use a remembered derived state to
        // minimize unnecessary compositions
        val showButton by remember {
            derivedStateOf {
                listState.firstVisibleItemIndex > 0
            }
        }

        AnimatedVisibility(visible = showButton) {
            ScrollToTopButton()
        }
    }
}
```

In this snippet, `firstVisibleItemIndex` changes any time the first visible item changes. As you scroll, the value becomes `0`, `1`, `2`, `3`, `4`, `5`, etc. However, recomposition only needs to occur if the value is greater than `0`. This mismatch in update frequency means that this is a good use case for `derivedStateOf`.

#### Incorrect usage

A common mistake is to assume that, when you combine two Compose state objects, you should use `derivedStateOf` because you are "deriving state". However, this is purely overhead and not required, as shown in the following snippet:

**Warning:** The following snippet shows an incorrect usage of `derivedStateOf`. Do not use this code in your project.

```kotlin
// DO NOT USE. Incorrect usage of derivedStateOf.
var firstName by remember { mutableStateOf("") }
var lastName by remember { mutableStateOf("") }

val fullNameBad by remember { derivedStateOf { "$firstName $lastName" } } // This is bad!!!
val fullNameCorrect = "$firstName $lastName" // This is correct
```

In this snippet, `fullName` needs to update just as often as `firstName` and `lastName`. Therefore, no excess recomposition is occurring, and using `derivedStateOf` is not necessary.

### `snapshotFlow`: convert Compose's State into Flows

[`snapshotFlow`](https://developer.android.com/reference/kotlin/androidx/compose/runtime/package-summary#snapshotFlow\(kotlin.Function0\))를 사용하여 [`State<T>`](https://developer.android.com/reference/kotlin/androidx/compose/runtime/State) 객체를 cold Flow로 변환하세요. 

- `snapshotFlow`는 collect될 때 블록을 실행하고 그 안에서 읽은 `State` 객체들의 결과를 방출합니다. 
- `snapshotFlow` 블록 내부에서 읽은 `State` 객체 중 하나가 변경되면, 새 값이 이전에 방출된 값과 [같지 않을](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin/-any/equals.html) 경우 Flow가 collector에게 새 값을 방출(`Flow.distinctUntilChanged`의 동작과 유사).

The following example shows a side effect that records when the user scrolls past the first item in a list to analytics:

```kotlin
val listState = rememberLazyListState()

LazyColumn(state = listState) {
    // ...
}

LaunchedEffect(listState) {
    snapshotFlow { listState.firstVisibleItemIndex }
        .map { index -> index > 0 }
        .distinctUntilChanged()
        .filter { it == true }
        .collect {
            MyAnalyticsService.sendScrolledPastFirstItemEvent()
        }
}
```

In the code above, `listState.firstVisibleItemIndex` is converted to a Flow that can benefit from the power of Flow's operators.


```kotlin
snapshotFlow { listState.firstVisibleItemIndex }
```

- `listState.firstVisibleItemIndex`가 변경될 때마다 새 값을 emit
- Cold Flow이므로 collect할 때만 활성화됨


```kotlin
.collect {
    MyAnalyticsService.sendScrolledPastFirstItemEvent()
}
```
- 사용자가 **첫 번째 아이템을 지나쳐 스크롤할 때마다** 이벤트 전송

#### `snapshotFlow` vs `derivedStateof`

- `snapshotFlow` : State -> Flow
  - Flow 연산자 사용 가능 (map, filter, debounce, etc)
  - 비동기 처리에 적합
  - Side effect 수행 (Analytics, logging, etc)
- `derivedStateOf` : State -> State
  - Compose State로 유지
  - UI 업데이트에 적합
  - Recomposition optimization


## Restarting effects

Some effects in Compose, like `LaunchedEffect`, `produceState`, or `DisposableEffect`, take a variable number of arguments, keys, that are used to cancel the running effect and start a new one with the new keys.  
Compose의 일부 effect(예: `LaunchedEffect`, `produceState`, `DisposableEffect`)는 실행 중인 effect를 취소하고 새로운 keys로 새로운 effect를 시작하기 위해 사용되는 가변 개수의 인자(keys)를 받습니다.


The typical form for these APIs is:

```kotlin
EffectName(restartIfThisKeyChanges, orThisKey, orThisKey, ...) { block }
```

이 동작의 미묘함 때문에, 효과를 재시작하는 데 사용되는 매개변수가 정확하지 않으면 문제가 발생할 수 있습니다: 
- effect를 필요한 것보다 덜 재시작하면 앱에 버그가 생길 수 있습니다. 
- effect를 필요한 것보다 더 자주 재시작하면 비효율적일 수 있습니다.


경험적으로, effect 블록 코드에서 사용되는 가변 및 불변 변수들은 effect composable의 파라미터로 추가되어야 합니다. 이들 외에도, effect를 강제로 재시작하기 위해 추가 파라미터를 더할 수 있습니다. 만약 변수의 변경이 effect를 재시작하지 않아야 한다면, 그 변수는 `rememberUpdatedState`로 감싸야 합니다. 변수가 키 없이 `remember`로 감싸져 있어서 절대 변경되지 않는다면, 그 변수를 effect의 키로 전달할 필요가 없습니다.

### 기본 규칙 : Effect 블록에서 사용하는 변수는 key로 추가

```kotlin
LaunchedEffect(userId, searchQuery) {  // 블록 내부에서 사용하는 변수들
    val result = fetchData(userId, searchQuery)
    // ...
}
```
- 이유: 이 변수들이 바뀌면 Effect가 재시작되어야 새로운 값으로 작업이 수행되기 때문

### 예외 1: 값이 바뀌어도 재시작하지 않으려면 `rememberUpdatedState` 사용

```kotlin
val currentOnClick by rememberUpdatedState(onClick)

LaunchedEffect(Unit) {  // onClick을 key로 추가하지 않음
    delay(5000)
    currentOnClick()  // 항상 최신 onClick 사용하지만 Effect는 재시작 안됨
}
```
- 콜백 함수처럼 최신 값은 참조하고 싶지만, 값이 바뀔 때마다 긴 작업(예: 5초 딜레이)을 다시 시작하고 싶지 않을 때

### 예외 2: `remember`로 감싸져서 절대 안 바뀌는 변수는 key 불필요
```kotlin
val repository = remember { MyRepository() }  // 절대 안 바뀜

LaunchedEffect(searchQuery) {  // ✅ repository는 key로 추가 안해도 됨
    val result = repository.search(searchQuery)
}
```

### 예시

```kotlin
@Composable
fun UserProfile(userId: String, onLogout: () -> Unit) {
    val currentOnLogout by rememberUpdatedState(onLogout)
    
    LaunchedEffect(userId) {  // userId가 바뀌면 재시작
        val userData = fetchUserData(userId)
        // ...
        
        // onLogout이 바뀌어도 Effect는 재시작 안됨
        // 하지만 항상 최신 onLogout 함수를 참조함
    }
}
```

> **Key Point:** Variables used in an effect should be added as a parameter of the effect composable, or use `rememberUpdatedState`.

In the `DisposableEffect` code shown above, the effect takes as a parameter the `lifecycleOwner` used in its block, because any change to them should cause the effect to restart.

```kotlin
@Composable
fun HomeScreen(
    lifecycleOwner: LifecycleOwner = LocalLifecycleOwner.current,
    onStart: () -> Unit, // Send the 'started' analytics event
    onStop: () -> Unit // Send the 'stopped' analytics event
) {
    // These values never change in Composition
    val currentOnStart by rememberUpdatedState(onStart)
    val currentOnStop by rememberUpdatedState(onStop)

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            /* ... */
        }

        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }
}
```

`currentOnStart`와 `currentOnStop`은 `rememberUpdatedState`의 사용으로 인해 Composition에서 값이 절대 변경되지 않기 때문에 `DisposableEffect` 키로 필요하지 않습니다. `lifecycleOwner`를 파라미터로 전달하지 않고 그것(`lifecycleOwner`)이 변경되면, `HomeScreen`은 recompose 되지만 `DisposableEffect`는 dispose되거나 재시작되지 않습니다. 이는 그 시점부터 잘못된 `lifecycleOwner`가 사용되기 때문에 문제를 일으킵니다.

- `lifecycleOwner` 이 변경되면 `HomeScreen`이 recompose 되는 이유
  - `LocalLifecycleOwner.current`는 Composition Local이고, 이것도 State처럼 동작
  - `val lifecycleOwner = LocalLifecycleOwner.current` ← 이것을 읽으면 구독자가 됨
  - `LocalLifecycleOwner.current`는 Composition Local(정확히는 `ProvidableCompositionLocal<LifecycleOwner>`)이고, composition 안에서 읽으면 그 제공값(provided value)이 바뀔 때 해당 composable이 재구성
  - 같은 LifecycleOwner의 내부 라이프사이클 상태 변화(ON_PAUSE → ON_RESUME)는 `LocalLifecycleOwner.current` 값이 바뀌지 않으므로 자동으로 재구성되지 않음

### Constants as keys

`true`와 같은 상수를 effect 키로 사용하여 **호출 지점의 생명주기를 따르도록** 만들 수 있습니다. 위에 표시된 `LaunchedEffect` 예시처럼 이에 대한 유효한 사용 사례가 있습니다. 그러나 그렇게 하기 전에 다시 한번 생각해보고 그것이 정말 필요한지 확인하세요.