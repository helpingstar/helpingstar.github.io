---
layout: single
title: "오픈소스로 알아보는 안드로이드 : 3. Bitwarden의 MVI 패턴"
date: 2026-06-16 17:22:50
lastmod : 2026-06-16 17:22:50
categories: android
tag: []
toc: true
toc_sticky: true
published: false
---

Bitwarden이 MVI 패턴을 어떻게 구현 했는지 알아본다.

일단 시작 전에 Bitwarden은 문서 어디에도 MVI라는 표현을 사용하지 않는다. 해당 프로젝트 내부 문서에서도 MVVM + UDF(Unidirectional Data Flow) 이라고 표현한다. 하지만 아래와 이유로 편의상 MVI라고 표현하였다.

1. 단일 불변 state : `MutableStateFlow<S>` 하나로 전체 화면 상태를 표현하고 밖으로는 읽기 전용 `StateFlow`만 노출한다.
2. Intent를 통한 입력 : 모든 입력이 `Action(A)` 타입으로 채널을 통해서만 들어온다.
3. 단일 진입점 reducer : `handleAction()` 한 곳에서 Action을 받아 동기적으로 State를 갱신한다.
4. 단방향 순환(UI -> Intent -> Model -> UI) : Action 채널 -> `handleAction` -> `mutableStateFlow` -> `stateFlow` -> UI
5. 일회성 Effect/Event 분리 : State와 별개로 `eventFlow`(Channel)로 일회성 이벤트 처리
6. Intent를 sealed `Action` 타임으로 모델링 : 입력을 데이터(sealed class Action)으로 구체화해서 채널로 흘려보냄 (MVI의 Intent)
7. 비동기 결과를 `Action.Internal`로 되돌려 단일 reducer로 처리
   - 코드의 `CLAUDE.md`는 다음과 같이 명시한다 (follow these rules: DO)
   - "Map async results to internal actions before updating state"
   - 코루틴 안에서 상태를 직접 바꾸지 않고 `sendAction(Action.Internal.*)`으로 다시 채널에 넣어 `handle(Internal)Action`에서 처리 -> 모든 상태 변경 경로가 단 하나의 reducer로 수렴
8. 코드 내부 Kdoc : "The screen adheres to the MVI pattern by observing state from [ReviewExportViewModel]" - ReviewExportScreen

Bitwarden의 MVI 패턴

Bitwarden의 MVI 

# `BaseViewModel`

```kotlin
interface SendChannel<in E>
```
- Kotlin Coroutines `Channel`의 송신 전용 인터페이스
- 값을 보내는 쪽 producer에게만 노출할 수 있는 API

`SendChannel<E>` 내부의 핵심 API를 단순화 하면 아래와 같은 구조이다.

```kotlin
interface SendChannel<in E> {
    suspend fun send(element: E)
    fun trySend(element: E): ChannelResult<Unit>
}
```
여기서 `<in E>`는 왜 붙은걸까
- `SendChannel`은 `E`를 밖으로 꺼내서 주는 API가 아니다. `E`를 channel에 넣는 API이다. 즉 `SendChannel`은 `E`를 소비하는 consumer이다.
- 이 구조에서는 `in`을 붙이는 것이 type-safe하다. 덕분에 더 넓은 type을 받을 수 있는 `SendChannel`을 더 좁은 type의 `SendChannel`처럼 사용할 수 있다.
- 편하게 설명하면 다음과 같다. 
  - `SendChannel<Cat>`이 필요한 곳에는, `Cat`보다 더 넓은 type을 받을 수 있는 `SendChannel<Animal>`이 들어와도 안전하다.

이제 `trySend`와 `send`에 대해 알아보자, 공식문서에서는 시그니처와 첫 문단을 각각 다음과 같이 설명한다.

```kotlin
abstract suspend fun send(element: E)
```
Sends the specified element to this channel.


```kotlin
abstract fun trySend(element: E): ChannelResult<Unit>
```
Attempts to add the specified element to this channel without waiting.

이제 하나하나씩 살펴보자 공식문서는 `send`, `trySend`에 대해 각각 다음과 같이 설명한다

`send`
- Sends the specified element to this channel.
- 이 함수는 element를 channel의 buffer로 전달하지 못하면 suspend된다.
- buffer가 없는 경우에는 receiving side로 직접 전달하지 못하면 suspend 된다.
- element 전달에 성공했는지 여부와 관계없이 cancel 될 수 있다.
- channel이 closed 상태이면 exception이 throw된다.

`trySend`
- Attempts to add the specified element to this channel without waiting.
- 절대 suspend 되지 않으며, exception을 throw하지도 않는다. 대신 operation의 결과를 캡슐화한 `ChannelResult`를 반환한다.
- 이 channel이 현재 가득 차 있어서 그 시점에 새 element를 받을 수 없거나, closed 상태라면, 이 함수는 failure를 나타내는 result를 반환한다.
  - 이 경우 element가 consumer에게 전달되지 않았다는 것이 보장된다. (channel 안으로 들어간 적이 없는 값으로 취급된다.)
  - 또한 `Channel` 생성 시 `onUndeliveredElement` callback이 제공되었더라도, 이 callback은 호출되지 않는다.
- channel의 buffer가 overflow되지 않는다는 것을 사전에 알고 있는 경우, `send`의 non-suspend 대안으로 사용할 수 있다.

표현의 차이
- trySend는 send가 아니라 add라는 표현을 쓴다. 
  - `send`의 "Sends the specified element to this channel."는 다음 의미에 가깝다.
    - element를 channel에 **보낸다**, 필요하면 suspend해서라도 sending operating을 완료한다.
  - `trySend`의 "Attempts to add the specified element to this channel without waiting."은 다음 의미에 가깝다.
    - element를 channel에 **추가하려고 시도한다**. 지금 추가할 수 없으면 기다리지 않는다.

정리하면 아래와 같다.

| 구분                     | `send(element)`                              | `trySend(element)`                                               |
| ---------------------- | -------------------------------------------- | ---------------------------------------------------------------- |
| 선언                     | `suspend fun send(element: E)`               | `fun trySend(element: E): ChannelResult<Unit>`                   |
| suspend 여부             | 가능                                           | 절대 suspend 안 함                                                   |
| channel이 가득 찬 경우       | buffer에 공간이 생기거나 receiver가 받을 때까지 suspend 가능 | 즉시 실패 결과 반환                                                      |
| closed channel에 보내는 경우 | exception throw                              | 실패한 `ChannelResult` 반환                                           |
| 실패 처리                  | `try/catch`, coroutine cancellation 처리       | `result.isSuccess`, `result.isFailure`, `result.isClosed` 등으로 확인 |
| 용도                     | 반드시 보내야 하는 값                                 | 못 보내도 괜찮거나, 즉시 결과가 필요한 값                                         |

`BaseViewModel`에서는 `send` `trySend`를 다음과 같이 활용한다.

```kotlin
abstract class BaseViewModel<S, E, A>(
    initialState: S,
) : ViewModel() {
    ...
    /**
     * Convenience method for sending an action to the [actionChannel].
     * (번역) [actionChannel]에 action을 보내기 위한 편의 method입니다.
     */
    fun trySendAction(action: A) {
        actionChannel.trySend(action)
    }

    /**
     * Helper method for sending an internal action.
     * (번역) 내부 action을 보내기 위한 helper method입니다.
     */
    protected suspend fun sendAction(action: A) {
        actionChannel.send(action)
    }
}
```

`trySendAction`
- 외부(UI) 에서 호출하는 진입점이다. (public)
- UI는 코루틴 스코프가 아닌 곳에서 Action을 던진다. 그런 자리에서 `suspend` 함수는 못 부르므로 `trySend`를 사용한다. 사용자 상호작용이 ViewModel이 들어오는 통로 역할을 한다.
  - Composable callback
  - Activity lifecycle
  - Button click

`sendAction`
- `protected` : ViewModel 자기 자신만 호출 가능
- `suspend` : 코루틴 안에서만 호출 가능하다.

trySendAction의 유실 걱정

dd
# `BackgroundEvent`

# `EventsEffect`

# 참고

- https://kotlinlang.org/docs/generics.html
- https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.channels/-send-channel/try-send.html
- https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.channels/-send-channel/send.html


## TODO

설계 의도를 하나씩 "왜 다른 선택지가 아니었나"까지 따져서 설명하겠습니다. 세 가지 결정이 맞물려 있습니다: **(A) visibility, (B) `trySendAction`이 non-suspend인 이유, (C) `sendAction`이 suspend인 이유.**

먼저 두 메서드를 다시 봅니다 ([BaseViewModel.kt:74-83](ui/src/main/kotlin/com/bitwarden/ui/platform/base/BaseViewModel.kt#L74)):

```kotlin
fun trySendAction(action: A) {                 // public, non-suspend
    actionChannel.trySend(action)
}

protected suspend fun sendAction(action: A) {  // protected, suspend
    actionChannel.send(action)
}
```

핵심 통찰: **이 두 메서드를 가르는 진짜 축은 "호출자가 누구인가(외부 UI vs ViewModel 자신)"이고, visibility와 suspend 유무는 그 축을 컴파일러가 강제하도록 만든 장치입니다.**

---

## A. Visibility: `public` vs `protected` — "누가 입력을 만들 권리가 있는가"

MVI에서 가장 중요한 불변식은 **"State를 바꾸는 경로는 하나뿐"** 입니다. 그런데 입력 출처는 성격이 완전히 다른 두 종류입니다.

1. **외부 입력 (UI/시스템)** — 버튼 클릭, Activity 콜백, 외부 이벤트. 이건 외부가 ViewModel에게 "이런 일이 일어났다"고 알리는 것. → **공개 진입점이어야 함**.
2. **내부 후속 입력 (ViewModel 자신)** — 비동기 작업이 끝난 뒤 그 결과를 reducer로 되돌리는 것. 이건 ViewModel의 구현 디테일.

만약 `sendAction`(내부용)도 `public`이었다면, 외부에서 `viewModel.sendAction(SomeAction.Internal.XxxResult(...))` 같은 **내부 전용 액션을 위조해서 주입**할 수 있게 됩니다. 그러면 "비동기 결과는 ViewModel만 만든다"는 보장이 깨지고, `Action.Internal` 컨벤션이 무의미해집니다.

그래서:
- `trySendAction` → `public`: 외부가 정상적인 입력을 던지는 **정문**.
- `sendAction` → `protected`: ViewModel(과 서브클래스)만 쓸 수 있는 **내부 메커니즘**. 외부 위조를 컴파일 타임에 차단.

즉 visibility는 단순 캡슐화가 아니라 **"입력 출처의 신뢰 경계(trust boundary)"를 타입 시스템으로 못박는 장치**입니다.

> 곁다리: `actionChannel` 자체는 `public val`로 노출돼 있는데, 이건 외부에서 Flow의 `onEach` 같은 연산 결과를 직접 채널로 흘려보내는 고급 용례(예: `MainViewModel`이 다른 Flow를 구독)를 위한 것입니다. 그래도 타입이 `SendChannel<A>`(받기 불가, 보내기만)로 제한돼 있어 읽기는 막혀 있습니다.

---

## B. `trySendAction`은 왜 non-suspend(`trySend`)인가 — "호출 환경이 코루틴이 아니다"

이게 가장 중요한 설계 이유입니다. **외부 호출 지점들은 대부분 `suspend`를 부를 수 없는 곳**입니다.

실제 호출처를 보면 명백합니다:

```kotlin
// MainActivity.kt:77 — Activity 콜백 (코루틴 아님)
mainViewModel.trySendAction(MainAction.DuoResult(it))

// Compose onClick = { viewModel.trySendAction(...) }  — 일반 람다, suspend 아님
```

Composable의 `onClick`, Activity의 `onCreate`/콜백, `LaunchedEffect` 밖의 람다 — 이들은 **non-suspend 컨텍스트**입니다. 여기서 `suspend fun`은 호출 자체가 컴파일되지 않습니다.

만약 외부 진입점을 `suspend`로 만들었다면, 모든 UI 호출처가 `rememberCoroutineScope().launch { ... }`로 감싸야 합니다:

```kotlin
// 만약 trySendAction이 suspend였다면 — UI마다 이런 보일러플레이트가 필요
val scope = rememberCoroutineScope()
Button(onClick = { scope.launch { viewModel.sendAction(...) } })
```

이건 모든 클릭마다 코루틴을 띄우는 낭비이고, 보일러플레이트이며, "단순히 액션 하나 던지는 데 왜 코루틴이 필요하지?"라는 인지 부조화를 낳습니다.

여기서 **`Channel.UNLIMITED`** 설계가 결정적으로 받쳐줍니다 ([BaseViewModel.kt:30](ui/src/main/kotlin/com/bitwarden/ui/platform/base/BaseViewModel.kt#L30)):

```kotlin
private val internalActionChannel: Channel<A> = Channel(capacity = Channel.UNLIMITED)
```

`send`가 suspend여야 하는 **유일한 이유는 "버퍼가 가득 차면 기다려야 하기 때문"**입니다. 그런데 버퍼가 무제한이면 가득 찰 일이 없으므로 **기다릴 필요가 영원히 없고**, 따라서 `trySend`(즉시 넣고 끝)로 충분합니다. `trySend`는 절대 실패하지 않고 절대 블로킹하지 않으므로 non-suspend로 안전하게 노출할 수 있습니다.

정리하면 인과는 이렇습니다:

```
외부 호출처는 non-suspend 환경 (onClick, 콜백)
        ↓ (그래서 외부 API도 non-suspend여야 함)
non-suspend로 만들려면 send 대신 trySend
        ↓ (trySend가 안전하려면 절대 안 막혀야 함)
그래서 Channel을 UNLIMITED로 설계
        ↓
backpressure 없는 "fire-and-forget" 입력 진입점 완성
```

세 가지(UNLIMITED 채널 + trySend + non-suspend)가 **서로를 가능하게 하는 한 묶음의 설계 결정**입니다.

---

## C. `sendAction`은 왜 suspend(`send`)인가 — "이미 코루틴 안이고, suspend가 용도를 강제한다"

내부용은 정반대 상황입니다. `sendAction`이 불리는 곳은 **항상 이미 코루틴 안**입니다 — 비동기 작업(네트워크/SDK)을 await한 직후이기 때문입니다.

```kotlin
// SendViewModel.kt:469 부근
viewModelScope.launch {                       // 이미 코루틴 스코프 안
    val result = repository.deleteSend(...)    // suspend — 비동기 작업
    sendAction(SendAction.Internal.DeleteSendResultReceive(result))  // 후속 액션
}
```

여기서 `suspend`를 붙인 데에는 두 가지 의도가 있습니다.

**① "용도를 컴파일러가 강제한다" (가장 핵심)**
`sendAction`은 *오직 코루틴 안에서만* 호출 가능합니다. 그런데 ViewModel이 코루틴을 쓰는 거의 유일한 이유는 "비동기 작업"입니다. 즉 `suspend` 키워드가 **"이 메서드는 비동기 결과를 되돌릴 때만 써라"는 의도를 타입으로 못박습니다.**

만약 내부용도 non-suspend였다면, 개발자가 `handleAction` 안(동기 컨텍스트)에서 `sendAction`을 직접 불러 액션을 재귀적으로 던지는 식의 안티패턴이 쉽게 생깁니다. suspend라는 마찰이 "여긴 코루틴이 끝난 비동기 경계다"라는 신호를 줍니다.

**② 의미적 정합성: `trySend`는 여기서 거짓말이 된다**
non-suspend인 `trySendAction`도 코루틴 안에서 부를 수는 있습니다. 하지만 `trySend`는 "넣고 결과는 안 본다(fire-and-forget)"는 의미입니다. 반면 비동기 결과 합류 지점에서는 **"이 결과 액션은 반드시 reducer에 들어가야 한다"**는 보장이 의미적으로 더 적절합니다. `send`(suspend)는 호출자에게 "넣을 때까지 책임진다"는 시맨틱을 줍니다. (UNLIMITED라 실제론 둘 다 즉시 성공하지만, **의도 표현**이 다릅니다.)

대비를 위해 `sendEvent`도 보세요 ([BaseViewModel.kt:88-90](ui/src/main/kotlin/com/bitwarden/ui/platform/base/BaseViewModel.kt#L88)):

```kotlin
protected fun sendEvent(event: E) {                    // non-suspend
    viewModelScope.launch { eventChannel.send(event) }  // 내부에서 코루틴을 직접 띄움
}
```

이벤트는 동기 컨텍스트(`handleAction` 안)에서도 자주 보내야 하므로, **편의를 위해 내부에서 `launch`를 감싸 non-suspend로 제공**합니다. 반면 `sendAction`은 일부러 그렇게 하지 않았습니다 — **"액션은 비동기 경계에서만 되돌려라"는 제약을 유지하기 위해 의도적으로 suspend를 남겨둔 것**입니다. 이 두 메서드의 대비가 설계 의도를 가장 잘 보여줍니다.

---

## 종합: 세 결정이 만드는 하나의 그림

| | `trySendAction` | `sendAction` |
|---|---|---|
| 호출자 | 외부 (UI/시스템) | ViewModel 자신 |
| Visibility | `public` (정문) | `protected` (내부 위조 차단) |
| suspend | ✗ (호출처가 코루틴 아님) | ✓ (호출처가 이미 코루틴) |
| 채널 연산 | `trySend` (fire-and-forget) | `send` (넣음을 보장하는 시맨틱) |
| 설계 의도 | non-suspend 환경에서 마찰 없이 입력 | suspend로 "비동기 결과 합류" 용도를 강제 |

한 문장으로: **호출 환경의 차이(non-suspend UI vs 코루틴 내부)가 suspend 유무를 결정했고, 입력 출처의 신뢰 차이(외부 vs 내부)가 visibility를 결정했으며, 이 모든 걸 가능하게 한 토대가 `Channel.UNLIMITED`(backpressure 제거)입니다.** 세 결정은 따로 고른 게 아니라 "UI에서 마찰 없이 액션을 던지되, 내부 비동기 결과는 단일 reducer로만 합류시킨다"는 하나의 목표에서 파생된 정합적 묶음입니다.