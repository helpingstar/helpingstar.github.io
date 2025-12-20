---
layout: single
title: "Android Navigation3"
date: 2025-12-20 23:00:00
lastmod : 2025-12-20 23:00:00
categories: android
tag: [android, compose, navigation]
toc: true
toc_sticky: true
published: true
---

![navigation3](../../assets/images/android/navigation3.png)

이미지 출처 : https://developer.android.com/guide/navigation/navigation-3/basics

위 이미지를 머리속에 넣어보자


## Navigation 3

Navigation 3에서 back stack은 content를 실제로 포함하지(contain) 않고 **key** 라고 알려진 것을 통해 content를 참조(reference)한다. refenrece 방식의 장점은 다음과 같다.

- back stack에 key를 push함으로써 간단하게 navigate한다.
- key가 serializable하면, back stack을 persistent storage에 저장할 수 있다.
  - persistent storage : 앱 프로세스가 종료되거나 재시작되더라도 데이터가 유지되는 저장공간
  - configuration 변경이나, 프로세스 종료 후에도 back stack 을 유지하여 마지막에 보던 content를 유지할 수 있다.

## `NavBackStack`

Navigation 3 API의 핵심 개념은 개발자가 back stack을 직접 소유한다는 것이다.

- back stack은 snapshot-stated backed `List<T>`여야 한다.
  - `T` : back stack의 `keys` 타입, `Any`를 사용할 수도 있고 더 강한 key 타입을 직접 정의할 수도 있다.
  - snapshot-state : Jetpack Compose의 Snapshot system을 말하며, Compose가 변경을 감지하고 recomposition을 트리거할 수 있는 state이다.
    - back stack 변경시 compose가 변경을 감지하고 `NavDisplay`가 recomposition하여 현재 back stack 상태가 UI에 반영될 수 있게 된다.
  - 라이브러리는 back stack을 observe하고 그 상태를 `NavDisplay`를 사용해 UI에 반영한다.

```kotlin
@Serializable(with = NavBackStackSerializer::class)
public class NavBackStack<T : NavKey> public constructor(internal val base: SnapshotStateList<T>) :
    MutableList<T> by base, StateObject by base, RandomAccess by base {

    public constructor() : this(base = mutableStateListOf())

    public constructor(vararg elements: T) : this(base = mutableStateListOf(*elements))
}
```

## `NavEntry`

Navigation 3 에서는 content를 `NavEntry`를 사용해 모델링한다. `NavEntry` 는 composable function을 포함하는 class로 하나의 destination을 나타낸다.

`NavEntry`는 content에 대한 정보인 `metadata: Map<String, Any>`도 포함할 수 있다. 이를 통해 content를 어떻게 표시할지 결정하는 데 사용한다.

`key`를 `NavEntry`로 변환하려면 Entry Provider를 생성해야 한다. Entry Provider는 `key`를 받아 해당 `key`에 대응하는 함수이다.

## Entry Provider

Entry Provider를 만드는 방법에는 두 가지가 있다.
- lambda function을 직접 작성
- `entryProvider` DSL을 사용하는 방법

`entryProvider` DSL을 사용하는 방법을 설명해본다.

```kotlin
// android/snippets
entryProvider = entryProvider {
    entry<ProductList> { Text("Product List") }
    entry<ProductDetail>(
        metadata = mapOf("extraDataKey" to "extraDataValue")
    ) { key -> Text("Product ${key.id} ") }
}
```

```kotlin
// skydoves/pokedex-composes
entryProvider = entryProvider<NavKey> {
    entry<PokedexScreen.Home> {
        PokedexHome(
            sharedTransitionScope = this@SharedTransitionLayout,
            animatedContentScope = LocalNavAnimatedContentScope.current,
        )
    }

    entry<PokedexScreen.Details> { screen ->
        PokedexDetails(
            sharedTransitionScope = this@SharedTransitionLayout,
            animatedContentScope = LocalNavAnimatedContentScope.current,
            pokemon = screen.pokemon,
        )
    }
}
```

```kotlin
// android/nowinandroid
val entryProvider = entryProvider {
    forYouEntry(navigator)
    bookmarksEntry(navigator)
    interestsEntry(navigator)
    topicEntry(navigator)
    searchEntry(navigator)
}

fun EntryProviderScope<NavKey>.forYouEntry(navigator: Navigator) {
    entry<ForYouNavKey> {
        ForYouScreen(
            onTopicClick = navigator::navigateToTopic,
        )
    }
}
```

- `entry`는 지정된 타입과 composable content를 가지는 `NavEntry`를 정의하는 데 사용된다.
- `entry`는 `NavEntry.metadata`를 설정하기 위한 `metadata` parameter를 받을 수 있다.

## `NavDisplay`

`NavDisplay`는 back stack을 관찰하고 적절히 UI를 업데이트 한다. `NavDisplay`는 다음과 같은 parameter로 생성한다.

- back stack : `SnapshotStateList<T>` 이어야 하며, 이는 observable `List`이므로 변경시 `NavDisplay`의 recomposition을 트리거한다.
  - `T` : back stack key 타입
- `entryProvider` : back stack에 들어 있는 key들을 `NavEntry` object로 변환하는 역할을 한다.
- `onBack` (optional) : 사용자가 back event를 발생시켰을 때 호출되는 lambda를 전달할 수 있다.

```kotlin
// android/snippets
data object Home
data class Product(val id: String)

@Composable
fun NavExample() {

    val backStack = remember { mutableStateListOf<Any>(Home) }

    NavDisplay(
        backStack = backStack,
        onBack = { backStack.removeLastOrNull() },
        entryProvider = { key ->
            when (key) {
                is Home -> NavEntry(key) {
                    ContentGreen("Welcome to Nav3") {
                        Button(onClick = {
                            backStack.add(Product("123"))
                        }) {
                            Text("Click to navigate")
                        }
                    }
                }

                is Product -> NavEntry(key) {
                    ContentBlue("Product ${key.id} ")
                }

                else -> NavEntry(Unit) { Text("Unknown route") }
            }
        }
    )
}
```

```kotlin
// skydoves/pokedex-composes
NavDisplay(
    backStack = backStack,
    onBack = { backStack.removeLastOrNull() },
    entryDecorators = listOf(rememberSaveableStateHolderNavEntryDecorator()),
    entryProvider = entryProvider<NavKey> {
        entry<PokedexScreen.Home> { ... }
        entry<PokedexScreen.Details> { ... }
    },
)
```

```kotlin
// android/nowinandroid
NavDisplay(
    entries = appState.navigationState.toEntries(entryProvider),
    sceneStrategy = listDetailStrategy,
    onBack = { navigator.goBack() },
)
```

![navigation3](../../assets/images/android/navigation3.png)

이미지 출처 : https://developer.android.com/guide/navigation/navigation-3/basics

1. Navigation event가 발생하면 변경이 시작된다  
    사용자 상호작용에 따라 key가 back stack에 추가 / 제거된다.
2. back stack state의 변경은 content 조회를 트리거한다
   - `NavDisplay`는 back stack을 observe, 기본 설정에서는 single pane layout으로 back stack의 가장 위에 있는 entry를 표시한다.
   - back stack의 top key가 변경되면, `NavDisplay`는 해당 key를 사용해 entry provider에 대응하는 content를 요청
3. Entry provider가 content를 제공한다.
   - `NavDisplay`로부터 key를 전달받으면, entry provider는 key와 content를 모두 포함하는 해당 `NavEntry`를 제공한다.
4. `NavDisplay`는 전달받은 `Entry`를 수신하고, 그 안에 포함된 content를 화면에 표시한다.

## `rememberNavBackStack`

configuration changes와 process death를 거쳐도 유지되는 back stack을 생성하도록 설계되었다.

`rememberNavBackStack`가 정상적으로 동작하려면, back stack에 포함된 각 key는 다음 요구사항을 충족해야 한다.
- back stack의 모든 key는 `NavKey` interface를 구현해야 한다. 이는 해당 key가 저장 가능함을 라이브러리에 알리는 marker interface 역할을 한다.
- `@Serializable` annotation이 선언되어 있어야 한다.

```kotlin
// android/snippets
@Serializable
data object Home : NavKey

@Composable
fun NavBackStack() {
    val backStack = rememberNavBackStack(Home)
}
```

```kotlin
// skydoves/pokedex-composes
sealed interface PokedexScreen : NavKey {
  @Serializable
  data object Home : PokedexScreen

  @Serializable
  data class Details(val pokemon: Pokemon) : PokedexScreen
}

val backStack = rememberNavBackStack(PokedexScreen.Home)
```

```kotlin
// android/nowinandroid
@Serializable
object ForYouNavKey : NavKey

val topLevelStack = rememberNavBackStack(startKey)
```

## `NavEntryDecorator`

`ViewModel`은 configuration change 전반에 걸쳐 UI 관련 상태를 유지하는 데 사용된다. 기본적으로 `ViewModel`은 가장 가까운 `ViewModelStoreOwner`에 스코프되며, 이는 일반적으로 `Activity` 또는 `Fragment`이다.

전체 `Activity`가 아니라 back stack 상의 특정 `NavEntry`에 `ViewModel`을 스코프하고 싶을 수 있는데, 이렇게 하면 해당 `NavEntry`가 back stack에 존재하는 동안에만 `ViewModel`의 상태가 유지되며, `NavEntry`가 pop될 때 `ViewModel`도 함께 정리된다.

`NavEntryDecorator`는 각 `NavEntry`마다 `ViewModelStoreOwner`를 제공한다. `NavEntry`의 content 내부에서 `ViewModel`을 생성하면 (예: Compose에서 `viewModel()`을 사용하는 경우), 해당 `ViewModel`은 back stack 상에서 그 `NavEntry`의 key에 자동으로 스코프된다. 즉 `ViewModel`은 `NavEntry`가 back stack에 추가될 때 생성되고 back stack에서 제거될 때 함께 정리된다.

`NavEntry`에 `ViewModel`을 스코프 하기 위해 `NavEntryDecorator`를 사용하는 방법은 다음과 같다.

- `NavDisplay`를 구성할 때 `entryDecorators` 목록에 `rememberSaveableStateHolderNavEntryDecorator()`, `rememberViewModelStoreNavEntryDecorator()`를 추가한다.

```kotlin
// android/snippets
NavDisplay(
    entryDecorators = listOf(
        // Add the default decorators for managing scenes and saving state
        rememberSaveableStateHolderNavEntryDecorator(),
        // Then add the view model store decorator
        rememberViewModelStoreNavEntryDecorator()
    ),
    backStack = backStack,
    entryProvider = entryProvider { },
)
```

```kotlin
// skydoves/pokedex-composes
NavDisplay(
    backStack = backStack,
    onBack = { backStack.removeLastOrNull() },
    entryDecorators = listOf(rememberSaveableStateHolderNavEntryDecorator()),
    entryProvider = entryProvider<NavKey> {
        entry<PokedexScreen.Home> { ... }
        entry<PokedexScreen.Details> { ... }
    },
)
```

```kotlin
// android/nowinandroid
/**
 * Convert NavigationState into NavEntries.
 */
@Composable
fun NavigationState.toEntries(
    entryProvider: (NavKey) -> NavEntry<NavKey>,
): SnapshotStateList<NavEntry<NavKey>> {
    val decoratedEntries = subStacks.mapValues { (_, stack) ->
        val decorators = listOf(
            rememberSaveableStateHolderNavEntryDecorator<NavKey>(),
            rememberViewModelStoreNavEntryDecorator<NavKey>(),
        )
        rememberDecoratedNavEntries(
            backStack = stack,
            entryDecorators = decorators,
            entryProvider = entryProvider,
        )
    }

    return topLevelStack
        .flatMap { decoratedEntries[it] ?: emptyList() }
        .toMutableStateList()
}

```


## 참고
- https://developer.android.com/guide/navigation/navigation-3/basics
- https://developer.android.com/guide/navigation/navigation-3/save-state
- https://github.com/android/snippets
- https://github.com/skydoves/pokedex-compose
- https://github.com/android/nowinandroid