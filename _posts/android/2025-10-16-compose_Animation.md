---
layout: single
title: "Compose Animation"
date: 2025-10-16 11:00:00
lastmod : 2025-10-16 11:00:00
categories: android
tag: [android, coroutine]
toc: true
toc_sticky: true
published: false
---

# Value-based animations

## Animate a single value with `animate*AsState`

The [`animate*AsState`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/package-summary#animateDpAsState\(androidx.compose.ui.unit.Dp,androidx.compose.animation.core.AnimationSpec,kotlin.String,kotlin.Function1\)) functions are the simplest animation APIs in Compose for animating a single value. You only provide the target value (or end value), and the API starts animation from the current value to the specified value.

Below is an example of animating alpha using this API. By simply wrapping the target value in [`animateFloatAsState`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/package-summary#animateFloatAsState\(kotlin.Float,androidx.compose.animation.core.AnimationSpec,kotlin.Float,kotlin.String,kotlin.Function1\)), the alpha value is now an animation value between the provided values (`1f` or `0.5f` in this case).

```kotlin
var enabled by remember { mutableStateOf(true) }

val animatedAlpha: Float by animateFloatAsState(if (enabled) 1f else 0.5f, label = "alpha")
Box(
    Modifier
        .fillMaxSize()
        .graphicsLayer { alpha = animatedAlpha }
        .background(Color.Red)
)
```

어떤 animation 클래스의 인스턴스를 생성하거나 인터럽션을 처리할 필요가 없다는 점에 유의하세요. 

- 내부적으로는 애니메이션 객체(정확히는 `Animatable` 인스턴스)가 생성되어 호출 지점(call site)에 기억됨
  - 여기서의 call site는 `val animatedAlpha: Float by animateFloatAsState(...)` 에 해당한다.
- 첫 번째 target value를 초기값으로 사용
  - 그 이후에는 이 composable에 다른 target value를 전달할 때마다 해당 값으로 자동으로 animation이 시작
  - 이미 animation이 진행 중이면 애니메이션은 현재 값(및 velocity)에서 시작하여 target value를 향해 애니메이션됨
- 애니메이션 도중 이 composable은 recomposed되고 매 프레임마다 업데이트된 animation value를 반환


Out of the box, Compose provides `animate*AsState` functions for `Float`, `Color`, `Dp`, `Size`, `Offset`, `Rect`, `Int`, `IntOffset`, and `IntSize`. You can easily add support for other data types by providing a `TwoWayConverter` to `animateValueAsState` that takes a generic type.

## Animate multiple properties simultaneously with a transition

[`Transition`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/Transition) manages one or more animations as its children and runs them simultaneously between multiple states.

The states can be of any data type. In many cases, you can use a custom `enum` type to ensure type safety, as in this example:

```kotlin
enum class BoxState {
    Collapsed,
    Expanded
}
```

[`updateTransition`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/package-summary#updateTransition\(kotlin.Any,kotlin.String\)) creates and remembers an instance of `Transition` and updates its state.

```kotlin
var currentState by remember { mutableStateOf(BoxState.Collapsed) }
val transition = updateTransition(currentState, label = "box state")
```

You can then use one of `animate*` extension functions to define a child animation in this transition. Specify the target values for each of the states. These `animate*` functions return an animation value that is updated every frame during the animation when the transition state is updated with `updateTransition`.

val rect by transition.animateRect(label = "rectangle") { state ->
    when (state) {
        BoxState.Collapsed -> Rect(0f, 0f, 100f, 100f)
        BoxState.Expanded -> Rect(100f, 100f, 300f, 300f)
    }
}
val borderWidth by transition.animateDp(label = "border width") { state ->
    when (state) {
        BoxState.Collapsed -> 1.dp
        BoxState.Expanded -> 0.dp
    }
}

Optionally, you can pass a `transitionSpec` parameter to specify a different `AnimationSpec` for each of the combinations of transition state changes. See [AnimationSpec](https://developer.android.com/develop/ui/compose/animation/customize#animationspec) for more information.

val color by transition.animateColor(
    transitionSpec = {
        when {
            BoxState.Expanded isTransitioningTo BoxState.Collapsed ->
                spring(stiffness = 50f)

            else ->
                tween(durationMillis = 500)
        }
    }, label = "color"
) { state ->
    when (state) {
        BoxState.Collapsed -> MaterialTheme.colorScheme.primary
        BoxState.Expanded -> MaterialTheme.colorScheme.background
    }
}

Once a transition has arrived at the target state, `Transition.currentState` will be the same as `Transition.targetState`. This can be used as a signal for whether the transition has finished.

We sometimes want to have an initial state different from the first target state. We can use `updateTransition` with `MutableTransitionState` to achieve this. For example, it allows us to start animation as soon as the code enters composition.

// Start in collapsed state and immediately animate to expanded
var currentState = remember { MutableTransitionState(BoxState.Collapsed) }
currentState.targetState = BoxState.Expanded
val transition = rememberTransition(currentState, label = "box state")
// ……

For a more complex transition involving multiple composable functions, you can use [`createChildTransition`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/Transition#\(androidx.compose.animation.core.Transition\).createChildTransition\(kotlin.String,kotlin.Function1\)) to create a child transition. This technique is useful for separating concerns among multiple subcomponents in a complex composable. The parent transition will be aware of all the animation values in the child transitions.

enum class DialerState { DialerMinimized, NumberPad }

@Composable
fun DialerButton(isVisibleTransition: Transition<Boolean>) {
    // `isVisibleTransition` spares the need for the content to know
    // about other DialerStates. Instead, the content can focus on
    // animating the state change between visible and not visible.
}

@Composable
fun NumberPad(isVisibleTransition: Transition<Boolean>) {
    // `isVisibleTransition` spares the need for the content to know
    // about other DialerStates. Instead, the content can focus on
    // animating the state change between visible and not visible.
}

@Composable
fun Dialer(dialerState: DialerState) {
    val transition = updateTransition(dialerState, label = "dialer state")
    Box {
        // Creates separate child transitions of Boolean type for NumberPad
        // and DialerButton for any content animation between visible and
        // not visible
        NumberPad(
            transition.createChildTransition {
                it == DialerState.NumberPad
            }
        )
        DialerButton(
            transition.createChildTransition {
                it == DialerState.DialerMinimized
            }
        )
    }
}

### Use transition with `AnimatedVisibility` and `AnimatedContent`

[`AnimatedVisibility`](https://developer.android.com/reference/kotlin/androidx/compose/animation/package-summary#\(androidx.compose.animation.core.Transition\).AnimatedVisibility\(kotlin.Function1,androidx.compose.ui.Modifier,androidx.compose.animation.EnterTransition,androidx.compose.animation.ExitTransition,kotlin.Function1\)) and [`AnimatedContent`](https://developer.android.com/reference/kotlin/androidx/compose/animation/package-summary#\(androidx.compose.animation.core.Transition\).AnimatedContent\(androidx.compose.ui.Modifier,kotlin.Function1,androidx.compose.ui.Alignment,kotlin.Function2\)) are available as extension functions of `Transition`. The `targetState` for `Transition.AnimatedVisibility` and `Transition.AnimatedContent` is derived from the `Transition`, and triggering enter/exit transitions as needed when the `Transition`'s `targetState` has changed. These extension functions allow all the enter/exit/sizeTransform animations that would otherwise be internal to `AnimatedVisibility`/`AnimatedContent` to be hoisted into the `Transition`. With these extension functions, `AnimatedVisibility`/`AnimatedContent`'s state change can be observed from outside. Instead of a boolean `visible` parameter, this version of `AnimatedVisibility` takes a lambda that converts the parent transition's target state into a boolean.

See [AnimatedVisibility](https://developer.android.com/develop/ui/compose/animation/composables-modifiers#animatedvisibility) and [AnimatedContent](https://developer.android.com/develop/ui/compose/animation/composables-modifiers#animatedcontent) for the details.

var selected by remember { mutableStateOf(false) }
// Animates changes when `selected` is changed.
val transition = updateTransition(selected, label = "selected state")
val borderColor by transition.animateColor(label = "border color") { isSelected ->
    if (isSelected) Color.Magenta else Color.White
}
val elevation by transition.animateDp(label = "elevation") { isSelected ->
    if (isSelected) 10.dp else 2.dp
}
Surface(
    onClick = { selected = !selected },
    shape = RoundedCornerShape(8.dp),
    border = BorderStroke(2.dp, borderColor),
    shadowElevation = elevation
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp)
    ) {
        Text(text = "Hello, world!")
        // AnimatedVisibility as a part of the transition.
        transition.AnimatedVisibility(
            visible = { targetSelected -> targetSelected },
            enter = expandVertically(),
            exit = shrinkVertically()
        ) {
            Text(text = "It is fine today.")
        }
        // AnimatedContent as a part of the transition.
        transition.AnimatedContent { targetState ->
            if (targetState) {
                Text(text = "Selected")
            } else {
                Icon(imageVector = Icons.Default.Phone, contentDescription = "Phone")
            }
        }
    }
}

### Encapsulate a transition and make it reusable

For simple use cases, defining transition animations in the same composable as your UI is a perfectly valid option. When you are working on a complex component with a number of animated values, however, you might want to separate the animation implementation from the composable UI.

You can do so by creating a class that holds all the animation values and an ‘update’ function that returns an instance of that class. The transition implementation can be extracted into the new separate function. This pattern is useful when there is a need to centralize the animation logic, or make complex animations reusable.

enum class BoxState { Collapsed, Expanded }

@Composable
fun AnimatingBox(boxState: BoxState) {
    val transitionData = updateTransitionData(boxState)
    // UI tree
    Box(
        modifier = Modifier
            .background(transitionData.color)
            .size(transitionData.size)
    )
}

// Holds the animation values.
private class TransitionData(
    color: State<Color>,
    size: State<Dp>
) {
    val color by color
    val size by size
}

// Create a Transition and return its animation values.
@Composable
private fun updateTransitionData(boxState: BoxState): TransitionData {
    val transition = updateTransition(boxState, label = "box state")
    val color = transition.animateColor(label = "color") { state ->
        when (state) {
            BoxState.Collapsed -> Color.Gray
            BoxState.Expanded -> Color.Red
        }
    }
    val size = transition.animateDp(label = "size") { state ->
        when (state) {
            BoxState.Collapsed -> 64.dp
            BoxState.Expanded -> 128.dp
        }
    }
    return remember(transition) { TransitionData(color, size) }
}

## Create an infinitely repeating animation with `rememberInfiniteTransition`

[`InfiniteTransition`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/InfiniteTransition) holds one or more child animations like `Transition`, but the animations start running as soon as they enter the composition and do not stop unless they are removed. You can create an instance of `InfiniteTransition` with `rememberInfiniteTransition`. Child animations can be added with `animateColor`, `animatedFloat`, or `animatedValue`. You also need to specify an [infiniteRepeatable](https://developer.android.com/develop/ui/compose/animation/value-based#infiniterepeatable) to specify the animation specifications.

val infiniteTransition = rememberInfiniteTransition(label = "infinite")
val color by infiniteTransition.animateColor(
    initialValue = Color.Red,
    targetValue = Color.Green,
    animationSpec = infiniteRepeatable(
        animation = tween(1000, easing = LinearEasing),
        repeatMode = RepeatMode.Reverse
    ),
    label = "color"
)

Box(
    Modifier
        .fillMaxSize()
        .background(color)
)

## Low-level animation APIs

All the high-level animation APIs mentioned in the previous section are built on top of the foundation of the low-level animation APIs.

The `animate*AsState` functions are the simplest APIs, that render an instant value change as an animation value. It is backed by `Animatable`, which is a coroutine-based API for animating a single value. `updateTransition` creates a transition object that can manage multiple animating values and run them based on a state change. `rememberInfiniteTransition` is similar, but it creates an infinite transition that can manage multiple animations that keep on running indefinitely. All of these APIs are composables except for `Animatable`, which means these animations can be created outside of composition.

All of these APIs are based on the more fundamental `Animation` API. Though most apps will not interact directly with `Animation`, some of the customization capabilities for `Animation` are available through higher-level APIs. See [Customize animations](https://developer.android.com/develop/ui/compose/animation/customize) for more information on `AnimationVector` and `AnimationSpec`.

![Diagram showing the relationship between the various low-level animation APIs](https://developer.android.com/static/develop/ui/compose/images/animation-low-level.svg)

### `Animatable`: Coroutine-based single value animation

[`Animatable`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/Animatable) is a value holder that can animate the value as it is changed via `animateTo`. This is the API backing up the implementation of `animate*AsState`. It ensures consistent continuation and mutual exclusiveness, meaning that the value change is always continuous and any ongoing animation will be canceled.

Many features of `Animatable`, including `animateTo`, are provided as suspend functions. This means that they need to be wrapped in an appropriate coroutine scope. For example, you can use the `LaunchedEffect` composable to create a scope just for the duration of the specified key value.

// Start out gray and animate to green/red based on `ok`
val color = remember { Animatable(Color.Gray) }
LaunchedEffect(ok) {
    color.animateTo(if (ok) Color.Green else Color.Red)
}
Box(
    Modifier
        .fillMaxSize()
        .background(color.value)
)

In the example above, we create and remember an instance of `Animatable` with the initial value of `Color.Gray`. Depending on the value of the boolean flag `ok`, the color animates to either `Color.Green` or `Color.Red`. Any subsequent change to the boolean value starts animation to the other color. If there's an ongoing animation when the value is changed, the animation is canceled, and the new animation starts from the current snapshot value with the current velocity.

This is the animation implementation that backs up the `animate*AsState` API mentioned in the previous section. Compared to `animate*AsState`, using `Animatable` directly gives us finer-grained control on several respects. First, `Animatable` can have an initial value different from its first target value. For example, the code example above shows a gray box at first, which immediately starts animating to either green or red. Second, `Animatable` provides more operations on the content value, namely `snapTo` and `animateDecay`. `snapTo` sets the current value to the target value immediately. This is useful when the animation itself is not the only source of truth and has to be synced with other states, such as touch events. `animateDecay` starts an animation that slows down from the given velocity. This is useful for implementing fling behavior. See [Gesture and animation](https://developer.android.com/develop/ui/compose/animation/advanced) for more information.

Out of the box, `Animatable` supports `Float` and `Color`, but any data type can be used by providing a `TwoWayConverter`. See [AnimationVector](https://developer.android.com/develop/ui/compose/animation/customize#animationvector) for more information.

You can customize the animation specifications by providing an `AnimationSpec`. See [AnimationSpec](https://developer.android.com/develop/ui/compose/animation/customize#animationspec) for more information.

### `Animation`: Manually controlled animation

[`Animation`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/Animation) is the lowest-level Animation API available. Many of the animations we've seen so far build ontop of Animation. There are two `Animation` subtypes: [`TargetBasedAnimation`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/TargetBasedAnimation) and [`DecayAnimation`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/DecayAnimation).

`Animation` should only be used to manually control the time of the animation. `Animation` is stateless, and it does not have any concept of lifecycle. It serves as an animation calculation engine that the higher-level APIs use.

**Note:** Unless there's a need to control the timing manually, it's generally recommended to use higher level animation APIs that build on top of these classes.

#### `TargetBasedAnimation`

Other APIs cover most use cases, but using `TargetBasedAnimation` directly allows you to control the animation play time yourself. In the example below, the play time of the `TargetAnimation` is manually controlled based on the frame time provided by `withFrameNanos`.

val anim = remember {
    TargetBasedAnimation(
        animationSpec = tween(200),
        typeConverter = Float.VectorConverter,
        initialValue = 200f,
        targetValue = 1000f
    )
}
var playTime by remember { mutableLongStateOf(0L) }

LaunchedEffect(anim) {
    val startTime = withFrameNanos { it }

    do {
        playTime = withFrameNanos { it } - startTime
        val animationValue = anim.getValueFromNanos(playTime)
    } while (someCustomCondition())
}

#### `DecayAnimation`

Unlike `TargetBasedAnimation`, [`DecayAnimation`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/DecayAnimation) does not require a `targetValue` to be provided. Instead, it calculates its `targetValue` based on the starting conditions, set by `initialVelocity` and `initialValue` and the supplied `DecayAnimationSpec`.

Decay animations are often used after a fling gesture to slow elements down to a stop. The animation velocity starts at the value set by `initialVelocityVector` and slows down over time.


# Customize animations

bookmark_border

Many of the Animation APIs commonly accept parameters for customizing their behavior.

## Customize animations with the `AnimationSpec` parameter

Most animation APIs allow developers to customize animation specifications by an optional `AnimationSpec` parameter.

val alpha: Float by animateFloatAsState(
    targetValue = if (enabled) 1f else 0.5f,
    // Configure the animation duration and easing.
    animationSpec = tween(durationMillis = 300, easing = FastOutSlowInEasing),
    label = "alpha"
)

There are different kinds of `AnimationSpec` for creating different types of animation.

### Create physics-based animation with `spring`

`spring` creates a physics-based animation between start and end values. It takes 2 parameters: `dampingRatio` and `stiffness`.

`dampingRatio` defines how bouncy the spring should be. The default value is `Spring.DampingRatioNoBouncy`.

**Figure 1**. Setting different spring damping ratios.

`stiffness` defines how fast the spring should move toward the end value. The default value is `Spring.StiffnessMedium`.

**Figure 2**. Setting different spring stiffness.

val value by animateFloatAsState(
    targetValue = 1f,
    animationSpec = spring(
        dampingRatio = Spring.DampingRatioHighBouncy,
        stiffness = Spring.StiffnessMedium
    ),
    label = "spring spec"
)

`spring` can handle interruptions more smoothly than duration-based `AnimationSpec` types because it guarantees the continuity of velocity when target value changes amid animations. `spring` is used as the default AnimationSpec by many animation APIs, such as `animate*AsState` and `updateTransition`.

For example, if we apply a `spring` config to the following animation that is driven by user touch, when interrupting the animation as its progressing, you can see that using `tween` doesn't respond as smoothly as using `spring`.

**Figure 3**. Setting `tween` versus `spring` specs for animation, and interrupting it.

### Animate between start and end values with easing curve with `tween`

`tween` animates between start and end values over the specified `durationMillis` using an easing curve. `tween` is short for the word between - as it goes _between_ two values.

You can also specify `delayMillis` to postpone the start of the animation.

val value by animateFloatAsState(
    targetValue = 1f,
    animationSpec = tween(
        durationMillis = 300,
        delayMillis = 50,
        easing = LinearOutSlowInEasing
    ),
    label = "tween delay"
)

See [Easing](https://developer.android.com/develop/ui/compose/animation/customize#easing) for more information.

### Animate to specific values at certain timings with `keyframes`

`keyframes` animates based on the snapshot values specified at different timestamps in the duration of the animation. At any given time, the animation value will be interpolated between two keyframe values. For each of these keyframes, Easing can be specified to determine the interpolation curve.

It is optional to specify the values at 0 ms and at the duration time. If you do not specify these values, they default to the start and end values of the animation, respectively.

val value by animateFloatAsState(
    targetValue = 1f,
    animationSpec = keyframes {
        durationMillis = 375
        0.0f at 0 using LinearOutSlowInEasing // for 0-15 ms
        0.2f at 15 using FastOutLinearInEasing // for 15-75 ms
        0.4f at 75 // ms
        0.4f at 225 // ms
    },
    label = "keyframe"
)

### Animate between keyframes smoothly with `keyframesWithSplines`

To create an animation that follows a smooth curve as it transitions between values, you can use `keyframesWithSplines` instead of `keyframes` animation specs.

val offset by animateOffsetAsState(
    targetValue = Offset(300f, 300f),
    animationSpec = keyframesWithSpline {
        durationMillis = 6000
        Offset(0f, 0f) at 0
        Offset(150f, 200f) atFraction 0.5f
        Offset(0f, 100f) atFraction 0.7f
    }
)

Spline-based keyframes are particularly useful for 2D movement of items on screen.

The following videos showcase the differences between `keyframes` and `keyframesWithSpline` given the same set of x, y coordinates that a circle should follow.

|`keyframes`|`keyframesWithSplines`|
|---|---|
|||

As you can see, the spline-based keyframes offer smoother transitions between points, as they use bezier curves to smoothly animate between items. This spec is useful for a preset animation. However,if you're working with user-driven points, it's preferable to use springs to achieve a similar smoothness between points because those are interruptible.

### Repeat an animation with `repeatable`

`repeatable` runs a duration-based animation (such as `tween` or `keyframes`) repeatedly until it reaches the specified iteration count. You can pass the `repeatMode` parameter to specify whether the animation should repeat by starting from the beginning (`RepeatMode.Restart`) or from the end (`RepeatMode.Reverse`).

val value by animateFloatAsState(
    targetValue = 1f,
    animationSpec = repeatable(
        iterations = 3,
        animation = tween(durationMillis = 300),
        repeatMode = RepeatMode.Reverse
    ),
    label = "repeatable spec"
)

### Repeat an animation infinitely with `infiniteRepeatable`

`infiniteRepeatable` is like `repeatable`, but it repeats for an infinite amount of iterations.

val value by animateFloatAsState(
    targetValue = 1f,
    animationSpec = infiniteRepeatable(
        animation = tween(durationMillis = 300),
        repeatMode = RepeatMode.Reverse
    ),
    label = "infinite repeatable"
)

In tests using [`ComposeTestRule`](https://developer.android.com/reference/kotlin/androidx/compose/ui/test/junit4/ComposeTestRule), animations using `infiniteRepeatable` are not run. The component will be rendered using the initial value of each animated value.

### Immediately snap to end value with `snap`

`snap` is a special `AnimationSpec` that immediately switches the value to the end value. You can specify `delayMillis` in order to delay the start of the animation.

val value by animateFloatAsState(
    targetValue = 1f,
    animationSpec = snap(delayMillis = 50),
    label = "snap spec"
)

**Note:** In the View system, you needed to use `ObjectAnimator` for duration-based animations, and `SpringAnimation` for physics-based animation. It was not straightforward to use these two different animation APIs simultaneously. `AnimationSpec` in Compose allows for to handling these in a unified manner.

## Set a custom easing function

Duration-based `AnimationSpec` operations (such as `tween` or `keyframes`) use `Easing` to adjust an animation's fraction. This allows the animating value to speed up and slow down, rather than moving at a constant rate. Fraction is a value between 0 (start) and 1.0 (end) indicating the current point in the animation.

Easing is in fact a function that takes a fraction value between 0 and 1.0 and returns a float. The returned value can be outside the boundary to represent overshoot or undershoot. A custom Easing can be created like the code below.

val CustomEasing = Easing { fraction -> fraction * fraction }

@Composable
fun EasingUsage() {
    val value by animateFloatAsState(
        targetValue = 1f,
        animationSpec = tween(
            durationMillis = 300,
            easing = CustomEasing
        ),
        label = "custom easing"
    )
    // ……
}

Compose provides several built-in `Easing` functions that cover most use cases. See [Speed - Material Design](https://m3.material.io/styles/motion/easing-and-duration/applying-easing-and-duration) for more information about what Easing to use depending on your scenario.

- `FastOutSlowInEasing`
- `LinearOutSlowInEasing`
- `FastOutLinearEasing`
- `LinearEasing`
- `CubicBezierEasing`
- [See more](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/package-summary#Ease\(\))

**Note:** Easing objects work the same way as instances of `Interpolator` classes in the platform. Instead of the `getInterpolation()` method, it has the `transform()` method.

## Animate custom data types by converting to and from `AnimationVector`

Most Compose animation APIs support `Float`, `Color`, `Dp`, and other basic data types as animation values by default, but you sometimes need to animate other data types including your custom ones. During animation, any animating value is represented as an `AnimationVector`. The value is converted into an `AnimationVector` and vice versa by a corresponding `TwoWayConverter` so that the core animation system can handle them uniformly. For example, an `Int` is represented as an `AnimationVector1D` that holds a single float value. `TwoWayConverter` for `Int` looks like this:

val IntToVector: TwoWayConverter<Int, AnimationVector1D> =
    TwoWayConverter({ AnimationVector1D(it.toFloat()) }, { it.value.toInt() })

`Color` is essentially a set of 4 values, red, green, blue, and alpha, so `Color` is converted into an `AnimationVector4D` that holds 4 float values. In this manner, every data type used in animations is converted to either `AnimationVector1D`, `AnimationVector2D`, `AnimationVector3D`, or `AnimationVector4D` depending on its dimensionality. This allows different components of the object to be animated independently, each with their own velocity tracking. Built-in converters for basic data types can be accessed using converters such as `Color.VectorConverter` or `Dp.VectorConverter`.

When you want to add support for a new data type as an animating value, you can create your own `TwoWayConverter` and provide it to the API. For example, you can use `animateValueAsState` to animate your custom data type like this:

data class MySize(val width: Dp, val height: Dp)

@Composable
fun MyAnimation(targetSize: MySize) {
    val animSize: MySize by animateValueAsState(
        targetSize,
        TwoWayConverter(
            convertToVector = { size: MySize ->
                // Extract a float value from each of the `Dp` fields.
                AnimationVector2D(size.width.value, size.height.value)
            },
            convertFromVector = { vector: AnimationVector2D ->
                MySize(vector.v1.dp, vector.v2.dp)
            }
        ),
        label = "size"
    )
}

The following list includes some built-in `VectorConverter`s:

- [`Color.VectorConverter`](https://developer.android.com/reference/kotlin/androidx/compose/ui/graphics/Color.Companion#\(androidx.compose.ui.graphics.Color.Companion\).VectorConverter\(\))
- [`Dp.VectorConverter`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/package-summary#\(androidx.compose.ui.unit.Dp.Companion\).VectorConverter\(\))
- [`Offset.VectorConverter`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/package-summary#\(androidx.compose.ui.geometry.Offset.Companion\).VectorConverter\(\))
- [`Int.VectorConverter`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/package-summary#\(kotlin.Int.Companion\).VectorConverter\(\))
- [`Float.VectorConverter`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/package-summary#\(kotlin.Float.Companion\).VectorConverter\(\))
- [`IntSize.VectorConverter`](https://developer.android.com/reference/kotlin/androidx/compose/animation/core/package-summary#\(androidx.compose.ui.unit.IntSize.Companion\).VectorConverter\(\))
