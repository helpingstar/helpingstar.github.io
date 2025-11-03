---
layout: single
title: "Kotlin Function"
date: 2025-10-10 11:00:00
lastmod : 2025-10-10 11:00:00
categories: android
tag: [android, coroutine]
toc: true
toc_sticky: true
published: false
---

# https://kotlinlang.org/docs/functions.html

# https://kotlinlang.org/docs/lambdas.html

# https://kotlinlang.org/docs/inline-functions.html

# Inline functions

Using [higher-order functions](https://kotlinlang.org/docs/lambdas.html) imposes certain runtime penalties: each function is an object, and it captures a closure. A closure is a scope of variables that can be accessed in the body of the function. Memory allocations (both for function objects and classes) and virtual calls introduce runtime overhead.  
[higher-order functions](https://kotlinlang.org/docs/lambdas.html)을 사용하면 일정한 런타임 비용이 발생합니다. 각 함수는 객체이며, closure를 캡처하기 때문입니다. closure는 함수 본문에서 접근할 수 있는 변수들의 범위를 의미합니다. 함수 객체와 클래스에 대한 메모리 할당, 그리고 가상 호출(virtual call)은 런타임 오버헤드를 유발합니다.


But it appears that in many cases this kind of overhead can be eliminated by inlining the lambda expressions. The functions shown below are good examples of this situation. The `lock()` function could be easily inlined at call-sites. Consider the following case:  
하지만 많은 경우 이러한 종류의 오버헤드는 lambda 표현식을 inline 처리함으로써 제거할 수 있습니다. 아래에 나오는 함수들이 이러한 상황의 좋은 예시입니다. 예를 들어 `lock()` 함수는 호출 지점에서 쉽게 inline될 수 있습니다. 다음 예시를 살펴보세요.


```kotlin
lock(l) { foo() }
```

Instead of creating a function object for the parameter and generating a call, the compiler could emit the following code:  
매개변수를 위한 함수 객체를 생성하고 호출을 발생시키는 대신, 컴파일러는 다음과 같은 코드를 생성할 수 있습니다.


```kotlin
l.lock()
try {
    foo()
} finally {
    l.unlock()
}
```

To make the compiler do this, mark the `lock()` function with the `inline` modifier:

```kotlin
inline fun <T> lock(lock: Lock, body: () -> T): T { ... }
```

The `inline` modifier affects both the function itself and the lambdas passed to it: all of those will be inlined into the call site.

Inlining may cause the generated code to grow. However, if you do it in a reasonable way (avoiding inlining large functions), it will pay off in performance, especially at "megamorphic" call-sites inside loops.  
Inlining은 생성되는 코드의 크기를 증가시킬 수 있습니다. 그러나 적절한 방식으로 사용한다면(예를 들어, 큰 함수를 inline하지 않는다면) 성능 면에서 이점이 있으며, 특히 루프 안의 "megamorphic" 호출 지점에서 효과가 큽니다.

## noinline

If you don't want all of the lambdas passed to an inline function to be inlined, mark some of your function parameters with the `noinline` modifier:

```kotlin
inline fun foo(inlined: () -> Unit, noinline notInlined: () -> Unit) { ... }
```

Inlinable lambdas can only be called inside inline functions or passed as inlinable arguments. `noinline` lambdas, however, can be manipulated in any way you like, including being stored in fields or passed around.  
inline 가능한 lambda는 오직 inline 함수 내부에서 호출되거나 inline 가능한 인자로 전달될 때만 사용할 수 있습니다. 반면 `noinline` lambda는 필드에 저장하거나 전달하는 등 어떤 방식으로든 자유롭게 사용할 수 있습니다.

> If an inline function has no inlinable function parameters and no [reified type parameters](https://kotlinlang.org/docs/inline-functions.html#reified-type-parameters), the compiler will issue a warning, since inlining such functions is very unlikely to be beneficial (you can use the `@Suppress("NOTHING_TO_INLINE")` annotation to suppress the warning if you are sure the inlining is needed).

## Non-local jump expressions

### Returns

In Kotlin, you can only use a normal, unqualified `return` to exit a named function or an anonymous function. To exit a lambda, use a [label](https://kotlinlang.org/docs/returns.html#return-to-labels). A bare `return` is forbidden inside a lambda because a lambda cannot make the enclosing function `return`:  
Kotlin에서는 일반적인(한정되지 않은) `return`을 사용해 이름이 있는 함수나 익명 함수에서만 빠져나올 수 있습니다. lambda에서 빠져나오려면 [label](https://kotlinlang.org/docs/returns.html#return-to-labels)을 사용해야 합니다. lambda 내부에서는 `return`을 단독으로 사용할 수 없습니다. 그 이유는 lambda가 자신을 감싸고 있는 함수의 `return`을 수행할 수 없기 때문입니다.


```kotlin
fun ordinaryFunction(block: () -> Unit) {
    println("hi!")
}
fun foo() {
    ordinaryFunction {
        return // ERROR: cannot make `foo` return here
    }
}
fun main() {
    foo()
}
```
```
'return' is prohibited here.
```

But if the function the lambda is passed to is inlined, the return can be inlined, as well. So it is allowed:

```kotlin
inline fun inlined(block: () -> Unit) {
    println("hi!")
}
fun foo() {
    inlined {
        return // OK: the lambda is inlined
        // 이 곳은 실행되지 않음
    }
    // 이 곳은 실행된다.
}
fun main() {
    foo()
}
```
```
hi!
```


Such returns (located in a lambda, but exiting the enclosing function) are called non-local returns. This sort of construct usually occurs in loops, which inline functions often enclose:  
이처럼 lambda 내부에 있지만 바깥 함수를 종료시키는 `return`을 non-local return이라고 부릅니다. 이러한 구조는 일반적으로 inline 함수가 감싸고 있는 루프(loop) 안에서 자주 발생합니다.

```kotlin
fun hasZeros(ints: List<Int>): Boolean {
    ints.forEach {
        if (it == 0) return true // returns from hasZeros
    }
    return false
}
```

Note that some inline functions may call the lambdas passed to them as parameters not directly from the function body, but from another execution context, such as a local object or a nested function. In such cases, non-local control flow is also not allowed in the lambdas. To indicate that the lambda parameter of the inline function cannot use non-local returns, mark the lambda parameter with the `crossinline` modifier:  
일부 inline 함수는 전달받은 lambda를 함수 본문에서 직접 호출하지 않고, 로컬 객체나 중첩 함수와 같은 다른 실행 컨텍스트에서 호출할 수도 있습니다. 이러한 경우 lambda 내부에서는 non-local 제어 흐름을 사용할 수 없습니다. inline 함수의 lambda 매개변수가 non-local return을 사용할 수 없음을 나타내기 위해 해당 매개변수에 `crossinline` 한정자를 표시합니다.

```kotlin
inline fun f(crossinline body: () -> Unit) {
    val f = object: Runnable {
        override fun run() = body()
    }
    // ...
}
```

### non-cocal return이란?

* 보통 `return`은 그 `return`이 속한 함수에서 빠져나갑니다.
* 그런데 `inline` 함수에 넘긴 람다는 컴파일 시 호출자(호출한 쪽) 코드에 그대로 “붙여넣기(inline)” 됩니다.
  그래서 람다 안에서 `return`을 쓰면 그 람다를 넘긴 바깥 함수(호출자)에서 바로 빠져나가는 — 즉 **non-local return**(비지역 반환)이 됩니다.

예:

```kotlin
inline fun repeatTwice(block: () -> Unit) {
    block()
    block()
}

fun outer() {
    repeatTwice {
        println("A")
        return   // ← 이 return은 outer()를 빠져나간다! (non-local return)
    }
    println("B") // 이 코드는 실행되지 않는다.
}
```

### 그런데 문제가 생기는 상황

람다가 **나중에 다른 컨텍스트에서 호출되면**(예: `Runnable`로 저장하거나 로컬 객체의 메서드에서 호출)
비지역 반환은 더 이상 안전하지 않습니다 — 왜냐면 그 `return`이 빠져나가려는 호출자(outer 함수)의 호출 스택이 이미 없을 수도 있기 때문입니다.

예: 람다를 `Runnable`에 넣는 경우

```kotlin
inline fun runLater(block: () -> Unit) {
    val r = object : Runnable {
        override fun run() = block()   // 람다가 객체 내부에서 호출된다
    }
    r.run()
}
```

이 경우 `block`이 인라인 되지 않고(또는 다른 컨텍스트에서 호출되기 때문에) 바깥 함수로의 non-local return이 허용되면 런타임/논리적으로 문제가 생길 수 있으므로 **컴파일러가 이를 금지**합니다.

#### `Runnable`

```kotlin
public interface Runnable {
    void run()
}
```

- `Runnable` : Java/Kotlin에서 아주 기본적인 인터페이스(interface), 형태는 아주 단순해서 `run()` 이라는 메서드 하나만 가진다.
- 보통 작업(코드 블록)을 실행할 때 그 작업을 나타내는 객체로 사용된다.
  - 예) 스레드에 전달하거나 콜백으로 전달할 때 사용

#### `object : Runnable { override fun run() = ... }`

```kotlin
val r = object : Runnable {
    override fun run() {
        // 여기 코드가 실행될 때 block()이 호출된다.
    }
}
```

- 익명 객체(anonymous object) 를 만드는 Kotlin 문법
- `Runnable`을 구현한 객체를 바로 만들고 그 인스턴스를 `r`에 넣겠다는 뜻이다.
- `object : Runnable { ... }` : 새로운 클래스(이름 없음)을 만들고, 그 클래스가 Runnable을 구현하게 해서 한 번만 인스턴스를 만든다.

```kotlin
inline fun runLater(block: () -> Unit) {
    val r = object : Runnable {
        override fun run() = block()   // (A) 여기서 block()을 호출
    }
    r.run()
}
```

- block()이 (A)처럼 run() 메서드 내부에서 호출된다고 생각해 보세요.
- 만약 람다 block 안에 return이 outer() 같은 바깥 함수를 빠져나가려 한다면, 그 return은 run() 안에서 다른 함수의 흐름을 끊고 바깥함수로 점프하려는 셈이 됩니다.
- 하지만 run()은 별도의 함수(익명 객체의 메서드) 이고, 컴파일러/런타임 입장에서는 run() 내부에서 호출된 코드가 바깥 함수의 제어 흐름을 갑자기 바꿔 버리는 걸 허용할 수 없습니다 — 호출 스택 구조가 맞지 않거나, 호출자(outer 함수)가 이미 끝났을 수도 있기 때문입니다.

좋아요 — 핵심 부분(왜 **`Runnable`**이나 **`object : Runnable`** 같은 다른 실행 컨텍스트에서 람다가 호출되면 non-local `return`이 위험한지) 을 초보자도 이해하기 쉽게 한 번에 정리해드릴게요. 천천히 따라오셔도 됩니다.

---

## 1) `Runnable`이 뭐예요?

* `Runnable`은 Java/Kotlin에서 아주 기본적인 **인터페이스(interface)** 입니다.
  형태는 아주 단순해서 **`run()`** 이라는 메서드 하나만 가집니다:

  ```java
  public interface Runnable {
      void run();
  }
  ```
* 보통 **작업(코드 블록)을 실행할 때 그 작업을 나타내는 객체**로 사용됩니다. 예: 스레드에 전달하거나 콜백으로 전달할 때 씁니다.

---

## 2) `object : Runnable { override fun run() = ... }` 이 무슨 문법인가요?

* 이것은 **익명 객체(anonymous object)** 를 만드는 Kotlin 문법입니다.
* 의미: `Runnable`을 구현한(구체화한) 객체를 바로 만들고 그 인스턴스를 `r`에 넣겠다는 뜻입니다.

  ```kotlin
  val r = object : Runnable {
      override fun run() {
          // 여기 코드가 실행될 때 block()이 호출된다
      }
  }
  ```
* 즉, `object : Runnable { ... }` 은 “새로운 클래스(이름 없음)를 만들고, 그 클래스가 Runnable을 구현하게 해서 한 번만 인스턴스를 만든다” 라고 생각하면 됩니다.

---

## 3) 문제의 본질 — **실행 컨텍스트(호출 위치)가 다르다**는 말의 의미

* 어떤 `return`이 **"non-local"** (비지역)이라는 건: 람다 안의 `return`이 람다 자신을 빠져나오는 게 아니라 **람다를 호출한 바깥 함수(호출자)** 를 즉시 종료시킨다는 뜻입니다.
* 이게 가능한 이유는 `inline` 함수에서는 람다의 코드가 호출자 쪽으로 **문자 그대로 붙여넣기(인라인)** 되기 때문입니다.

  * 예: `inline fun callNow(block: ()->Unit) { block() }` 에서 `block` 몸체는 마치 `callNow`를 호출한 지점에 붙여넣어진 것처럼 동작할 수 있어서 `return`이 그 바깥 함수에서 나가게 됩니다.

근데 **문제는** 람다가 *다른 객체의 메서드 내부*에서 호출될 때입니다. 예를 들어:

```kotlin
inline fun runLater(block: () -> Unit) {
    val r = object : Runnable {
        override fun run() = block()   // (A) 여기서 block()을 호출
    }
    r.run()
}
```

* `block()`이 **(A)처럼 `run()` 메서드 내부**에서 호출된다고 생각해 보세요.
* 만약 람다 `block` 안에 `return`이 `outer()` 같은 바깥 함수를 빠져나가려 한다면, 그 `return`은 `run()` 안에서 **다른 함수의 흐름을 끊고** 바깥함수로 점프하려는 셈이 됩니다.
* **하지만 `run()`은 별도의 함수(익명 객체의 메서드)** 이고, 컴파일러/런타임 입장에서는 `run()` 내부에서 호출된 코드가 바깥 함수의 제어 흐름을 갑자기 바꿔 버리는 걸 허용할 수 없습니다 — 호출 스택 구조가 맞지 않거나, 호출자(outer 함수)가 이미 끝났을 수도 있기 때문입니다.

---

## 4) 직관적 예시(스택 관점)

* 호출 스택(실행 순서)을 단순화하면:

  ```
  outer() 호출
   └─ runLater() 호출
       └─ r.run() 호출   ← run() 내부에서 block() 호출
           └─ block() 실행 (여기서 return으로 outer()를 빠져나가려고 함)
  ```
* `block()`에서 `return`이 `outer()`를 끝내려고 하면 실행 흐름이 `run()`의 정상 종료를 건너뛰고 `outer()` 바로 위로 뛰어올라야 합니다 — **이건 안전하지 않음**.

  * 예: `r.run()`이 나중에 다른 스레드에서 실행되거나, `r` 객체가 저장된 뒤 나중에 호출되는 상황이면 더더욱 위험합니다.

---

## 5) 그래서 컴파일러가 금지하는 이유

* 컴파일러는 “이 람다가 다른 함수(또는 객체의 메서드) 안에서 호출될 수 있다”는 사실을 알고 있으면, 람다 내부에서 **non-local `return`을 허용할 수 없습니다**.
* 그 대신 **컴파일 에러**를 내거나, 개발자가 명시적으로 `crossinline`을 붙여 “이 람다에선 non-local return을 쓰지 않겠다”를 약속하게 합니다.

---

## 6) 해결 방법(안전하게 쓰는 방법들)

1. **`crossinline` 사용**

   * 람다 파라미터에 `crossinline`을 붙이면 컴파일러에게 “이 람다 안에서 non-local return을 금지”한다고 알려줍니다.
   * 그러면 람다 내부에서는 `return`이 안 되고, `return@label` 같은 **라벨(return@...)** 형태로만 람다 자체만 종료할 수 있습니다.

   ```kotlin
   inline fun runLater(crossinline block: () -> Unit) {
       val r = object : Runnable {
           override fun run() = block()
       }
       r.run()
   }
   ```

2. **`noinline` 사용**

   * `noinline`을 붙이면 그 람다는 인라인되지 않고 일반 객체(클래스 인스턴스)로 전달됩니다.
   * 이 상태에선 non-local return 자체가 불가능하므로 컴파일러가 허용하지 않습니다. (안전)

   ```kotlin
   inline fun runLater(noinline block: () -> Unit) { /* ... */ }
   ```

3. **라벨 반환 사용**

   * 람다에서 바깥 함수를 빠져나가는 대신 `return@functionName` 같은 라벨을 사용해 “람다만 종료”하게 만듭니다.

   ```kotlin
   runLater {
       println("start")
       return@runLater   // 람다만 끝나고 outer는 계속 실행
   }
   ```


# 3) `crossinline`의 역할 — 뭐가 달라지나?

* `crossinline`을 붙이면 **그 람다 안에서는 non-local return을 사용할 수 없다**는 것을 컴파일러에 명시합니다.
* 즉 람다는 여전히 인라인 되거나(가능하면) 성능 이득은 취하되, **람다 내부에서 `return`은 오직 람다 자체로의 반환(지역 반환)**만 허용됩니다.
* 따라서 람다를 다른 실행 컨텍스트(객체, 스레드, 콜백 등)에 전달해도 안전합니다.

예 (원문과 같음):

```kotlin
inline fun f(crossinline body: () -> Unit) {
    val runnable = object: Runnable {
        override fun run() = body()   // body 안에서 non-local return을 못 쓰므로 안전
    }
    runnable.run()
}
```

# 4) 구체적 예 — 차이를 직접 비교

### (A) `inline` + **no modifier** → non-local return 가능

```kotlin
inline fun callNow(block: () -> Unit) {
    block()
}

fun foo() {
    callNow {
        println("before")
        return   // foo()에서 바로 빠져나간다 (허용)
    }
    println("after") // 실행 안 됨
}
```

### (B) `inline` + `crossinline` → non-local return 금지

```kotlin
inline fun callLater(crossinline block: () -> Unit) {
    val r = Runnable { block() }
    r.run()
}

fun bar() {
    callLater {
        println("hi")
        return   // 컴파일 에러! "cannot use 'return' from here"
    }
    println("after") // 이 코드는 정상적으로 실행된다, 물론 컴파일이 안됨
}
```

**대신** 람다 내부에서 람다만 빠져나가려면 라벨을 써야 합니다:

```kotlin
callLater {
    println("before")
    return@callLater   // 람다만 종료하고 바깥 함수는 계속 실행
    println("after")   // 실행 안 됨
}
```

# 5) `noinline`과의 비교 (참고)

* `noinline`을 붙이면 **그 람다는 인라인되지 않는다**(즉 일반 객체로 전달). 이 경우 non-local return 자체가 애초에 불가능합니다.
* `crossinline`은 **인라인은 하되** non-local return만 금지하는 중간 성격입니다.

# 6) 왜 이렇게 설계했나 (정리)

* 비지역 반환(non-local return)은 편리하지만, 람다가 **다른 실행 컨텍스트에서 호출될 때**는 동작 원리가 깨질 수 있다.
* `crossinline`은 그런 상황에서 안전하게 람다를 다른 컨텍스트로 전달할 수 있게 해준다.
* 개발자는 "이 람다에서 non-local return을 사용하지 않겠다"는 의사를 명시적으로 표현하는 것.

### Break and continue

Similar to non-local `return`, you can apply `break` and `continue` [jump expressions](https://kotlinlang.org/docs/returns.html) in lambdas passed as arguments to an inline function that encloses a loop:  
non-local `return`과 마찬가지로, 루프를 감싸는 inline 함수에 인자로 전달된 lambda 안에서도 [`break`와 `continue` 같은 점프 표현식(jump expressions)](https://kotlinlang.org/docs/returns.html)을 사용할 수 있습니다.


```kotlin
fun processList(elements: List<Int>): Boolean {
    for (element in elements) {
        val variable = element.nullableMethod() ?: run {
            log.warning("Element is null or invalid, continuing...")
            continue
        }
        if (variable == 0) return true
    }
    return false
}
```

## Reified type parameters

Sometimes you need to access a type passed as a parameter:

```kotlin
fun <T> TreeNode.findParentOfType(clazz: Class<T>): T? {
    var p = parent
    while (p != null && !clazz.isInstance(p)) {
        p = p.parent
    }
    @Suppress("UNCHECKED_CAST") // 컴파일러 경고를 억지로 숨기는 어노테이션
    return p as T?
}
```
- JVM에서 제네릭은 타입 소거(type erasure) 때문에 런타임에 제네릭 타입 정보를 알 수 없을 때가 많습니다.
  - JVM의 제네릭은 설계 당시의 하위 호환성 때문에 **런타임에 제네릭 타입 정보를 보관하지 않도록** 되어 있습니다(이를 type erasure라 부릅니다). 컴파일러가 타입을 검사하고 안전한 코드를 생성하지만, 실행 중에는 구체 타입 매개변수(`String`, `Int` 등)가 제거됩니다.
- T가 런타임에는 사라지기 때문에 컴파일러가 정말 안전한지 확인할 수 없어서 경고를 냅니다. 그러나 개발자가 “내가 안전한 걸 알고 있다”면 경고를 없애고 싶을 수 있고, 그때 `@Suppress("UNCHECKED_CAST")`를 씁니다.



Here, you walk up a tree and use reflection to check whether a node has a certain type. It's all fine, but the call site is not very pretty:

```kotlin
treeNode.findParentOfType(MyTreeNode::class.java)
```

A better solution would be to simply pass a type to this function. You can call it as follows:

```kotlin
treeNode.findParentOfType<MyTreeNode>()
```

To enable this, inline functions support reified type parameters, so you can write something like this:

```kotlin
inline fun <reified T> TreeNode.findParentOfType(): T? {
    var p = parent
    while (p != null && p !is T) {
        p = p.parent
    }
    return p as T?
}
```

The code above qualifies the type parameter with the `reified` modifier to make it accessible inside the function, almost as if it were a normal class. Since the function is inlined, no reflection is needed and normal operators like `!is` and `as` are now available for you to use. Also, you can call the function as shown above: `myTree.findParentOfType<MyTreeNodeType>()`.  
위 코드에서는 타입 매개변수에 `reified` 한정자를 지정하여, 함수 내부에서 마치 일반 클래스처럼 해당 타입에 접근할 수 있도록 합니다. 함수가 inline되기 때문에 reflection이 필요하지 않으며, `!is`나 `as` 같은 일반 연산자를 그대로 사용할 수 있습니다. 또한 다음과 같이 함수를 호출할 수도 있습니다: `myTree.findParentOfType<MyTreeNodeType>()`.


Though reflection may not be needed in many cases, you can still use it with a reified type parameter:

```kotlin
inline fun <reified T> membersOf() = T::class.members

fun main(s: Array<String>) {
    println(membersOf<StringBuilder>().joinToString("\n"))
}
```

Normal functions (not marked as inline) cannot have reified parameters. A type that does not have a run-time representation (for example, a non-reified type parameter or a fictitious type like `Nothing`) cannot be used as an argument for a reified type parameter.  
일반 함수(즉, `inline`으로 표시되지 않은 함수)는 `reified` 매개변수를 가질 수 없습니다. 또한 런타임 표현(run-time representation)이 없는 타입(예: `reified`되지 않은 타입 매개변수나 `Nothing`과 같은 가상의 타입)은 `reified` 타입 매개변수의 인자로 사용할 수 없습니다.


## Inline properties

The `inline` modifier can be used on accessors of properties that don't have [backing fields](https://kotlinlang.org/docs/properties.html#backing-fields). You can annotate individual property accessors:






```kotlin
val foo: Foo
    inline get() = Foo()
```

* 호출부에서 `val x = foo`가 실행되면, 컴파일러는 `Foo()` 호출 코드를 `foo`의 getter를 호출하는 지점에 **그대로 넣습니다**.
* 즉 런타임에서 별도의 `getFoo()` 메서드를 호출하지 않고 `Foo()`가 직접 실행됩니다.

```kotlin
var bar: Bar
    get() = ...
    inline set(v) { ... }
```
* `bar = something` 는 `set(value)` 호출이 아니라 `store(something)` 코드로 치환(inline)됩니다.

You can also annotate an entire property, which marks both of its accessors as `inline`:

```kotlin
inline var bar: Bar
    get() = ...
    set(v) { ... }
```
- 위와 동일하게 `get`과 `set` 모두 인라인 처리.

1. **백킹 필드가 없어야 함**

   * 프로퍼티가 자동 생성된 `field`를 사용하면(예: `val x = 10` 또는 `var y: Int = 0`) 인라인 accessor를 못 씁니다.
   * 이유: `inline`은 접근자 본문을 호출 지점으로 붙여넣는데, `field`는 그 프로퍼티의 실제 저장소에 직접 접근해야 하므로 의미상 충돌합니다.

2. **액세서 단위로 표시 가능**

   * `get`만 inline으로 만들거나 `set`만 inline으로 만들 수 있습니다. 또는 프로퍼티 전체에 `inline`을 붙여 둘 다 inline 처리할 수 있습니다.

3. **동일한 장단점이 적용**

   * 장점: 호출 오버헤드 제거(짧은 getter/setter에서 유용).
   * 단점: 호출 지점이 많으면 코드 크기 증가(인라인된 본문이 중복) — 함수 인라인의 trade-off와 동일.

4. **동작은 함수 인라인과 동일**

   * 예외(try/catch)나 non-local return 등에 대한 규칙은 함수 인라인과 동일하게 적용됩니다(예: 람다/return 관련 제약은 그 문맥에 따라 동일하게 고려).



At the call site, inline accessors are inlined as regular inline functions.

## Restrictions for public API inline functions

When an inline function is `public` or `protected` but is not a part of a `private` or `internal` declaration, it is considered a [module](https://kotlinlang.org/docs/visibility-modifiers.html#modules)'s public API. It can be called in other modules and is inlined at such call sites as well.  
inline 함수가 `public` 또는 `protected`이면서 `private`이나 `internal` 선언의 일부가 아닐 경우, 이는 [module](https://kotlinlang.org/docs/visibility-modifiers.html#modules)의 public API로 간주됩니다. 이러한 함수는 다른 모듈에서도 호출될 수 있으며, 해당 호출 지점에서도 inline 처리됩니다.


This imposes certain risks of binary incompatibility caused by changes in the module that declares an inline function in case the calling module is not re-compiled after the change.   
이 경우, inline 함수를 선언한 모듈이 변경되었는데 호출하는 모듈이 다시 컴파일되지 않는다면, 이진 호환성(binary incompatibility) 문제가 발생할 위험이 있습니다.


To eliminate the risk of such incompatibility being introduced by a change in a non-public API of a module, public API inline functions are not allowed to use non-public-API declarations, i.e. `private` and `internal` declarations and their parts, in their bodies.  
모듈의 비공개 API(non-public API) 변경으로 인해 이러한 호환성 문제가 발생하는 것을 방지하기 위해, public API inline 함수는 함수 본문에서 `private`이나 `internal` 선언(또는 그 구성 요소)을 사용할 수 없습니다.


An `internal` declaration can be annotated with `@PublishedApi`, which allows its use in public API inline functions. When an `internal` inline function is marked as `@PublishedApi`, its body is checked too, as if it were public.  
`internal` 선언은 `@PublishedApi` annotation을 사용하여 public API inline 함수 내에서 사용할 수 있도록 허용할 수 있습니다. 또한 `internal` inline 함수에 `@PublishedApi`가 지정되면, 해당 함수의 본문 역시 public 함수와 동일한 방식으로 검사됩니다.


# https://kotlinlang.org/docs/operator-overloading.html


# https://kotlinlang.org/docs/type-safe-builders.html

# https://kotlinlang.org/docs/using-builders-with-builder-inference.html