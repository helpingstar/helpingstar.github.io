---
layout: single
title: "Kotlin Coroutine"
date: 2025-10-10 11:00:00
lastmod : 2025-10-10 11:00:00
categories: android
tag: [android, coroutine]
toc: true
toc_sticky: true
published: false
---

# 알렉세이 세두노프, 『코틀린 완벽 가이드』(길벗, 2022)

## 13.1 코루틴

코블린 프로그램에서도 자바 동시성 기본 요소를 쉽게 사용해 스레드 안정성을 달성할 수 있다.

하지만 자바 동시성 요소를 사용해도 대부분의 동시성 연산이 블로킹(blocking) 연산이므로 여전히 몇 가지 문제가 남는다. 다른 말로 설명하면 `Thread sleep()` `Thread join()` `Object wait()`는 실행이 끝날 때까지 블락된다. 스레드를 블락하고 나중에 실행을 재개하려면 시스템 수준에서 계산 비용이 많이 드는 문맥 전환(context switch)을 해야 하므로 프로그램 성능에 부정적인 영향을 미칠 수 있다. 실상가상으로 스레드마다 상당한 양의 시스템 자원을 유지해야 하기 때문에 동시성 스레드를 아주 많이 사용하는 것은 비실용적이거나 (운영체제나 시스템 종류에 따라) 아예 불가능할 수도 있다.

더 효율적인 접근 방법은 비동기(asynchronous) 프로그램이다. 동시성 연산에 대해 해당 연산이 완료될 때 호출될 수 있는 람다를 제공할 수 있고, 원래 스레드는 블라킹 상태로 작업 완료를 기다리는 대신 (고객 요청을 처리하거나 UI 이벤트를 처리하는 등의) 다른 유용한 작업을 계속 수행할 수 있다. 이런 접근 방법의 가장 큰 문제점은 일반적인 명령형 제어 흐름을 사용할 수 없어서 코드 복잡도가 엄청나게 높아진다는 것이다.

코틀린에서는 두 접근 방법의 장점을 함께 취할 수 있다. 코루틴이라는 강력한 메커니즘 덕분에 우리에게 익숙한 명령형 스타일로 코드를 작성하면 컴파일러가 코드를 효율적인 비동기 계산으로 자동 변환해준다. 이런 메커니즘은 실행을 잠시 중단했다가 나중에 중단한 지점부터 실행을 다시 재개할 수 있는 일시 중단 가능한 함수라는 개념을 중심으로 이뤄진다.

대부분의 코루틴 가능이 별도 라이브러리로 제공되기 때문에 명시적으로 프로젝트 설정에 이들 추가해야 한다. 이 책에서 사용하는 버전은 `org.jetbrains.kotlin:kotlinx-coroutines-core:1.4.3`이라는 메이븐 좌표(Maven coordinate)를 통해 사용할 수 있다.

### 13.1.1 코루틴과 일시 중단 함수

자체 코루틴 라이브러리를 뒷받침하는 기본 요소는 일시 중단 함수다. 이 함수는 일반적인 함수를 더 일반화해 함수 본문의 원하는 지점에서 함수에 필요한 모든 런타임 문맥을 저장하고 함수 실행을 중단한 다음, 나중에 필요할 때 다시 실행을 계속 진행할 수 있게 한 것이다. 코틀린에서는 이런 함수에 `suspend`라는 변경자를 붙인다.

```kotlin
suspend fun foo() {
  println("Task started")
  delay(100)
  println("Task finished")
}
```

`delay()` 함수는 코루틴 라이브러리에 정의된 일시 중단 함수다. 이 함수는 `Thread.sleep()`과 비슷한 일을 한다. 하지만 `delay()`는 현재 스레드를 블락시키지 않고 자신을 호출한 함수를 일시 중단시키며 스레드를 (다른 일시 중단된 함수를 다시 계속 실행하는 등의) 다른 작업을 수행할 수 있게 풀어준다.

일시 중단 함수는 일시 중단 함수와 일반 함수를 원하는 대로 호출할 수 있다. 일시 중단 함수를 호출하면 해당 호출 지점이 일시 중단 지점이 된다. 일시 중단 지점은 임시로 실행을 중단했다가 나중에 재개할 수 있는 지점을 말한다. 반면 일반 함수 호출은 (지금까지 우리가 다뤄온) 일반 함수처럼 작동해서 함수 실행이 다 끝난 뒤 호출한 함수로 제어가 돌아온다. 반면에 코틀린은 일반 함수가 일시 중단 함수를 호출하는 것을 금지한다.

```kotlin
fun foo() {
  println("Task started")
  delay(100) // error: delay is a suspend function
  println("Task finished")
}
```

일시 중단 함수를 일시 중단 함수에서만 호출할 수 있다면, 어떻게 일반 함수에서 일시 중단 함수를 호출할 수 있을까? 가장 분명한 방법은 `main()`을 `suspend`로 표시하는 것이다.

```kotlin
import kotlinx.coroutines.delay

suspend fun main() {
  println("Task started")
  delay(100)
  println("Task finished")
}
```

이 코드를 실행하면 예상대로 다음 두 문장이 100밀리초 간격으로 표시되는 것을 볼 수 있다.

```
Task started
Task finished
```

그러나 현실적인 경우 동시성 코드의 동작을 제어하고 싶기 때문에 공통적인 생명 주기(life cycle)와 문맥이 정해진 몇몇 작업(task)이 정의된 구체적인 영역 안에서만 동시성 함수를 호출한다. (이런 구체적 영역을 제공하기 위해) 코루틴을 실행할 때 사용하는 여러 가지 함수를 코루틴 빌더(coroutine builder)라고 부른다. 코루틴 빌더는 `CoroutineScope` 인스턴스의 확장 함수로 쓰인다. `CoroutineScope`에 대한 구현 중 가장 기본적인 것으로 `GlobalScope` 객체가 있다. `GlobalScope` 객체를 사용하면 독립적인 코루틴을 만들 수 있고, 이 코루틴은 자신만의 작업을 내포할 수 있다. 이제 자주 사용하는 `launch()`, `async()`, `runBlocking()`이라는 코루틴 빌더를 살펴보자.

### 13.1.2 코루틴 빌더

`launch()` 함수는 코루틴을 시작하고, 코루틴을 실행 중인 작업의 상태를 추적하고 변경할 수 있는 `Job` 객체를 돌려준다. 이 함수는 `CoroutineScope.() -> Unit` 타입의 일시 중단 람다를 받는다. 이 람다는 새 코루틴의 본문에 해당한다. 간단한 예제를 보자.

```kotlin
import kotlinx.coroutines.*
import java.lang.System.*

fun main() {  // main이 suspend 함수가 아님에 유의
  val time = currentTimeMillis()
  
  GlobalScope.launch {
    delay(100)
    println("Task 1 finished in ${currentTimeMillis() - time} ms")
  }
  
  GlobalScope.launch {
    delay(100)
    println("Task 2 finished in ${currentTimeMillis() - time} ms")
  }
  
  Thread.sleep(200)
}
```

이 코드를 실행하면 다음과 같은 동작을 볼 수 있다.

```
Task 2 finished in 176 ms
Task 1 finished in 176 ms
```

두 작업이 프로그램을 시작한 시점을 기준으로 거의 동시에 끝났다는 점에서 알 수 있는 것처럼 두 작업이 실제로 병렬적으로 실행했다는 점에 주목해보자. 다만 실행 순서가 항상 일정하게 보장되지는 않으므로 상황에 따라 둘 중 어느 한쪽이 더 먼저 표시될 수 있다. 코루틴 라이브러리는 필요할 때 실행 순서를 강제할 수 있는 도구도 제공한다. 이는 동시성 통신에 관련한 절에서 설명한다.

`main()` 함수 자체는 `Thread.sleep()`을 통해 메인 스레드 실행을 잠시 중단한다. 이를 통해 코루틴 스레드가 완료될 수 있도록 충분한 시간을 제공한다. 코루틴을 처리하는 스레드는 데몬 모드(daemon mode)로 실행되기 때문에 `main()` 스레드가 이 스레드보다 빨리 끝나버리면 자동으로 실행이 종료된다.

일시 중단 함수의 내부에서 `sleep()`과 같은 스레드를 블락시키는 함수를 실행할 수도 있지만, 그런 식의 코드는 코루틴을 사용하는 목적에 위배된다는 점을 염두에 뒤야 한다. 그래서 동시성 작업의 내부에서는 일시 중단 함수인 `delay()`를 사용해야 한다.

코루틴은 스레드보다 훨씬 가볍다. 특히 코루틴은 유지해야 하는 상태가 더 간단하며 일시 중단되고 재개될 때 완전한 문맥 전환을 사용하지 않아도 되므로 엄청난 수의 코루틴을 중분히 동시에 실행할 수 있다.

- `launch()` 빌더
  - 동시성 작업이 결과를 만들어내지 않는 경우 적합
  - `Unit` 타입을 반환하는 람다를 인자로 받음
- `async()` 빌더
  - 결과가 필요한 경우 사용
  - `Deferred`의 인스턴스를 돌려줌
- `Deferred` 인스턴스
  - `Job`의 하위 타입으로 `await()` 메서드를 통해 계산 결과에 접근할 수 있게 해준다. 
  - `await()` 메서드를 호출하면 `await()`는 계산이 완료되거나(따라서 결과가 만들어지거나) 계산 작업이 취소될 때까지 현재 코루틴을 일시 중단시킨다. 
  - 작업이 취소되는 경우 `await()`는 예외를 발생시키면서 실패한다. 

`async()`를 자바의 퓨처(future)에 해당하는 코루틴 빌더라고 생각할 수 있다. 예제를 살펴보자.

```kotlin
import kotlinx.coroutines.*

suspend fun main() {
  val message = GlobalScope.async {
    delay(100)
    "abc"
  }
  
  val count = GlobalScope.async {
    delay(100)
    1 + 2
  }
  
  delay(200)
  
  val result = message.await().repeat(count.await())
  println(result)
}
```
실행 흐름

1. `message` coroutine → 100ms 뒤 `"abc"`
2. `count` coroutine → 100ms 뒤 3
3. `main` coroutine → 200ms 대기
4. 두 `async` 결과를 `await()` 해서 가져옴
5. `"abc".repeat(3)` → `"abcabcabc"` 출력

`fun CharSequence.repeat(n: Int): String`
  - 매개변수 : 반복할 횟수 (정수형, Int)
  - 반환값 : 원본 문자열을 n번 반복한 결과 문자열을 반환

이 경우에는 `main()`을 `suspend`로 표시해서 두 `Deferred` 작업에 대해 직접 `await()` 메서드를 호출했다. 출력은 기대한 대로 다음과 같다.

```
abcabcabc
```

`launch()`와 `async()` 빌더의 경우 (일시 중단 함수 내부에서) 스레드 호출을 블럭시키지는 않지만, 백그라운드 스레드를 공유하는 풀(pool)을 통해 작업을 실행한다. 앞에서 살펴본 `launch()` 예제에서는 메인 스레드가 처리할 일이 별로 없었기 때문에 `sleep()`을 통해 백그라운드 스레드에서 실행되는 작업이 완료될 때까지 기다려야 했다. 반대로 

`runBlocking()` 빌더
- 디폴트로 현재 스레드에서 실행되는 코루틴을 만들고 코루틴이 완료될 때까지 현재 스레드의 실행을 블럭시킨다. 
- 코루틴이 성공적으로 끝나면 일시 중단 람다의 결과가 `runBlocking()` 호출의 결괏값이 된다. 
- 코루틴이 취소되면 `runBlocking()`은 예외를 던진다. 
- 반면에 블럭된 스레드가 인터럽트되면 `runBlocking()`에 의해 시작된 코루틴도 취소된다.

```kotlin
import kotlinx.coroutines.*

fun main() {
  GlobalScope.launch {
    delay(100)
    println("Background task: ${Thread.currentThread().name}")
  }
  runBlocking {
    println("Primary task: ${Thread.currentThread().name}")
    delay(200)
  }
}
```

이 프로그램을 실행하면 다음과 비슷한 결과를 볼 수 있다.

```
Primary task: main
Background task: DefaultDispatcher-worker-1
```

`runBlocking()` 내부의 코루틴은 메인 스레드에서 실행된 반면, `launch()`로 시작한 코루틴은 공유 풀에서 백그라운드 스레드를 할당받았음을 알 수 있다.

이런 블러킹 동작 때문에 `runBlocking()`을 다른 코루틴 안에서 사용하면 안 된다.

`runBlocking()`
- 블러킹 호출과 넌블러킹 호출 사이의 다리 역할을 하기 위해 고안된 코루틴 빌더
- 테스트나 메인 함수에서 최상위 빌더로 사용하는 등의 경우에만 `runBlocking()`을 써야 한다.

### 13.1.3 코루틴 영역과 구조적 동시성

지금까지 살펴본 예제 코루틴은 전역 영역(global scope)에서 실행했다. 전역 영역이란 코루틴의 생명 주기가 전체 애플리케이션의 생명 주기에 의해서만 제약되는 영역이다. 경우에 따라서는 코루틴이 어떤 연산을 수행하는 도중에만 실행되길 바랄 수도 있다. 동시성 작업 사이의 부모 자식 관계로 인해 이런 실행 시간 제한이 가능하다. 어떤 코루틴을 다른 코루틴의 문맥에서 실행하면 후자가 전자의 부모가 된다. 이 경우 자식의 실행이 모두 끝나야 부모가 끝날 수 있도록 부모와 자식의 생명 주기가 연관된다.

이런 기능을 구조적 동시성(structured concurrency)이라고 부르며, 지역 변수 영역 안에서 블럭이나 서브루틴을 사용하는 경우와 구조적 동시성을 비교할 수 있다. 예제를 살펴보자.

```kotlin
import kotlinx.coroutines.*

fun main() {
  runBlocking {
    println("Parent task started")
    launch {
      println("Task A started")
      delay(200)
      println("Task A finished")
    }
    launch {
      println("Task B started")
      delay(200)
      println("Task B finished")
    }
    delay(100)
    println("Parent task finished")
  }
  println("Shutting down...")
}
```

이 코드는 최상위 코루틴을 시작하고 현재 `CoroutineScope` 인스턴스 안에서 `launch`를 호출(영역 객체는 일시 중단 람다의 수신 객체로 전달된다)해 두 가지 자식 코루틴을 시작한다. 이 프로그램을 실행하면 다음 결과를 볼 수 있다.

```
Parent task started
Task A started
Task B started
Parent task finished
Task A finished
Task B finished
Shutting down...
```

지연을 100밀리초만 줬기 때문에 `runBlocking()` 호출의 일시 중단 람다로 이뤄진 부모 코루틴의 주 본문이 더 빨리 끝난다. 하지만 부모 코루틴 자체는 이 시점에 실행이 끝나지 않고 일시 중단 상태로 두 자식이 모두 끝날 때까지 기다린다. `runBlocking()`이 메인 스레드를 블럭하고 있었기 때문에 부모 스레드가 끝나야 메인 스레드의 블럭이 풀리고 마지막 메시지가 출력된다.

`coroutineScope()` 호출로 코드 블럭을 감싸면 커스텀 영역을 도입할 수도 있다. `runBlocking()`과 마찬가지로 `coroutineScope()` 호출은 람다의 결과를 반환하고, 자식들이 완료되기 전까지 실행이 완료되지 않는다. `coroutineScope()`와 `runBlocking()`의 가장 큰 차이는 `coroutineScope()`가 일시 중단 함수라 현재 스레드를 블럭시키지 않는다는 점이다.

```kotlin
import kotlinx.coroutines.*

fun main() {
  runBlocking {
    println("Custom scope start")
    coroutineScope {
      launch {
        delay(100)
        println("Task 1 finished")
      }
      
      launch {
        delay(100)
        println("Task 2 finished")
      }
    }
    println("Custom scope end")
  }
}
```

```
Custom scope start
Task 1 finished
Task 2 finished
Custom scope end
```

두 자식 코루틴 실행이 끝날 때까지 앞의 `coroutineScope()` 호출이 일시 중단되므로 `Custom scope end` 메시지가 마지막에 표시된다.

일반적으로 부모 자식 관계는 예외 처리와 취소 요청을 공유하는 영역을 정의하는 더 복잡한 코루틴 계층 구조를 만들어낼 수 있다. 이 주제는 코루틴 잡(job)과 취소를 설명할 때 다시 다루겠다.

# https://kotlinlang.org/docs/coroutines-basics.html (16 February 2022, Running on v.2.2.20)

# Coroutines basics

To create applications that perform multiple tasks at once, a concept known as concurrency, Kotlin **uses coroutines**. A coroutine is a suspendable computation that lets you write concurrent code in a clear, sequential style. Coroutines can run concurrently with other coroutines and potentially in parallel.

On the JVM and in Kotlin/Native, all concurrent code, such as coroutines, runs on **threads**, managed by the operating system. Coroutines can suspend their execution instead of blocking a thread. This allows one coroutine to suspend while waiting for some data to arrive and another coroutine to run on the same thread, ensuring effective resource utilization.

![Comparing parallel and concurrent threads](https://kotlinlang.org/docs/images/parallelism-and-concurrency.svg "Comparing parallel and concurrent threads")

For more information about the differences between coroutines and threads, see [Comparing coroutines and JVM threads](https://kotlinlang.org/docs/coroutines-basics.html#comparing-coroutines-and-jvm-threads).

## Suspending functions

The most basic building block of coroutines is the **suspending function**. It allows a running operation to pause and resume later without affecting the structure of your code.

To declare a suspending function, use the `suspend` keyword:

```kotlin
suspend fun greet() {
    println("Hello world from a suspending function")
}
```

You can only call a suspending function from another suspending function. To call suspending functions at the entry point of a Kotlin application, mark the `main()` function with the `suspend` keyword:

```kotlin
suspend fun main() {
    showUserInfo()
}

suspend fun showUserInfo() {
    println("Loading user...")
    greet()
    println("User: John Smith")
}

suspend fun greet() {
    println("Hello world from a suspending function")
}
```
```
Loading user...
Hello world from a suspending function
User: John Smith
```

This example doesn't use concurrency yet, but by marking the functions with the `suspend` keyword, you allow them to call other suspending functions and run concurrent code inside.

While the `suspend` keyword is part of the core Kotlin language, most coroutine features are available through the `kotlinx.coroutines`

## Add the kotlinx.coroutines library to your project

To include the `kotlinx.coroutines` library in your project, add the corresponding dependency configuration based on your build tool:


```kotlin
// build.gradle.kts
repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
}
```

## Create your first coroutines

> The examples on this page use the explicit `this` expression with the coroutine builder functions `CoroutineScope.launch()` and `CoroutineScope.async()`. These coroutine builders are [extension functions](https://kotlinlang.org/docs/extensions.html) on `CoroutineScope`, and the `this` expression refers to the current `CoroutineScope` as the receiver.

To create a coroutine in Kotlin, you need the following:

- A [suspending function](https://kotlinlang.org/docs/coroutines-basics.html#suspending-functions).
- A [coroutine scope](https://kotlinlang.org/docs/coroutines-basics.html#coroutine-scope-and-structured-concurrency) in which it can run, for example inside the `withContext()` function.
- A [coroutine builder](https://kotlinlang.org/docs/coroutines-basics.html#coroutine-builder-functions) like `CoroutineScope.launch()` to start it.
- A [dispatcher](https://kotlinlang.org/docs/coroutines-basics.html#coroutine-dispatchers) to control which threads it uses.

Let's look at an example that uses multiple coroutines in a multithreaded environment:

1. Import the `kotlinx.coroutines` library:
    ```kotlin
    import kotlinx.coroutines.*
    ```
2. Mark functions that can pause and resume with the `suspend` keyword:
    ```kotlin
    suspend fun greet() {
        println("The greet() on the thread: ${Thread.currentThread().name}")
    }
    suspend fun main() {}
    ```
    > While you can mark the `main()` function as `suspend` in some projects, it may not be possible when integrating with existing code or using a framework. In that case, check the framework's documentation to see if it supports calling suspending functions. If not, use [`runBlocking()`](https://kotlinlang.org/docs/coroutines-basics.html#runblocking) to call them by blocking the current thread.
3. Add the [`delay()`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/delay.html#) function to simulate a suspending task, such as fetching data or writing to a database:
    ```kotlin
    suspend fun greet() {
        println("The greet() on the thread: ${Thread.currentThread().name}")
        delay(1000L)
    }
    ```
4. Use [`withContext(Dispatchers.Default)`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-context.html#) to define an entry point for multithreaded concurrent code that runs on a shared thread pool:
    ```kotlin
    suspend fun main() {
        withContext(Dispatchers.Default) {
            // Add the coroutine builders here
        }
    }
    ```
    > The suspending `withContext()` function is typically used for [context switching](https://kotlinlang.org/docs/coroutine-context-and-dispatchers.html#jumping-between-threads), but in this example, it also defines a non-blocking entry point for concurrent code. It uses the [`Dispatchers.Default` dispatcher](https://kotlinlang.org/docs/coroutines-basics.html#coroutine-dispatchers) to run code on a shared thread pool for multithreaded execution. By default, this pool uses up to as many threads as there are CPU cores available at runtime, with a minimum of two threads.
    > 
    > The coroutines launched inside the `withContext()` block share the same coroutine scope, which ensures [structured concurrency](https://kotlinlang.org/docs/coroutines-basics.html#coroutine-scope-and-structured-concurrency).
5. Use a [coroutine builder function](https://kotlinlang.org/docs/coroutines-basics.html#coroutine-builder-functions) like [`CoroutineScope.launch()`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/launch.html) to start the coroutine:
    ```kotlin
    suspend fun main() {
        withContext(Dispatchers.Default) { // this: CoroutineScope
            // Starts a coroutine inside the scope with CoroutineScope.launch()
            this.launch { greet() }
            println("The withContext() on the thread: ${Thread.currentThread().name}")
        }
    }
    ```
6. Combine these pieces to run multiple coroutines at the same time on a shared pool of threads:
    ```kotlin
    // Imports the coroutines library
    import kotlinx.coroutines.*

    // Imports the kotlin.time.Duration to express duration in seconds
    import kotlin.time.Duration.Companion.seconds

    // Defines a suspending function
    suspend fun greet() {
        println("The greet() on the thread: ${Thread.currentThread().name}")
        // Suspends for 1 second and releases the thread
        delay(1.seconds) 
        // The delay() function simulates a suspending API call here
        // You can add suspending API calls here like a network request
    }

    suspend fun main() {
        // Runs the code inside this block on a shared thread pool
        withContext(Dispatchers.Default) { // this: CoroutineScope
            this.launch() {
                greet()
            }
    ​        // Starts another coroutine
            this.launch() {
                println("The CoroutineScope.launch() on the thread: ${Thread.currentThread().name}")
                delay(1.seconds)
                // The delay function simulates a suspending API call here
                // You can add suspending API calls here like a network request
            }
    ​
            println("The withContext() on the thread: ${Thread.currentThread().name}")
        }
    }
    ```
    ```
    The withContext() on the thread: DefaultDispatcher-worker-1
    The greet() on the thread: DefaultDispatcher-worker-2 @coroutine#1
    The CoroutineScope.launch() on the thread: DefaultDispatcher-worker-1 @coroutine#2
    ```

Try running the example multiple times. You may notice that the output order and thread names may change each time you run the program, because the OS decides when threads run.

> You can display coroutine names next to thread names in the output of your code for additional information. To do so, pass the `-Dkotlinx.coroutines.debug` VM option in your build tool or IDE run configuration.
> 
> See [Debugging coroutines](https://github.com/Kotlin/kotlinx.coroutines/blob/master/docs/topics/debugging.md) for more information.

## Coroutine scope and structured concurrency

When you run many coroutines in an application, you need a way to manage them as groups. Kotlin coroutines rely on a principle called **structured concurrency** to provide this structure.

According to this principle, coroutines form a tree hierarchy of parent and child tasks with linked lifecycles. A coroutine's lifecycle is the sequence of states from its creation until completion, failure, or cancellation.

A parent coroutine waits for its children to complete before it finishes. If the parent coroutine fails or gets canceled, all its child coroutines are recursively canceled too. Keeping coroutines connected this way makes cancellation and error handling predictable and safe.

To maintain structured concurrency, new coroutines can only be launched in a [`CoroutineScope`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-coroutine-scope/) that defines and manages their lifecycle. The `CoroutineScope` includes the coroutine context, which defines the dispatcher and other execution properties. When you start a coroutine inside another coroutine, it automatically becomes a child of its parent scope.

Calling a [coroutine builder function](https://kotlinlang.org/docs/coroutines-basics.html#coroutine-builder-functions), such as `CoroutineScope.launch()` on a `CoroutineScope`, starts a child coroutine of the coroutine associated with that scope. Inside the builder's block, the [receiver](https://kotlinlang.org/docs/lambdas.html#function-literals-with-receiver) is a nested `CoroutineScope`, so any coroutines you launch there become its children.

### Create a coroutine scope with the `coroutineScope()` function

To create a new coroutine scope with the current coroutine context, use the [`coroutineScope()`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/coroutine-scope.html) function. This function creates a root coroutine of the coroutine subtree. It's the direct parent of coroutines launched inside the block and the indirect parent of any coroutines they launch. `coroutineScope()` executes the suspending block and waits until the block and any coroutines launched in it complete.

Here's an example:

```kotlin
// Imports the kotlin.time.Duration to express duration in seconds
import kotlin.time.Duration.Companion.seconds

import kotlinx.coroutines.*

// If the coroutine context doesn't specify a dispatcher,
// CoroutineScope.launch() uses Dispatchers.Default
suspend fun main() {
    // Root of the coroutine subtree
    coroutineScope { // this: CoroutineScope
        this.launch {
            this.launch {
                delay(2.seconds)
                println("Child of the enclosing coroutine completed")
            }
            println("Child coroutine 1 completed")
        }
        this.launch {
            delay(1.seconds)
            println("Child coroutine 2 completed")
        }
    }
    // Runs only after all children in the coroutineScope have completed
    println("Coroutine scope completed")
}
```

```
Child coroutine 1 completed
Child coroutine 2 completed
Child of the enclosing coroutine completed
Coroutine scope completed
```

Since no [dispatcher](https://kotlinlang.org/docs/coroutines-basics.html#coroutine-dispatchers) is specified in this example, the `CoroutineScope.launch()` builder functions in the `coroutineScope()` block inherit the current context. If that context doesn't have a specified dispatcher, `CoroutineScope.launch()` uses `Dispatchers.Default`, which runs on a shared pool of threads.

### Extract coroutine builders from the coroutine scope

In some cases, you may want to extract coroutine builder calls, such as [`CoroutineScope.launch()`](https://kotlinlang.org/docs/coroutines-basics.html#coroutinescope-launch), into separate functions.

Consider the following example:

```kotlin
suspend fun main() {
    coroutineScope { // this: CoroutineScope
        // Calls CoroutineScope.launch() where CoroutineScope is the receiver
        this.launch { println("1") }
        this.launch { println("2") }
    }
}
```

> You can also write `this.launch` without the explicit `this` expression, as `launch`. These examples use explicit `this` expressions to highlight that it's an extension function on `CoroutineScope`.  
> `this.launch` 표현에서 `this`를 명시하지 않고 단순히 `launch`로도 작성할 수 있습니다.
이 예제들에서는 `CoroutineScope`의 **확장 함수(extension function)** 임을 강조하기 위해 `this` 표현을 명시적으로 사용했습니다.
> 
> For more information on how lambdas with receivers work in Kotlin, see [Function literals with receiver](https://kotlinlang.org/docs/lambdas.html#function-literals-with-receiver).
> 

The `coroutineScope()` function takes a lambda with a `CoroutineScope` receiver. Inside this lambda, the implicit receiver is a `CoroutineScope`, so builder functions like `CoroutineScope.launch()` and [`CoroutineScope.async()`](https://kotlinlang.org/docs/coroutines-basics.html#coroutinescope-async) resolve as [extension functions](https://kotlinlang.org/docs/extensions.html#extension-functions) on that receiver.

To extract the coroutine builders into another function, that function must declare a `CoroutineScope` receiver, otherwise a compilation error occurs:

```kotlin
import kotlinx.coroutines.*
suspend fun main() {
    coroutineScope {
        launchAll()
    }
}

fun CoroutineScope.launchAll() { // this: CoroutineScope
    // Calls .launch() on CoroutineScope
    this.launch { println("1") }
    this.launch { println("2") } 
}
/* -- Calling launch without declaring CoroutineScope as the receiver results in a compilation error --

fun launchAll() {
    // Compilation error: this is not defined
    this.launch { println("1") }
    this.launch { println("2") }
}
 */
```

```
2
1
```

## Coroutine builder functions

A coroutine builder function is a function that accepts a `suspend` [lambda](https://kotlinlang.org/docs/lambdas.html) that defines a coroutine to run. Here are some examples:

- [`CoroutineScope.launch()`](https://kotlinlang.org/docs/coroutines-basics.html#coroutinescope-launch)
- [`CoroutineScope.async()`](https://kotlinlang.org/docs/coroutines-basics.html#coroutinescope-async)
- [`runBlocking()`](https://kotlinlang.org/docs/coroutines-basics.html#runblocking)
- [`withContext()`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-context.html)
- [`coroutineScope()`](https://kotlinlang.org/docs/coroutines-basics.html#create-a-coroutine-scope-with-the-coroutinescope-function)

Coroutine builder functions require a `CoroutineScope` to run in. This can be an existing scope or one you create with helper functions such as `coroutineScope()`, [`runBlocking()`](https://kotlinlang.org/docs/coroutines-basics.html#runblocking), or [`withContext()`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-context.html#). Each builder defines how the coroutine starts and how you interact with its result.

### `CoroutineScope.launch()`

The [`CoroutineScope.launch()`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/launch.html#) coroutine builder function is an extension function on `CoroutineScope`. It starts a new coroutine without blocking the rest of the scope, inside an existing [coroutine scope](https://kotlinlang.org/docs/coroutines-basics.html#coroutine-scope-and-structured-concurrency).

Use `CoroutineScope.launch()` to run a task alongside other work when the result isn't needed or you don't want to wait for it:

```kotlin
// Imports the kotlin.time.Duration to enable expressing duration in milliseconds
import kotlin.time.Duration.Companion.milliseconds

import kotlinx.coroutines.*

suspend fun main() {
    withContext(Dispatchers.Default) {
        performBackgroundWork()
    }
}

suspend fun performBackgroundWork() = coroutineScope { // this: CoroutineScope
    // Starts a coroutine that runs without blocking the scope
    this.launch {
        // Suspends to simulate background work
        delay(100.milliseconds)
        println("Sending notification in background")
    }

    // Main coroutine continues while a previous one suspends
    println("Scope continues")
}
```
```
Scope continues
Sending notification in background
```


After running this example, you can see that the `main()` function isn't blocked by `CoroutineScope.launch()` and keeps running other code while the coroutine works in the background.

> The `CoroutineScope.launch()` function returns a [`Job`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/) handle. Use this handle to wait for the launched coroutine to complete. For more information, see [Cancellation and timeouts](https://kotlinlang.org/docs/cancellation-and-timeouts.html#cancelling-coroutine-execution).

### `CoroutineScope.async()`

The [`CoroutineScope.async()`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/async.html) coroutine builder function is an extension function on `CoroutineScope`. It starts a concurrent computation inside an existing [coroutine scope](https://kotlinlang.org/docs/coroutines-basics.html#coroutine-scope-and-structured-concurrency) and returns a [`Deferred`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-deferred/) handle that represents an eventual result. Use the `.await()` function to suspend the code until the result is ready:

```kotlin
// Imports the kotlin.time.Duration to enable expressing duration in milliseconds
import kotlin.time.Duration.Companion.milliseconds

import kotlinx.coroutines.*

suspend fun main() = withContext(Dispatchers.Default) { // this: CoroutineScope
    // Starts downloading the first page
    val firstPage = this.async {
        delay(50.milliseconds)
        "First page"
    }

    // Starts downloading the second page in parallel
    val secondPage = this.async {
        delay(100.milliseconds)
        "Second page"
    }

    // Awaits both results and compares them
    val pagesAreEqual = firstPage.await() == secondPage.await()
    println("Pages are equal: $pagesAreEqual")
}
```
```
Pages are equal: false
```


### `runBlocking()`

The [`runBlocking()`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/run-blocking.html) coroutine builder function creates a coroutine scope and blocks the current [thread](https://kotlinlang.org/docs/coroutines-basics.html#comparing-coroutines-and-jvm-threads) until the coroutines launched in that scope finish.

Use `runBlocking()` only when there is no other option to call suspending code from non-suspending code:

```kotlin
import kotlin.time.Duration.Companion.milliseconds
import kotlinx.coroutines.*

// A third-party interface you can't change
interface Repository {
    fun readItem(): Int
}

object MyRepository : Repository {
    override fun readItem(): Int {
        // Bridges to a suspending function
        return runBlocking {
            myReadItem()
        }
    }
}

suspend fun myReadItem(): Int {
    delay(100.milliseconds)
    return 4
}
```

## Coroutine dispatchers

A [coroutine dispatcher](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-dispatchers/#) controls which thread or thread pool coroutines use for their execution. Coroutines aren't always tied to a single thread. They can pause on one thread and resume on another, depending on the dispatcher. This lets you run many coroutines at the same time without allocating a separate thread for every coroutine.

> Even though coroutines can suspend and resume on different threads, values written before the coroutine suspends are still guaranteed to be available within the same coroutine when it resumes.

A dispatcher works together with the [coroutine scope](https://kotlinlang.org/docs/coroutines-basics.html#coroutine-scope-and-structured-concurrency) to define when coroutines run and where they run. While the coroutine scope controls the coroutine's lifecycle, the dispatcher controls which threads are used for execution.

> You don't have to specify a dispatcher for every coroutine. By default, coroutines inherit the dispatcher from their parent scope. You can specify a dispatcher to run a coroutine in a different context.  
> 모든 coroutine마다 dispatcher를 명시할 필요는 없습니다. 기본적으로 coroutine은 부모 scope로부터 dispatcher를 상속받습니다. 필요한 경우 coroutine을 다른 context에서 실행하도록 별도의 dispatcher를 지정할 수도 있습니다.
> 
> If the coroutine context doesn't include a dispatcher, coroutine builders use `Dispatchers.Default`.

The `kotlinx.coroutines` library includes different dispatchers for different use cases. For example, [`Dispatchers.Default`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-dispatchers/-default.html) runs coroutines on a shared pool of threads, performing work in the background, separate from the main thread. This makes it an ideal choice for CPU-intensive operations like data processing.

To specify a dispatcher for a coroutine builder like `CoroutineScope.launch()`, pass it as an argument:

```kotlin
suspend fun runWithDispatcher() = coroutineScope { // this: CoroutineScope
    this.launch(Dispatchers.Default) {
        println("Running on ${Thread.currentThread().name}")
    }
}
```

Alternatively, you can use a `withContext()` block to run all code in it on a specified dispatcher:

```kotlin
// Imports the kotlin.time.Duration to enable expressing duration in milliseconds
import kotlin.time.Duration.Companion.milliseconds

import kotlinx.coroutines.*

suspend fun main() = withContext(Dispatchers.Default) { // this: CoroutineScope
    println("Running withContext block on ${Thread.currentThread().name}")

    val one = this.async {
        println("First calculation starting on ${Thread.currentThread().name}")
        val sum = (1L..500_000L).sum()
        delay(200L)
        println("First calculation done on ${Thread.currentThread().name}")
        sum
    }

    val two = this.async {
        println("Second calculation starting on ${Thread.currentThread().name}")
        val sum = (500_001L..1_000_000L).sum()
        println("Second calculation done on ${Thread.currentThread().name}")
        sum
    }

    // Waits for both calculations and prints the result
    println("Combined total: ${one.await() + two.await()}")
}
```
```
Running withContext block on DefaultDispatcher-worker-1
First calculation starting on DefaultDispatcher-worker-2 @coroutine#1
Second calculation starting on DefaultDispatcher-worker-1 @coroutine#2
Second calculation done on DefaultDispatcher-worker-1 @coroutine#2
First calculation done on DefaultDispatcher-worker-2 @coroutine#1
Combined total: 500000500000
```

To learn more about coroutine dispatchers and their uses, including other dispatchers like [`Dispatchers.IO`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-dispatchers/-i-o.html) and [`Dispatchers.Main`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-dispatchers/-main.html), see [Coroutine context and dispatchers](https://kotlinlang.org/docs/coroutine-context-and-dispatchers.html).

| Dispatcher               | 설명                                                               | 주요 사용 예시                                          |
| ------------------------ | ---------------------------------------------------------------- | ------------------------------------------------- |
| `Dispatchers.Default`    | 기본적인 Dispatcher로, CPU 연산에 최적화된 스레드 풀 사용 (CPU 코어 수 기반)            | 데이터 처리, 연산, 정렬, 파싱 등                              |
| `Dispatchers.IO`         | I/O에 최적화된 스레드 풀 (파일, 네트워크, DB 등)                                 | Retrofit, Room, 파일 읽기/쓰기                          |
| `Dispatchers.Main`       | UI 스레드 (Android의 메인 스레드)에서 실행                                    | UI 업데이트, View 조작                                  |
| `Dispatchers.Unconfined` | 스레드를 특정하지 않음. 처음엔 호출한 컨텍스트에서 시작, suspend 후 재개 시 다른 스레드로 옮겨질 수 있음 | 특수한 상황(테스트, low-level coroutine control)에서만 사용 권장 |


## Comparing coroutines and JVM threads

While coroutines are suspendable computations that run code concurrently like threads on the JVM, they work differently under the hood.

A thread is managed by the operating system. Threads can run tasks in parallel on multiple CPU cores and represent a standard approach to concurrency on the JVM. When you create a thread, the operating system allocates memory for its stack and uses the kernel to switch between threads. This makes threads powerful but also resource-intensive. Each thread usually needs a few megabytes of memory, and typically the JVM can only handle a few thousand threads at once.

On the other hand, a coroutine isn't bound to a specific thread. It can suspend on one thread and resume on another, so many coroutines can share the same thread pool. When a coroutine suspends, the thread isn't blocked and remains free to run other tasks. This makes coroutines much lighter than threads and allows running millions of them in one process without exhausting system resources.

반면 coroutine은 특정 thread에 묶여 있지 않습니다. 하나의 thread에서 일시 중단(suspend)되었다가 다른 thread에서 다시 실행(resume)될 수 있으므로, 여러 coroutine이 동일한 thread pool을 공유할 수 있습니다. coroutine이 일시 중단되면 thread는 차단되지 않으며, 다른 작업을 수행할 수 있는 상태로 남습니다. 이러한 특성 덕분에 coroutine은 thread보다 훨씬 가볍고, 시스템 리소스를 소모하지 않으면서 하나의 프로세스에서 수백만 개의 coroutine을 실행할 수 있습니다.


![Comparing coroutines and threads](https://kotlinlang.org/docs/images/coroutines-and-threads.svg "Comparing coroutines and threads")

Let's look at an example where 50,000 coroutines each wait five seconds and then print a period (`.`):

```kotlin
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.*

suspend fun main() {
    withContext(Dispatchers.Default) {
        // Launches 50,000 coroutines that each wait five seconds, then print a period
        printPeriods()
    }
}

suspend fun printPeriods() = coroutineScope { // this: CoroutineScope
    // Launches 50,000 coroutines that each wait five seconds, then print a period
    repeat(50_000) {
        this.launch {
            delay(5.seconds)
            print(".")
        }
    }
}
```


Now let's look at the same example using JVM threads:

```kotlin
import kotlin.concurrent.thread

fun main() {
    repeat(50_000) {
        thread {
            Thread.sleep(5000L)
            print(".")
        }
    }
}
```


Running this version uses much more memory because each thread needs its own memory stack. For 50,000 threads, that can be up to 100 GB, compared to roughly 500 MB for the same number of coroutines.

Depending on your operating system, JDK version, and settings, the JVM thread version may throw an out-of-memory error or slow down thread creation to avoid running too many threads at once.

## What's next

- Discover more about combining suspending functions in [Composing suspending functions](https://kotlinlang.org/docs/composing-suspending-functions.html).
- Dive deeper into coroutine execution and thread management in [Coroutine context and dispatchers](https://kotlinlang.org/docs/coroutine-context-and-dispatchers.html).

# Cancellation and timeouts

This section covers coroutine cancellation and timeouts.

## Cancelling coroutine execution

In a long-running application, you might need fine-grained control on your background coroutines. For example, a user might have closed the page that launched a coroutine, and now its result is no longer needed and its operation can be cancelled. The [launch](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/launch.html) function returns a [Job](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/index.html) that can be used to cancel the running coroutine:

```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking {
    val job = launch {
        repeat(1000) { i ->
            println("job: I'm sleeping $i ...")
            delay(500L)
        }
    }
    delay(1300L) // delay a bit
    println("main: I'm tired of waiting!")
    job.cancel() // cancels the job
    job.join() // waits for job's completion 
    println("main: Now I can quit.")    
}
```

It produces the following output:

```
job: I'm sleeping 0 ...
job: I'm sleeping 1 ...
job: I'm sleeping 2 ...
main: I'm tired of waiting!
main: Now I can quit.
```

As soon as main invokes `job.cancel`, we don't see any output from the other coroutine because it was cancelled. There is also a [Job](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/index.html) extension function [cancelAndJoin](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/cancel-and-join.html) that combines [cancel](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/cancel.html) and [join](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/join.html) invocations.

### `cancel()` 뒤에 `join()`을 하는 이유

- `cancel()`은 "그만해 주세요"라는 요청만 보낼 뿐 즉시 멈추게 하지는 않음
- `join()`은 실제로 그 코루틴이 완전히 종료될 때까지 기다림. 
- 순서 보장과 정리(cleanup)를 위해 둘을 같이 쓰는 경우가 많다.

자세히 풀어보면:

* `cancel()`은 호출한 쪽에서 즉시 반환됩니다. 내부적으로는 `CancellationException`을 통해 **코루틴에게 취소 신호**를 보냅니다(코루틴이 협력적으로 취소를 받아들여야 멈춥니다).
* 코루틴이 `delay`, `yield`, `suspend` 등을 하고 있으면 그곳에서 `CancellationException`을 던져 바로 빠져나오지만,

  * CPU 바운드 처럼 취소 체크를 안 하는 긴 루프를 돌고 있거나,
  * `try { ... } finally { ... }`에서 정리 작업을 하도록 되어 있는 경우(특히 `withContext(NonCancellable)`를 써서 정리를 보장하면)
    취소 이후에도 **정리 코드가 끝날 때까지** 시간이 걸립니다.
* `join()`은 그 코루틴이 **완전히 종료될 때까지(정리 코드 포함)** 호출한 쪽을 일시 중단(suspend)합니다. 그래서 `cancel()` + `join()` 은 "취소 요청을 보내고, 그 작업이 완전히 끝날 때까지 기다린다"는 의미입니다.

예시로 비교해 볼게요:

```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking {
    val job = launch {
        try {
            repeat(5) { i ->
                println("job: I'm sleeping $i ...")
                delay(500L)
            }
        } finally {
            println("job: finally start")
            withContext(NonCancellable) {
                delay(1000L) // 정리 작업 (예: 리소스 해제)
                println("job: cleanup done")
            }
        }
    }

    delay(800L)
    job.cancel()           // 취소 요청 (즉시 반환)
    println("after cancel")
    job.join()             // 정리까지 끝날 때까지 기다림
    println("after join")
}
```

예상 출력 (순서가 보장됨):

```
job: I'm sleeping 0 ...
job: I'm sleeping 1 ...
after cancel
job: finally start
job: cleanup done
after join
```

만약 `join()`을 호출하지 않으면:

```
job: I'm sleeping 0 ...
job: I'm sleeping 1 ...
after cancel
after join
job: finally start
job: cleanup done
```


## Cancellation is cooperative

Coroutine cancellation is cooperative. A coroutine code has to cooperate to be cancellable. All the suspending functions in `kotlinx.coroutines` are cancellable. They check for cancellation of coroutine and throw [CancellationException](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-cancellation-exception/index.html) when cancelled. However, if a coroutine is working in a computation and does not check for cancellation, then it cannot be cancelled, like the following example shows:  
코루틴의 취소는 협력적입니다. 코루틴 코드는 취소되려면 스스로 협력해야 합니다. `kotlinx.coroutines`에 있는 모든 suspend 함수들은 취소 가능하도록 작성되어 있어 코루틴의 취소를 검사하고 취소되면 CancellationException을 던집니다. 그러나 코루틴이 계산 작업을 수행하면서 취소 여부를 확인하지 않으면, 다음 예시가 보여주듯 취소할 수 없습니다.


```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking {
    val startTime = System.currentTimeMillis()
    val job = launch(Dispatchers.Default) {
        var nextPrintTime = startTime
        var i = 0
        while (i < 5) { // computation loop, just wastes CPU
            // print a message twice a second
            if (System.currentTimeMillis() >= nextPrintTime) {
                println("job: I'm sleeping ${i++} ...")
                nextPrintTime += 500L
            }
        }
    }
    delay(1300L) // delay a bit
    println("main: I'm tired of waiting!")
    job.cancelAndJoin() // cancels the job and waits for its completion
    println("main: Now I can quit.")    
}
```

```
job: I'm sleeping 0 ...
job: I'm sleeping 1 ...
job: I'm sleeping 2 ...
main: I'm tired of waiting!
job: I'm sleeping 3 ...
job: I'm sleeping 4 ...
main: Now I can quit.
```

Run it to see that it continues to print "I'm sleeping" even after cancellation until the job completes by itself after five iterations.

The same problem can be observed by catching a [CancellationException](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-cancellation-exception/index.html) and not rethrowing it:

```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking {
    val job = launch(Dispatchers.Default) {
        repeat(5) { i ->
            try {
                // print a message twice a second
                println("job: I'm sleeping $i ...")
                delay(500)
            } catch (e: Exception) {
                // log the exception
                println(e)
            }
        }
    }
    delay(1300L) // delay a bit
    println("main: I'm tired of waiting!")
    job.cancelAndJoin() // cancels the job and waits for its completion
    println("main: Now I can quit.")    
}
```

```
job: I'm sleeping 0 ...
job: I'm sleeping 1 ...
job: I'm sleeping 2 ...
main: I'm tired of waiting!
kotlinx.coroutines.JobCancellationException: StandaloneCoroutine was cancelled; job="coroutine#2":StandaloneCoroutine{Cancelling}@5496365b
job: I'm sleeping 3 ...
kotlinx.coroutines.JobCancellationException: StandaloneCoroutine was cancelled; job="coroutine#2":StandaloneCoroutine{Cancelling}@5496365b
job: I'm sleeping 4 ...
kotlinx.coroutines.JobCancellationException: StandaloneCoroutine was cancelled; job="coroutine#2":StandaloneCoroutine{Cancelling}@5496365b
main: Now I can quit.
```


While catching `Exception` is an anti-pattern, this issue may surface in more subtle ways, like when using the [`runCatching`](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin/run-catching.html) function, which does not rethrow [CancellationException](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-cancellation-exception/index.html).  
Exception을 잡는 것은 안티패턴이지만, 이 문제는 runCatching 함수를 사용할 때처럼 더 미묘한 방식으로 드러날 수 있습니다. runCatching은 CancellationException을 다시 던지지 않기 때문입니다.


## Making computation code cancellable

There are two approaches to making computation code cancellable. The first one is periodically invoking a suspending function that checks for cancellation. There are the [yield](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/yield.html) and [ensureActive](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/ensure-active.html) functions, which are great choices for that purpose. The other one is explicitly checking the cancellation status using [isActive](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/is-active.html). Let us try the latter approach.

Replace `while (i < 5)` in the previous example with `while (isActive)` and rerun it.

```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking {
    val startTime = System.currentTimeMillis()
    val job = launch(Dispatchers.Default) {
        var nextPrintTime = startTime
        var i = 0
        while (isActive) { // cancellable computation loop
            // prints a message twice a second
            if (System.currentTimeMillis() >= nextPrintTime) {
                println("job: I'm sleeping ${i++} ...")
                nextPrintTime += 500L
            }
        }
    }
    delay(1300L) // delay a bit
    println("main: I'm tired of waiting!")
    job.cancelAndJoin() // cancels the job and waits for its completion
    println("main: Now I can quit.")    
}
```

```
job: I'm sleeping 0 ...
job: I'm sleeping 1 ...
job: I'm sleeping 2 ...
main: I'm tired of waiting!
main: Now I can quit.
```

As you can see, now this loop is cancelled. [isActive](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/is-active.html) is an extension property available inside the coroutine via the [CoroutineScope](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-coroutine-scope/index.html) object.

## Closing resources with finally

Cancellable suspending functions throw [CancellationException](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-cancellation-exception/index.html) on cancellation, which can be handled in the usual way. For example, the `try {...} finally {...}` expression and Kotlin's [use](https://kotlinlang.org/api/core/kotlin-stdlib/kotlin.io/use.html) function execute their finalization actions normally when a coroutine is cancelled:

```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking {
    val job = launch {
        try {
            repeat(1000) { i ->
                println("job: I'm sleeping $i ...")
                delay(500L)
            }
        } finally {
            println("job: I'm running finally")
        }
    }
    delay(1300L) // delay a bit
    println("main: I'm tired of waiting!")
    job.cancelAndJoin() // cancels the job and waits for its completion
    println("main: Now I can quit.")    
}
```


Both [join](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/join.html) and [cancelAndJoin](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/cancel-and-join.html) wait for all finalization actions to complete, so the example above produces the following output:

```
job: I'm sleeping 0 ...
job: I'm sleeping 1 ...
job: I'm sleeping 2 ...
main: I'm tired of waiting!
job: I'm running finally
main: Now I can quit.
```

## Run non-cancellable block

Any attempt to use a suspending function in the `finally` block of the previous example causes [CancellationException](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-cancellation-exception/index.html), because the coroutine running this code is cancelled. Usually, this is not a problem, since all well-behaved closing operations (closing a file, cancelling a job, or closing any kind of communication channel) are usually non-blocking and do not involve any suspending functions. However, in the rare case when you need to suspend in a cancelled coroutine you can wrap the corresponding code in `withContext(NonCancellable) {...}` using [withContext](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-context.html) function and [NonCancellable](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-non-cancellable/index.html) context as the following example shows:  
이전 예제의 `finally` 블록에서 suspending function를 사용하려는 모든 시도는, 이 코드를 실행하는 코루틴이 취소되어 `CancellationException`을 발생시킵니다. 보통은 문제가 되지 않는데, 그 이유는 파일 닫기나 `job` 취소, 또는 어떤 종류의 통신 채널을 닫는 것 같은 잘 동작하는 종료 작업(well-behaved closing operations)들은 보통 비차단(non-blocking)이며 suspending function를 포함하지 않기 때문입니다. 그러나 취소된 코루틴에서 suspend가 정말로 필요할 때는 해당 코드를 `withContext(NonCancellable) { ... }`로 감싸서 `withContext` 함수와 `NonCancellable` 컨텍스트를 사용하면 됩니다. 다음 예제가 보여주듯이요:


```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking {
    val job = launch {
        try {
            repeat(1000) { i ->
                println("job: I'm sleeping $i ...")
                delay(500L)
            }
        } finally {
            withContext(NonCancellable) {
                println("job: I'm running finally")
                delay(1000L)
                println("job: And I've just delayed for 1 sec because I'm non-cancellable")
            }
        }
    }
    delay(1300L) // delay a bit
    println("main: I'm tired of waiting!")
    job.cancelAndJoin() // cancels the job and waits for its completion
    println("main: Now I can quit.")    
}
```

```
job: I'm sleeping 0 ...
job: I'm sleeping 1 ...
job: I'm sleeping 2 ...
main: I'm tired of waiting!
job: I'm running finally
job: And I've just delayed for 1 sec because I'm non-cancellable
main: Now I can quit.
```

## Timeout

The most obvious practical reason to cancel execution of a coroutine is because its execution time has exceeded some timeout. While you can manually track the reference to the corresponding [Job](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/index.html) and launch a separate coroutine to cancel the tracked one after delay, there is a ready to use [withTimeout](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-timeout.html) function that does it. Look at the following example:  
코루틴의 실행을 취소하는 가장 명백한 실용적 이유는 실행 시간이 어떤 타임아웃을 초과했기 때문입니다. 대응되는 `Job`에 대한 참조를 수동으로 추적하고, 별도의 코루틴을 `launch`해서 `delay` 후 추적된 코루틴을 취소하도록 할 수도 있지만, 이를 바로 처리해주는 `withTimeout` 함수가 이미 준비되어 있습니다. 다음 예제를 보세요:



```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking {
    withTimeout(1300L) {
        repeat(1000) { i ->
            println("I'm sleeping $i ...")
            delay(500L)
        }
    }
}
```


It produces the following output:

```
I'm sleeping 0 ...
I'm sleeping 1 ...
I'm sleeping 2 ...
Exception in thread "main" kotlinx.coroutines.TimeoutCancellationException: Timed out waiting for 1300 ms
```

The [TimeoutCancellationException](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-timeout-cancellation-exception/index.html) that is thrown by [withTimeout](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-timeout.html) is a subclass of [CancellationException](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-cancellation-exception/index.html). We have not seen its stack trace printed on the console before. That is because inside a cancelled coroutine `CancellationException` is considered to be a normal reason for coroutine completion. However, in this example we have used `withTimeout` right inside the `main` function.  
`withTimeout`에 의해 던져지는 `TimeoutCancellationException`은 `CancellationException`의 서브클래스입니다. 우리는 이전에 그 스택 트레이스가 콘솔에 출력되는 것을 본 적이 없습니다. 이는 취소된 코루틴 내부에서 `CancellationException`이 코루틴 완료의 정상적인 이유로 간주되기 때문입니다. 그러나 이 예제에서는 `withTimeout`을 `main` 함수 바로 내부에서 사용했습니다.


Since cancellation is just an exception, all resources are closed in the usual way. You can wrap the code with timeout in a `try {...} catch (e: TimeoutCancellationException) {...}` block if you need to do some additional action specifically on any kind of timeout or use the [withTimeoutOrNull](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-timeout-or-null.html) function that is similar to [withTimeout](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-timeout.html) but returns `null` on timeout instead of throwing an exception:  
취소는 단지 예외이므로 모든 리소스는 평소와 같은 방식으로 정리됩니다. 타임아웃이 발생했을 때 특별히 추가 작업을 해야 한다면, 타임아웃이 적용된 코드를 `try {...} catch (e: TimeoutCancellationException) {...}` 블록으로 감싸 처리할 수 있습니다. 또는 `withTimeout`과 유사하지만 타임아웃 시 예외를 던지는 대신 `null`을 반환하는 `withTimeoutOrNull` 함수를 사용할 수도 있습니다.


```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking {
    val result = withTimeoutOrNull(1300L) {
        repeat(1000) { i ->
            println("I'm sleeping $i ...")
            delay(500L)
        }
        "Done" // will get cancelled before it produces this result
    }
    println("Result is $result")
}
```

There is no longer an exception when running this code:

```
I'm sleeping 0 ...
I'm sleeping 1 ...
I'm sleeping 2 ...
Result is null
```

## Asynchronous timeout and resources

The timeout event in [withTimeout](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-timeout.html) is asynchronous with respect to the code running in its block and may happen at any time, even right before the return from inside of the timeout block. Keep this in mind if you open or acquire some resource inside the block that needs closing or release outside of the block.  
`withTimeout`에서의 타임아웃 이벤트는 해당 블록에서 실행되는 코드와 비동기적이며 언제든지 발생할 수 있습니다 — 심지어 타임아웃 블록 내부에서 값을 `return`하기 바로 직전에 일어날 수도 있습니다. 블록 내부에서 열거나 획득한 리소스를 블록 밖에서 닫거나 해제해야 하는 경우에는 이 점을 염두에 두세요.


For example, here we imitate a closeable resource with the `Resource` class that simply keeps track of how many times it was created by incrementing the `acquired` counter and decrementing the counter in its `close` function. Now let us create a lot of coroutines, each of which creates a `Resource` at the end of the `withTimeout` block and releases the resource outside the block. We add a small delay so that it is more likely that the timeout occurs right when the `withTimeout` block is already finished, which will cause a resource leak.  
예를 들어, 여기서는 `Resource` 클래스와 같이 닫을 수 있는 리소스를 흉내냅니다. 이 클래스는 생성될 때 `acquired` 카운터를 증가시켜 몇 번 생성되었는지를 기록하고, `close` 함수에서는 카운터를 감소시킵니다. 이제 많은 코루틴을 만들어 보겠습니다. 각 코루틴은 `withTimeout` 블록의 끝에서 `Resource`를 생성하고 블록 밖에서 그 리소스를 해제합니다. 타임아웃이 `withTimeout` 블록이 이미 끝나려는 바로 그 순간에 발생할 가능성을 높이기 위해 작은 지연을 추가하면, 그 결과로 자원 누수가 발생합니다.


```kotlin
import kotlinx.coroutines.*

var acquired = 0

class Resource {
    init { acquired++ } // Acquire the resource
    fun close() { acquired-- } // Release the resource
}

fun main() {
    runBlocking {
        repeat(10_000) { // Launch 10K coroutines
            launch { 
                val resource = withTimeout(60) { // Timeout of 60 ms
                    delay(50) // Delay for 50 ms
                    Resource() // Acquire a resource and return it from withTimeout block     
                }
                resource.close() // Release the resource
            }
        }
    }
    // Outside of runBlocking all coroutines have completed
    println(acquired) // Print the number of resources still acquired
}
```


If you run the above code, you'll see that it does not always print zero, though it may depend on the timings of your machine. You may need to tweak the timeout in this example to actually see non-zero values.

> Note that incrementing and decrementing `acquired` counter here from 10K coroutines is completely thread-safe, since it always happens from the same thread, the one used by `runBlocking`. More on that will be explained in the chapter on coroutine context.

To work around this problem you can store a reference to the resource in a variable instead of returning it from the `withTimeout` block.

```kotlin
import kotlinx.coroutines.*

var acquired = 0

class Resource {
    init { acquired++ } // Acquire the resource
    fun close() { acquired-- } // Release the resource
}

fun main() {
    runBlocking {
        repeat(10_000) { // Launch 10K coroutines
            launch { 
                var resource: Resource? = null // Not acquired yet
                try {
                    withTimeout(60) { // Timeout of 60 ms
                        delay(50) // Delay for 50 ms
                        resource = Resource() // Store a resource to the variable if acquired      
                    }
                    // We can do something else with the resource here
                } finally {  
                    resource?.close() // Release the resource if it was acquired
                }
            }
        }
    }
    // Outside of runBlocking all coroutines have completed
    println(acquired) // Print the number of resources still acquired
}
```

// Outside of runBlocking all coroutines have completed

println(acquired) // Print the number of resources still acquired

This example always prints zero. Resources do not leak.

# Asynchronous Flow

A suspending function asynchronously returns a single value, but how can we return multiple asynchronously computed values? This is where Kotlin Flows come in.

## Representing multiple values

Multiple values can be represented in Kotlin using [collections](https://kotlinlang.org/docs/reference/collections-overview.html). For example, we can have a `simple` function that returns a [List](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/-list/) of three numbers and then print them all using [forEach](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/for-each.html):

```kotlin
fun simple(): List<Int> = listOf(1, 2, 3)
 
fun main() {
    simple().forEach { value -> println(value) } 
}
```

This code outputs:

```
1
2
3
```

### Sequences

If we are computing the numbers with some CPU-consuming blocking code (each computation taking 100ms), then we can represent the numbers using a [Sequence](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.sequences/):

```kotlin
fun simple(): Sequence<Int> = sequence { // sequence builder
    for (i in 1..3) {
        Thread.sleep(100) // pretend we are computing it
        yield(i) // yield next value
    }
}

fun main() {
    simple().forEach { value -> println(value) } 
}
```


This code outputs the same numbers, but it waits 100ms before printing each one.

### Suspending functions

However, this computation blocks the main thread that is running the code. When these values are computed by asynchronous code we can mark the `simple` function with a `suspend` modifier, so that it can perform its work without blocking and return the result as a list:

```kotlin
import kotlinx.coroutines.*                 
                           
suspend fun simple(): List<Int> {
    delay(1000) // pretend we are doing something asynchronous here
    return listOf(1, 2, 3)
}

fun main() = runBlocking<Unit> {
    simple().forEach { value -> println(value) } 
}
```



This code prints the numbers after waiting for a second.

### Flows

Using the `List<Int>` result type, means we can only return all the values at once. To represent the stream of values that are being computed asynchronously, we can use a [`Flow<Int>`](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/-flow/index.html) type just like we would use a `Sequence<Int>` type for synchronously computed values:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

              
fun simple(): Flow<Int> = flow { // flow builder
    for (i in 1..3) {
        delay(100) // pretend we are doing something useful here
        emit(i) // emit next value
    }
}

fun main() = runBlocking<Unit> {
    // Launch a concurrent coroutine to check if the main thread is blocked
    launch {
        for (k in 1..3) {
            println("I'm not blocked $k")
            delay(100)
        }
    }
    // Collect the flow
    simple().collect { value -> println(value) } 
}
```

This code waits 100ms before printing each number without blocking the main thread. This is verified by printing "I'm not blocked" every 100ms from a separate coroutine that is running in the main thread:

```
I'm not blocked 1
1
I'm not blocked 2
2
I'm not blocked 3
3
```

Notice the following differences in the code with the [Flow](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/-flow/index.html) from the earlier examples:

- A builder function of [Flow](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/-flow/index.html) type is called [flow](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flow.html).
    
- Code inside a `flow { ... }` builder block can suspend.
    
- The `simple` function is no longer marked with a `suspend` modifier.
    
- Values are emitted from the flow using an [emit](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/-flow-collector/emit.html) function.
    
- Values are collected from the flow using a [collect](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/collect.html) function.
    

> We can replace [delay](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/delay.html) with `Thread.sleep` in the body of `simple`'s `flow { ... }` and see that the main thread is blocked in this case.

## Flows are cold

Flows are cold streams similar to sequences — the code inside a [flow](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flow.html) builder does not run until the flow is collected. This becomes clear in the following example:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

     
fun simple(): Flow<Int> = flow { 
    println("Flow started")
    for (i in 1..3) {
        delay(100)
        emit(i)
    }
}

fun main() = runBlocking<Unit> {
    println("Calling simple function...")
    val flow = simple()
    println("Calling collect...")
    flow.collect { value -> println(value) } 
    println("Calling collect again...")
    flow.collect { value -> println(value) } 
}
```



Which prints:

```
Calling simple function...
Calling collect...
Flow started
1
2
3
Calling collect again...
Flow started
1
2
3
```

This is a key reason the `simple` function (which returns a flow) is not marked with `suspend` modifier. The `simple()` call itself returns quickly and does not wait for anything. The flow starts afresh every time it is collected and that is why we see "Flow started" every time we call `collect` again.

## Flow cancellation basics

Flows adhere to the general cooperative cancellation of coroutines. As usual, flow collection can be cancelled when the flow is suspended in a cancellable suspending function (like [delay](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/delay.html)). The following example shows how the flow gets cancelled on a timeout when running in a [withTimeoutOrNull](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-timeout-or-null.html) block and stops executing its code:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

          
fun simple(): Flow<Int> = flow { 
    for (i in 1..3) {
        delay(100)          
        println("Emitting $i")
        emit(i)
    }
}

fun main() = runBlocking<Unit> {
    withTimeoutOrNull(250) { // Timeout after 250ms 
        simple().collect { value -> println(value) } 
    }
    println("Done")
}
```

Notice how only two numbers get emitted by the flow in the `simple` function, producing the following output:

```
Emitting 1
1
Emitting 2
2
Done
```

See [Flow cancellation checks](https://kotlinlang.org/docs/flow.html#flow-cancellation-checks) section for more details.

## Flow builders

The `flow { ... }` builder from the previous examples is the most basic one. There are other builders that allow flows to be declared:

- The [flowOf](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flow-of.html) builder defines a flow that emits a fixed set of values.
    
- Various collections and sequences can be converted to flows using the `.asFlow()` extension function.
    

For example, the snippet that prints the numbers 1 to 3 from a flow can be rewritten as follows:

// Convert an integer range to a flow

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

fun main() = runBlocking<Unit> {
    // Convert an integer range to a flow
    (1..3).asFlow().collect { value -> println(value) } 
}
```

## Intermediate flow operators

Flows can be transformed using operators, in the same way as you would transform collections and sequences. Intermediate operators are applied to an upstream flow and return a downstream flow. These operators are cold, just like flows are. A call to such an operator is not a suspending function itself. It works quickly, returning the definition of a new transformed flow.

The basic operators have familiar names like [map](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/map.html) and [filter](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/filter.html). An important difference of these operators from sequences is that blocks of code inside these operators can call suspending functions.

For example, a flow of incoming requests can be mapped to its results with a [map](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/map.html) operator, even when performing a request is a long-running operation that is implemented by a suspending function:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

          
suspend fun performRequest(request: Int): String {
    delay(1000) // imitate long-running asynchronous work
    return "response $request"
}

fun main() = runBlocking<Unit> {
    (1..3).asFlow() // a flow of requests
        .map { request -> performRequest(request) }
        .collect { response -> println(response) }
}
```

It produces the following three lines, each appearing one second after the previous:

```
response 1
response 2
response 3
```

### Transform operator

Among the flow transformation operators, the most general one is called [transform](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/transform.html). It can be used to imitate simple transformations like [map](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/map.html) and [filter](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/filter.html), as well as implement more complex transformations. Using the `transform` operator, we can [emit](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/-flow-collector/emit.html) arbitrary values an arbitrary number of times.

For example, using `transform` we can emit a string before performing a long-running asynchronous request and follow it with a response:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

suspend fun performRequest(request: Int): String {
    delay(1000) // imitate long-running asynchronous work
    return "response $request"
}

fun main() = runBlocking<Unit> {
    (1..3).asFlow() // a flow of requests
        .transform { request ->
            emit("Making request $request") 
            emit(performRequest(request)) 
        }
        .collect { response -> println(response) }
}
```

The output of this code is:

```
Making request 1
response 1
Making request 2
response 2
Making request 3
response 3
```

### Size-limiting operators

Size-limiting intermediate operators like [take](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/take.html) cancel the execution of the flow when the corresponding limit is reached. Cancellation in coroutines is always performed by throwing an exception, so that all the resource-management functions (like `try { ... } finally { ... }` blocks) operate normally in case of cancellation:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

fun numbers(): Flow<Int> = flow {
    try {                          
        emit(1)
        emit(2) 
        println("This line will not execute")
        emit(3)    
    } finally {
        println("Finally in numbers")
    }
}

fun main() = runBlocking<Unit> {
    numbers() 
        .take(2) // take only the first two
        .collect { value -> println(value) }
}            
```

The output of this code clearly shows that the execution of the `flow { ... }` body in the `numbers()` function stopped after emitting the second number:

```
1
2
Finally in numbers
```

## Terminal flow operators

Terminal operators on flows are suspending functions that start a collection of the flow. The [collect](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/collect.html) operator is the most basic one, but there are other terminal operators, which can make it easier:
- Conversion to various collections like [toList](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/to-list.html) and [toSet](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/to-set.html).
- Operators to get the [first](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/first.html) value and to ensure that a flow emits a [single](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/single.html) value.
- Reducing a flow to a value with [reduce](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/reduce.html) and [fold](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/fold.html).
    

For example:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

fun main() = runBlocking<Unit> {

    val sum = (1..5).asFlow()
        .map { it * it } // squares of numbers from 1 to 5                           
        .reduce { a, b -> a + b } // sum them (terminal operator)
    println(sum)     
}
```

Prints a single number:

```
55
```

## Flows are sequential

Each individual collection of a flow is performed sequentially unless special operators that operate on multiple flows are used. The collection works directly in the coroutine that calls a terminal operator. No new coroutines are launched by default. Each emitted value is processed by all the intermediate operators from upstream to downstream and is then delivered to the terminal operator after.

See the following example that filters the even integers and maps them to strings:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

fun main() = runBlocking<Unit> {

    (1..5).asFlow()
        .filter {
            println("Filter $it")
            it % 2 == 0              
        }              
        .map { 
            println("Map $it")
            "string $it"
        }.collect { 
            println("Collect $it")
        }                      
}
```

Producing:

```
Filter 1
Filter 2
Map 2
Collect string 2
Filter 3
Filter 4
Map 4
Collect string 4
Filter 5
```

## Flow context

Collection of a flow always happens in the context of the calling coroutine. For example, if there is a `simple` flow, then the following code runs in the context specified by the author of this code, regardless of the implementation details of the `simple` flow:

```kotlin
withContext(context) {
    simple().collect { value ->
        println(value) // run in the specified context
    }
}
```

This property of a flow is called context preservation.

So, by default, code in the `flow { ... }` builder runs in the context that is provided by a collector of the corresponding flow. For example, consider the implementation of a `simple` function that prints the thread it is called on and emits three numbers:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

fun log(msg: String) = println("[${Thread.currentThread().name}] $msg")
           
fun simple(): Flow<Int> = flow {
    log("Started simple flow")
    for (i in 1..3) {
        emit(i)
    }
}  

fun main() = runBlocking<Unit> {
    simple().collect { value -> log("Collected $value") } 
}            
```

Running this code produces:

```
[main @coroutine#1] Started simple flow
[main @coroutine#1] Collected 1
[main @coroutine#1] Collected 2
[main @coroutine#1] Collected 3
```

Since `simple().collect` is called from the main thread, the body of `simple`'s flow is also called in the main thread. This is the perfect default for fast-running or asynchronous code that does not care about the execution context and does not block the caller.

### A common pitfall when using withContext

However, the long-running CPU-consuming code might need to be executed in the context of [Dispatchers.Default](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-dispatchers/-default.html) and UI-updating code might need to be executed in the context of [Dispatchers.Main](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-dispatchers/-main.html). Usually, [withContext](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-context.html) is used to change the context in the code using Kotlin coroutines, but code in the `flow { ... }` builder has to honor the context preservation property and is not allowed to [emit](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/-flow-collector/emit.html) from a different context.

Try running the following code:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
                      
fun simple(): Flow<Int> = flow {
    // The WRONG way to change context for CPU-consuming code in flow builder
    kotlinx.coroutines.withContext(Dispatchers.Default) {
        for (i in 1..3) {
            Thread.sleep(100) // pretend we are computing it in CPU-consuming way
            emit(i) // emit next value
        }
    }
}

fun main() = runBlocking<Unit> {
    simple().collect { value -> println(value) } 
}            
```

This code produces the following exception:

```
Exception in thread "main" java.lang.IllegalStateException: Flow invariant is violated:
		Flow was collected in [CoroutineId(1), "coroutine#1":BlockingCoroutine{Active}@5511c7f8, BlockingEventLoop@2eac3323],
		but emission happened in [CoroutineId(1), "coroutine#1":DispatchedCoroutine{Active}@2dae0000, Dispatchers.Default].
		Please refer to 'flow' documentation or use 'flowOn' instead
	at ...
```

### flowOn operator

The exception refers to the [flowOn](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flow-on.html) function that shall be used to change the context of the flow emission. The correct way to change the context of a flow is shown in the example below, which also prints the names of the corresponding threads to show how it all works:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

fun log(msg: String) = println("[${Thread.currentThread().name}] $msg")
           
fun simple(): Flow<Int> = flow {
    for (i in 1..3) {
        Thread.sleep(100) // pretend we are computing it in CPU-consuming way
        log("Emitting $i")
        emit(i) // emit next value
    }
}.flowOn(Dispatchers.Default) // RIGHT way to change context for CPU-consuming code in flow builder

fun main() = runBlocking<Unit> {
    simple().collect { value ->
        log("Collected $value") 
    } 
}
```

Notice how `flow { ... }` works in the background thread, while collection happens in the main thread:

```
[DefaultDispatcher-worker-1 @coroutine#2] Emitting 1
[main @coroutine#1] Collected 1
[DefaultDispatcher-worker-1 @coroutine#2] Emitting 2
[main @coroutine#1] Collected 2
[DefaultDispatcher-worker-1 @coroutine#2] Emitting 3
[main @coroutine#1] Collected 3
```

Another thing to observe here is that the [flowOn](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flow-on.html) operator has changed the default sequential nature of the flow. Now collection happens in one coroutine ("coroutine#1") and emission happens in another coroutine ("coroutine#2") that is running in another thread concurrently with the collecting coroutine. The [flowOn](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flow-on.html) operator creates another coroutine for an upstream flow when it has to change the [CoroutineDispatcher](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-coroutine-dispatcher/index.html) in its context.

## Buffering

Running different parts of a flow in different coroutines can be helpful from the standpoint of the overall time it takes to collect the flow, especially when long-running asynchronous operations are involved. For example, consider a case when the emission by a `simple` flow is slow, taking 100 ms to produce an element; and collector is also slow, taking 300 ms to process an element. Let's see how long it takes to collect such a flow with three numbers:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlin.system.*

fun simple(): Flow<Int> = flow {
    for (i in 1..3) {
        delay(100) // pretend we are asynchronously waiting 100 ms
        emit(i) // emit next value
    }
}

fun main() = runBlocking<Unit> { 
    val time = measureTimeMillis {
        simple().collect { value -> 
            delay(300) // pretend we are processing it for 300 ms
            println(value) 
        } 
    }   
    println("Collected in $time ms")
}
```

It produces something like this, with the whole collection taking around 1200 ms (three numbers, 400 ms for each):

```
1
2
3
Collected in 1220 ms
```

We can use a [buffer](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/buffer.html) operator on a flow to run emitting code of the `simple` flow concurrently with collecting code, as opposed to running them sequentially:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlin.system.*

fun simple(): Flow<Int> = flow {
    for (i in 1..3) {
        delay(100) // pretend we are asynchronously waiting 100 ms
        emit(i) // emit next value
    }
}

fun main() = runBlocking<Unit> { 
    val time = measureTimeMillis {
        simple()
            .buffer() // buffer emissions, don't wait
            .collect { value -> 
                delay(300) // pretend we are processing it for 300 ms
                println(value) 
            } 
    }   
    println("Collected in $time ms")
}
```

It produces the same numbers just faster, as we have effectively created a processing pipeline, having to only wait 100 ms for the first number and then spending only 300 ms to process each number. This way it takes around 1000 ms to run:

```
1
2
3
Collected in 1071 ms
```

> Note that the [flowOn](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flow-on.html) operator uses the same buffering mechanism when it has to change a [CoroutineDispatcher](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-coroutine-dispatcher/index.html), but here we explicitly request buffering without changing the execution context.

### Conflation

When a flow represents partial results of the operation or operation status updates, it may not be necessary to process each value, but instead, only most recent ones. In this case, the [conflate](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/conflate.html) operator can be used to skip intermediate values when a collector is too slow to process them. Building on the previous example:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlin.system.*

fun simple(): Flow<Int> = flow {
    for (i in 1..3) {
        delay(100) // pretend we are asynchronously waiting 100 ms
        emit(i) // emit next value
    }
}

fun main() = runBlocking<Unit> { 
    val time = measureTimeMillis {
        simple()
            .conflate() // conflate emissions, don't process each one
            .collect { value -> 
                delay(300) // pretend we are processing it for 300 ms
                println(value) 
            } 
    }   
    println("Collected in $time ms")
}
```

We see that while the first number was still being processed the second, and third were already produced, so the second one was conflated and only the most recent (the third one) was delivered to the collector:

```
1
3
Collected in 758 ms
```

### Processing the latest value

Conflation is one way to speed up processing when both the emitter and collector are slow. It does it by dropping emitted values. The other way is to cancel a slow collector and restart it every time a new value is emitted. There is a family of `xxxLatest` operators that perform the same essential logic of a `xxx` operator, but cancel the code in their block on a new value. Let's try changing [conflate](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/conflate.html) to [collectLatest](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/collect-latest.html) in the previous example:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlin.system.*

fun simple(): Flow<Int> = flow {
    for (i in 1..3) {
        delay(100) // pretend we are asynchronously waiting 100 ms
        emit(i) // emit next value
    }
}

fun main() = runBlocking<Unit> { 
    val time = measureTimeMillis {
        simple()
            .collectLatest { value -> // cancel & restart on the latest value
                println("Collecting $value") 
                delay(300) // pretend we are processing it for 300 ms
                println("Done $value") 
            } 
    }   
    println("Collected in $time ms")
}
```

Since the body of [collectLatest](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/collect-latest.html) takes 300 ms, but new values are emitted every 100 ms, we see that the block is run on every value, but completes only for the last value:

```
Collecting 1
Collecting 2
Collecting 3
Done 3
Collected in 741 ms
```

## Composing multiple flows

There are lots of ways to compose multiple flows.

### Zip

Just like the [Sequence.zip](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.sequences/zip.html) extension function in the Kotlin standard library, flows have a [zip](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/zip.html) operator that combines the corresponding values of two flows:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

fun main() = runBlocking<Unit> { 

    val nums = (1..3).asFlow() // numbers 1..3
    val strs = flowOf("one", "two", "three") // strings 
    nums.zip(strs) { a, b -> "$a -> $b" } // compose a single string
        .collect { println(it) } // collect and print
}
```

This example prints:

```
1 -> one
2 -> two
3 -> three
```

### Combine

When flow represents the most recent value of a variable or operation (see also the related section on [conflation](https://kotlinlang.org/docs/flow.html#conflation)), it might be needed to perform a computation that depends on the most recent values of the corresponding flows and to recompute it whenever any of the upstream flows emit a value. The corresponding family of operators is called [combine](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/combine.html).  
flow가 변수나 연산의 가장 최근 값을 나타낼 때(관련 내용은 conflation 섹션도 참고), 해당 흐름들에 대응하는 최신 값들에 의존하는 계산을 수행해야 하거나 상류의 어느 flow가 값을 방출할 때마다 그 계산을 다시 수행해야 할 수 있습니다. 이러한 연산자 계열을 combine이라고 합니다.



For example, if the numbers in the previous example update every 300ms, but strings update every 400 ms, then zipping them using the [zip](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/zip.html) operator will still produce the same result, albeit results that are printed every 400 ms:

> We use a [onEach](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/on-each.html) intermediate operator in this example to delay each element and make the code that emits sample flows more declarative and shorter.

- `onEach` : `Flow<T>`의 intermediate operator로, 흐름의 각 요소를 받았을 때 부수효과(side-effect) 를 실행하고 원래의 요소를 그대로 다음으로 흘려보낸다. 람다에는 suspend를 쓸 수 있어서 delay, I/O, 로그 등 비동기 작업을 안전하게 할 수 있다.
​

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

fun main() = runBlocking<Unit> { 

    val nums = (1..3).asFlow().onEach { delay(300) } // numbers 1..3 every 300 ms
    val strs = flowOf("one", "two", "three").onEach { delay(400) } // strings every 400 ms
    val startTime = System.currentTimeMillis() // remember the start time 
    nums.zip(strs) { a, b -> "$a -> $b" } // compose a single string with "zip"
        .collect { value -> // collect and print 
            println("$value at ${System.currentTimeMillis() - startTime} ms from start") 
        } 
}
```

```
1 -> one at 445 ms from start
2 -> two at 845 ms from start
3 -> three at 1248 ms from start
```

However, when using a [combine](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/combine.html) operator here instead of a [zip](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/zip.html):

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

fun main() = runBlocking<Unit> { 

    val nums = (1..3).asFlow().onEach { delay(300) } // numbers 1..3 every 300 ms
    val strs = flowOf("one", "two", "three").onEach { delay(400) } // strings every 400 ms          
    val startTime = System.currentTimeMillis() // remember the start time 
    nums.combine(strs) { a, b -> "$a -> $b" } // compose a single string with "combine"
        .collect { value -> // collect and print 
            println("$value at ${System.currentTimeMillis() - startTime} ms from start") 
        } 
}
```

We get quite a different output, where a line is printed at each emission from either `nums` or `strs` flows:

```
1 -> one at 452 ms from start
2 -> one at 651 ms from start
2 -> two at 854 ms from start
3 -> two at 952 ms from start
3 -> three at 1256 ms from start
```

## Flattening flows

Flows represent asynchronously received sequences of values, and so it is quite easy to get into a situation where each value triggers a request for another sequence of values. For example, we can have the following function that returns a flow of two strings 500 ms apart:

```
fun requestFlow(i: Int): Flow<String> = flow {
    emit("$i: First")
    delay(500) // wait 500 ms
    emit("$i: Second")
}
```

Now if we have a flow of three integers and call `requestFlow` on each of them like this:

```
(1..3).asFlow().map { requestFlow(it) }
```

Then we will end up with a flow of flows (`Flow<Flow<String>>`) that needs to be flattened into a single flow for further processing. Collections and sequences have [flatten](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.sequences/flatten.html) and [flatMap](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.sequences/flat-map.html) operators for this. However, due to the asynchronous nature of flows they call for different modes of flattening, and hence, a family of flattening operators on flows exists.

### flatMapConcat

Concatenation of flows of flows is provided by the [flatMapConcat](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flat-map-concat.html) and [flattenConcat](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flatten-concat.html) operators. They are the most direct analogues of the corresponding sequence operators. They wait for the inner flow to complete before starting to collect the next one as the following example shows:

val startTime = System.currentTimeMillis() // remember the start time 

(1..3).asFlow().onEach { delay(100) } // emit a number every 100 ms 

    .flatMapConcat { requestFlow(it) }                                                                           

    .collect { value -> // collect and print 

        println("$value at ${System.currentTimeMillis() - startTime} ms from start") 

    } 






The sequential nature of [flatMapConcat](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flat-map-concat.html) is clearly seen in the output:

```
1: First at 121 ms from start
1: Second at 622 ms from start
2: First at 727 ms from start
2: Second at 1227 ms from start
3: First at 1328 ms from start
3: Second at 1829 ms from start
```

### flatMapMerge

Another flattening operation is to concurrently collect all the incoming flows and merge their values into a single flow so that values are emitted as soon as possible. It is implemented by [flatMapMerge](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flat-map-merge.html) and [flattenMerge](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flatten-merge.html) operators. They both accept an optional `concurrency` parameter that limits the number of concurrent flows that are collected at the same time (it is equal to [DEFAULT_CONCURRENCY](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/-d-e-f-a-u-l-t_-c-o-n-c-u-r-r-e-n-c-y.html) by default).

val startTime = System.currentTimeMillis() // remember the start time 

(1..3).asFlow().onEach { delay(100) } // a number every 100 ms 

    .flatMapMerge { requestFlow(it) }                                                                           

    .collect { value -> // collect and print 

        println("$value at ${System.currentTimeMillis() - startTime} ms from start") 

    } 






The concurrent nature of [flatMapMerge](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flat-map-merge.html) is obvious:

```
1: First at 136 ms from start
2: First at 231 ms from start
3: First at 333 ms from start
1: Second at 639 ms from start
2: Second at 732 ms from start
3: Second at 833 ms from start
```

> Note that the [flatMapMerge](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flat-map-merge.html) calls its block of code (`{ requestFlow(it) }` in this example) sequentially, but collects the resulting flows concurrently, it is the equivalent of performing a sequential `map { requestFlow(it) }` first and then calling [flattenMerge](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flatten-merge.html) on the result.

### flatMapLatest

In a similar way to the [collectLatest](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/collect-latest.html) operator, that was described in the section ["Processing the latest value"](https://kotlinlang.org/docs/flow.html#processing-the-latest-value), there is the corresponding "Latest" flattening mode where the collection of the previous flow is cancelled as soon as new flow is emitted. It is implemented by the [flatMapLatest](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flat-map-latest.html) operator.

val startTime = System.currentTimeMillis() // remember the start time 

(1..3).asFlow().onEach { delay(100) } // a number every 100 ms 

    .flatMapLatest { requestFlow(it) }                                                                           

    .collect { value -> // collect and print 

        println("$value at ${System.currentTimeMillis() - startTime} ms from start") 

    } 






The output here in this example is a good demonstration of how [flatMapLatest](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flat-map-latest.html) works:

```
1: First at 142 ms from start
2: First at 322 ms from start
3: First at 425 ms from start
3: Second at 931 ms from start
```

> Note that [flatMapLatest](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flat-map-latest.html) cancels all the code in its block (`{ requestFlow(it) }` in this example) when a new value is received. It makes no difference in this particular example, because the call to `requestFlow` itself is fast, not-suspending, and cannot be cancelled. However, a differnce in output would be visible if we were to use suspending functions like `delay` in `requestFlow`.

## Flow exceptions

Flow collection can complete with an exception when an emitter or code inside the operators throw an exception. There are several ways to handle these exceptions.

### Collector try and catch

A collector can use Kotlin's [`try/catch`](https://kotlinlang.org/docs/reference/exceptions.html) block to handle exceptions:

fun simple(): Flow<Int> = flow {

    for (i in 1..3) {

        println("Emitting $i")

        emit(i) // emit next value

    }

}

​

fun main() = runBlocking<Unit> {

    try {

        simple().collect { value ->         

            println(value)

            check(value <= 1) { "Collected $value" }

        }

    } catch (e: Throwable) {

        println("Caught $e")

    } 

}            






This code successfully catches an exception in [collect](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/collect.html) terminal operator and, as we see, no more values are emitted after that:

```
Emitting 1
1
Emitting 2
2
Caught java.lang.IllegalStateException: Collected 2
```

### Everything is caught

The previous example actually catches any exception happening in the emitter or in any intermediate or terminal operators. For example, let's change the code so that emitted values are [mapped](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/map.html) to strings, but the corresponding code produces an exception:

fun simple(): Flow<String> = 

    flow {

        for (i in 1..3) {

            println("Emitting $i")

            emit(i) // emit next value

        }

    }

    .map { value ->

        check(value <= 1) { "Crashed on $value" }                 

        "string $value"

    }

​

fun main() = runBlocking<Unit> {

    try {

        simple().collect { value -> println(value) }

    } catch (e: Throwable) {

        println("Caught $e")

    } 

}            






This exception is still caught and collection is stopped:

```
Emitting 1
string 1
Emitting 2
Caught java.lang.IllegalStateException: Crashed on 2
```

## Exception transparency

But how can code of the emitter encapsulate its exception handling behavior?

Flows must be transparent to exceptions and it is a violation of the exception transparency to [emit](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/-flow-collector/emit.html) values in the `flow { ... }` builder from inside of a `try/catch` block. This guarantees that a collector throwing an exception can always catch it using `try/catch` as in the previous example.

The emitter can use a [catch](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/catch.html) operator that preserves this exception transparency and allows encapsulation of its exception handling. The body of the `catch` operator can analyze an exception and react to it in different ways depending on which exception was caught:

- Exceptions can be rethrown using `throw`.
    
- Exceptions can be turned into emission of values using [emit](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/-flow-collector/emit.html) from the body of [catch](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/catch.html).
    
- Exceptions can be ignored, logged, or processed by some other code.
    

For example, let us emit the text on catching an exception:

simple()

    .catch { e -> emit("Caught $e") } // emit on exception

    .collect { value -> println(value) }






The output of the example is the same, even though we do not have `try/catch` around the code anymore.

### Transparent catch

The [catch](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/catch.html) intermediate operator, honoring exception transparency, catches only upstream exceptions (that is an exception from all the operators above `catch`, but not below it). If the block in `collect { ... }` (placed below `catch`) throws an exception then it escapes:

fun simple(): Flow<Int> = flow {

    for (i in 1..3) {

        println("Emitting $i")

        emit(i)

    }

}

​

fun main() = runBlocking<Unit> {

    simple()

        .catch { e -> println("Caught $e") } // does not catch downstream exceptions

        .collect { value ->

            check(value <= 1) { "Collected $value" }                 

            println(value) 

        }

}            






A "Caught ..." message is not printed despite there being a `catch` operator:

```
Emitting 1
1
Emitting 2
Exception in thread "main" java.lang.IllegalStateException: Collected 2
	at ...
```

### Catching declaratively

We can combine the declarative nature of the [catch](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/catch.html) operator with a desire to handle all the exceptions, by moving the body of the [collect](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/collect.html) operator into [onEach](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/on-each.html) and putting it before the `catch` operator. Collection of this flow must be triggered by a call to `collect()` without parameters:

simple()

    .onEach { value ->

        check(value <= 1) { "Collected $value" }                 

        println(value) 

    }

    .catch { e -> println("Caught $e") }

    .collect()






Now we can see that a "Caught ..." message is printed and so we can catch all the exceptions without explicitly using a `try/catch` block:

```
Emitting 1
1
Emitting 2
Caught java.lang.IllegalStateException: Collected 2
```

## Flow completion

When flow collection completes (normally or exceptionally) it may need to execute an action. As you may have already noticed, it can be done in two ways: imperative or declarative.

### Imperative finally block

In addition to `try`/`catch`, a collector can also use a `finally` block to execute an action upon `collect` completion.

fun simple(): Flow<Int> = (1..3).asFlow()

​

fun main() = runBlocking<Unit> {

    try {

        simple().collect { value -> println(value) }

    } finally {

        println("Done")

    }

}            






This code prints three numbers produced by the `simple` flow followed by a "Done" string:

```
1
2
3
Done
```

### Declarative handling

For the declarative approach, flow has [onCompletion](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/on-completion.html) intermediate operator that is invoked when the flow has completely collected.

The previous example can be rewritten using an [onCompletion](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/on-completion.html) operator and produces the same output:

simple()

    .onCompletion { println("Done") }

    .collect { value -> println(value) }






The key advantage of [onCompletion](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/on-completion.html) is a nullable `Throwable` parameter of the lambda that can be used to determine whether the flow collection was completed normally or exceptionally. In the following example the `simple` flow throws an exception after emitting the number 1:

fun simple(): Flow<Int> = flow {

    emit(1)

    throw RuntimeException()

}

​

fun main() = runBlocking<Unit> {

    simple()

        .onCompletion { cause -> if (cause != null) println("Flow completed exceptionally") }

        .catch { cause -> println("Caught exception") }

        .collect { value -> println(value) }

}            






As you may expect, it prints:

```
1
Flow completed exceptionally
Caught exception
```

The [onCompletion](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/on-completion.html) operator, unlike [catch](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/catch.html), does not handle the exception. As we can see from the above example code, the exception still flows downstream. It will be delivered to further `onCompletion` operators and can be handled with a `catch` operator.

### Successful completion

Another difference with [catch](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/catch.html) operator is that [onCompletion](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/on-completion.html) sees all exceptions and receives a `null` exception only on successful completion of the upstream flow (without cancellation or failure).

fun simple(): Flow<Int> = (1..3).asFlow()

​

fun main() = runBlocking<Unit> {

    simple()

        .onCompletion { cause -> println("Flow completed with $cause") }

        .collect { value ->

            check(value <= 1) { "Collected $value" }                 

            println(value) 

        }

}






We can see the completion cause is not null, because the flow was aborted due to downstream exception:

```
1
Flow completed with java.lang.IllegalStateException: Collected 2
Exception in thread "main" java.lang.IllegalStateException: Collected 2
```

## Imperative versus declarative

Now we know how to collect flow, and handle its completion and exceptions in both imperative and declarative ways. The natural question here is, which approach is preferred and why? As a library, we do not advocate for any particular approach and believe that both options are valid and should be selected according to your own preferences and code style.

## Launching flow

It is easy to use flows to represent asynchronous events that are coming from some source. In this case, we need an analogue of the `addEventListener` function that registers a piece of code with a reaction for incoming events and continues further work. The [onEach](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/on-each.html) operator can serve this role. However, `onEach` is an intermediate operator. We also need a terminal operator to collect the flow. Otherwise, just calling `onEach` has no effect.

If we use the [collect](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/collect.html) terminal operator after `onEach`, then the code after it will wait until the flow is collected:

// Imitate a flow of events

fun events(): Flow<Int> = (1..3).asFlow().onEach { delay(100) }

​

fun main() = runBlocking<Unit> {

    events()

        .onEach { event -> println("Event: $event") }

        .collect() // <--- Collecting the flow waits

    println("Done")

}            






As you can see, it prints:

```
Event: 1
Event: 2
Event: 3
Done
```

The [launchIn](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/launch-in.html) terminal operator comes in handy here. By replacing `collect` with `launchIn` we can launch a collection of the flow in a separate coroutine, so that execution of further code immediately continues:

fun main() = runBlocking<Unit> {

    events()

        .onEach { event -> println("Event: $event") }

        .launchIn(this) // <--- Launching the flow in a separate coroutine

    println("Done")

}            






It prints:

```
Done
Event: 1
Event: 2
Event: 3
```

The required parameter to `launchIn` must specify a [CoroutineScope](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-coroutine-scope/index.html) in which the coroutine to collect the flow is launched. In the above example this scope comes from the [runBlocking](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/run-blocking.html) coroutine builder, so while the flow is running, this [runBlocking](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/run-blocking.html) scope waits for completion of its child coroutine and keeps the main function from returning and terminating this example.

In actual applications a scope will come from an entity with a limited lifetime. As soon as the lifetime of this entity is terminated the corresponding scope is cancelled, cancelling the collection of the corresponding flow. This way the pair of `onEach { ... }.launchIn(scope)` works like the `addEventListener`. However, there is no need for the corresponding `removeEventListener` function, as cancellation and structured concurrency serve this purpose.

Note that [launchIn](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/launch-in.html) also returns a [Job](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/index.html), which can be used to [cancel](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/cancel.html) the corresponding flow collection coroutine only without cancelling the whole scope or to [join](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/join.html) it.

### Flow cancellation checks

For convenience, the [flow](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/flow.html) builder performs additional [ensureActive](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/ensure-active.html) checks for cancellation on each emitted value. It means that a busy loop emitting from a `flow { ... }` is cancellable:

fun foo(): Flow<Int> = flow { 

    for (i in 1..5) {

        println("Emitting $i") 

        emit(i) 

    }

}

​

fun main() = runBlocking<Unit> {

    foo().collect { value -> 

        if (value == 3) cancel()  

        println(value)

    } 

}






We get only numbers up to 3 and a [CancellationException](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-cancellation-exception/index.html) after trying to emit number 4:

```
Emitting 1
1
Emitting 2
2
Emitting 3
3
Emitting 4
Exception in thread "main" kotlinx.coroutines.JobCancellationException: BlockingCoroutine was cancelled; job="coroutine#1":BlockingCoroutine{Cancelled}@6d7b4f4c
```

However, most other flow operators do not do additional cancellation checks on their own for performance reasons. For example, if you use [IntRange.asFlow](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/as-flow.html) extension to write the same busy loop and don't suspend anywhere, then there are no checks for cancellation:

fun main() = runBlocking<Unit> {

    (1..5).asFlow().collect { value -> 

        if (value == 3) cancel()  

        println(value)

    } 

}






All numbers from 1 to 5 are collected and cancellation gets detected only before return from `runBlocking`:

```
1
2
3
4
5
Exception in thread "main" kotlinx.coroutines.JobCancellationException: BlockingCoroutine was cancelled; job="coroutine#1":BlockingCoroutine{Cancelled}@3327bd23
```

#### Making busy flow cancellable

In the case where you have a busy loop with coroutines you must explicitly check for cancellation. You can add `.onEach { currentCoroutineContext().ensureActive() }`, but there is a ready-to-use [cancellable](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.flow/cancellable.html) operator provided to do that:

fun main() = runBlocking<Unit> {

    (1..5).asFlow().cancellable().collect { value -> 

        if (value == 3) cancel()  

        println(value)

    } 

}






With the `cancellable` operator only the numbers from 1 to 3 are collected:

```
1
2
3
Exception in thread "main" kotlinx.coroutines.JobCancellationException: BlockingCoroutine was cancelled; job="coroutine#1":BlockingCoroutine{Cancelled}@5ec0a365
```

## Flow and Reactive Streams

For those who are familiar with [Reactive Streams](https://www.reactive-streams.org/) or reactive frameworks such as RxJava and project Reactor, design of the Flow may look very familiar.

Indeed, its design was inspired by Reactive Streams and its various implementations. But Flow main goal is to have as simple design as possible, be Kotlin and suspension friendly and respect structured concurrency. Achieving this goal would be impossible without reactive pioneers and their tremendous work. You can read the complete story in [Reactive Streams and Kotlin Flows](https://medium.com/@elizarov/reactive-streams-and-kotlin-flows-bfd12772cda4) article.

While being different, conceptually, Flow is a reactive stream and it is possible to convert it to the reactive (spec and TCK compliant) Publisher and vice versa. Such converters are provided by `kotlinx.coroutines` out-of-the-box and can be found in corresponding reactive modules (`kotlinx-coroutines-reactive` for Reactive Streams, `kotlinx-coroutines-reactor` for Project Reactor and `kotlinx-coroutines-rx2`/`kotlinx-coroutines-rx3` for RxJava2/RxJava3). Integration modules include conversions from and to `Flow`, integration with Reactor's `Context` and suspension-friendly ways to work with various reactive entities.


# Job

interface [Job](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/index.html) : [CoroutineContext.Element](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.coroutines/-coroutine-context/-element/index.html)([source](https://github.com/kotlin/kotlinx.coroutines/tree/master/kotlinx-coroutines-core/common/src/Job.kt#L120))

A background job. Conceptually, a job is a cancellable thing with a lifecycle that concludes in its completion.

Jobs can be arranged into parent-child hierarchies where the cancellation of a parent leads to the immediate cancellation of all its [children](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/children.html) recursively. Failure of a child with an exception other than [CancellationException](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-cancellation-exception/index.html) immediately cancels its parent and, consequently, all its other children. This behavior can be customized using [SupervisorJob](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-supervisor-job.html).

The most basic instances of the `Job` interface are created like this:

- A **coroutine job** is created with the [launch](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/launch.html) coroutine builder. It runs a specified block of code and completes upon completion of this block.
    
- [**CompletableJob**](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-completable-job/index.html) is created with a `Job()` factory function. It is completed by calling [CompletableJob.complete](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-completable-job/complete.html).
    

Conceptually, an execution of a job does not produce a result value. Jobs are launched solely for their side effects. See the [Deferred](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-deferred/index.html) interface for a job that produces a result.

### Job states

A job has the following states:

|**State**|[isActive](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/is-active.html)|[isCompleted](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/is-completed.html)|[isCancelled](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/is-cancelled.html)|
|---|---|---|---|
|_New_ (optional initial state)|`false`|`false`|`false`|
|_Active_ (default initial state)|`true`|`false`|`false`|
|_Completing_ (transient state)|`true`|`false`|`false`|
|_Cancelling_ (transient state)|`false`|`false`|`true`|
|_Cancelled_ (final state)|`false`|`true`|`true`|
|_Completed_ (final state)|`false`|`true`|`false`|

Note that these states are mentioned in italics below to make them easier to distinguish.

Usually, a job is created in the _active_ state (it is created and started). However, coroutine builders that provide an optional `start` parameter create a coroutine in the _new_ state when this parameter is set to [CoroutineStart.LAZY](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-coroutine-start/-l-a-z-y/index.html). Such a job can be made _active_ by invoking [start](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/start.html) or [join](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/join.html).

A job is in the _active_ state while the coroutine is working or until the [CompletableJob](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-completable-job/index.html) completes, fails, or is cancelled.

Failure of an _active_ job with an exception transitions the state to the _cancelling_ state. A job can be cancelled at any time with the [cancel](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/cancel.html) function that forces it to transition to the _cancelling_ state immediately. The job becomes _cancelled_ when it finishes executing its work and all its children complete.

Completion of an _active_ coroutine's body or a call to [CompletableJob.complete](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-completable-job/complete.html) transitions the job to the _completing_ state. It waits in the _completing_ state for all its children to complete before transitioning to the _completed_ state. Note that _completing_ state is purely internal to the job. For an outside observer, a _completing_ job is still active, while internally it is waiting for its children.

```kotlin
                                          wait children
    +-----+ start  +--------+ complete   +-------------+  finish  +-----------+
    | New | -----> | Active | ---------> | Completing  | -------> | Completed |
    +-----+        +--------+            +-------------+          +-----------+
                     |  cancel / fail       |
                     |     +----------------+
                     |     |
                     V     V
                 +------------+                           finish  +-----------+
                 | Cancelling | --------------------------------> | Cancelled |
                 +------------+                                   +-----------+
```

A `Job` instance in the [coroutineContext](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.coroutines/coroutine-context.html) represents the coroutine itself.

### Cancellation cause

A coroutine job is said to _complete exceptionally_ when its body throws an exception; a [CompletableJob](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-completable-job/index.html) is completed exceptionally by calling [CompletableJob.completeExceptionally](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-completable-job/complete-exceptionally.html). An exceptionally completed job is cancelled, and the corresponding exception becomes the _cancellation cause_ of the job.

Normal cancellation of a job is distinguished from its failure by the exception that caused its cancellation. A coroutine that throws a [CancellationException](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-cancellation-exception/index.html) is considered to be _cancelled_ normally. If a different exception causes the cancellation, then the job has _failed_. When a job has _failed_, its parent gets cancelled with the same type of exception, thus ensuring transparency in delegating parts of the job to its children.

Note, that the [cancel](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/cancel.html) function on a job only accepts a [CancellationException](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-cancellation-exception/index.html) as a cancellation cause, thus calling [cancel](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/cancel.html) always results in a normal cancellation of a job, which does not lead to cancellation of its parent. This way, a parent can [cancel](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-job/cancel.html) his children (cancelling all their children recursively, too) without cancelling himself.

### Concurrency and synchronization

All functions on this interface and on all interfaces derived from it are **thread-safe** and can be safely invoked from concurrent coroutines without external synchronization.

# Docs

## SupervisorJob

```kotlin
fun SupervisorJob(parent: Job? = null): CompletableJob
```

Creates a _supervisor_ job object in an active state. Children of a supervisor job can fail independently of each other.

A failure or cancellation of a child does not cause the supervisor job to fail and does not affect its other children, so a supervisor can implement a custom policy for handling failures of its children:

- A failure of a child job that was created using [launch](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/launch.html) can be handled via [CoroutineExceptionHandler](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-coroutine-exception-handler/index.html) in the context.
    
- A failure of a child job that was created using [async](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/async.html) can be handled via [Deferred.await](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-deferred/await.html) on the resulting deferred value.
    

If a [parent](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-supervisor-job.html) job is specified, then this supervisor job becomes a child job of the [parent](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-supervisor-job.html) and is cancelled when the parent fails or is cancelled. All this supervisor's children are cancelled in this case, too.