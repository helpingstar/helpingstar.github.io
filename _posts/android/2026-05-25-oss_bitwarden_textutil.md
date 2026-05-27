---
layout: single
title: "오픈소스로 알아보는 안드로이드 : 2. Bitwarden의 TextUtil"
date: 2026-05-24 05:27:53
lastmod : 2026-05-24 05:27:53
categories: android
tag: []
toc: true
toc_sticky: true
published: true
---

개발 도중 ViewModel에 context가 주입되고 있는 것을 발견했다. 왜인지 모르겠지만 본능적인 거부감을 느꼈다. 

다국어 지원을 위한 코드였는데, ViewModel에서 아래와 같이 string을 호출하고 있었다.

```kotlin
context.getString(...)
```

그래서 다른 오픈소스를 뒤져본 결과 ViewModel에 Context가 주입된 사례는 찾아볼 수 없었다. 

왜 그런가 해보니 요약해보면 다음과 같은 이유로 ViewModel에서는 context를 주입하지 않는다고 한다.

Activity 보다 오래 살아남는 ViewModel이 Activity Context에 대한 참조를 들고 있으면, 이미 파괴되어야 할 Activity가 ViewModel 때문에 GC되지 못한다.  
-> 메모리 누수, crash 발생 가능


그래서 Bitwarden의 코드를 보고 이것을 어떻게 해결했는지 참고하고자 한다.

코드는 아래에서 볼 수 있다.

[https://github.com/bitwarden/android/blob/main/ui/src/main/kotlin/com/bitwarden/ui/util/Text.kt](https://github.com/bitwarden/android/blob/main/ui/src/main/kotlin/com/bitwarden/ui/util/Text.kt)

## 사전 정보

Bitwarden은 strings를 다음과 같이 저장하고 있다.

```
ui/src/main/res/
├ values/            ← 기본값 (영어)
│     └ strings.xml
├ values-ko-rKR/     ← 한국어
│     └ strings.xml
├ values-ja-rJP/     ← 일본어
│     └ strings.xml
├ values-de-rDE/     ← 독일어
│     └ strings.xml
└ ... (총 60개 이상의 언어)
```

그리고 다음과 같이 typealias를 사용한다.

```kotlin
typealias BitwardenString = com.bitwarden.ui.R.string
```

결국 어느 모듈의 `R.string`인지 헷갈리지 않고 쉽게 호출하기 위한 수단이다.

## 선 요약

1. `Text`는 아직 해석되지 않은 문자열로, ViewModel은 "Resource ID를 보여줘"라는 의도만 `Text` 객체에 만들어 상태에 담는다.
2. 실제 `String`으로의 변환은 Compose UI 계층에서 Resource가 있을 때 마지막 순간에 일어난다.

```
ViewModel: state = errorText: Text (R.string.xxx라는 레시피만 보관)
↓
Compose UI: Text() -> invoke(LocalResources.current) -> 실제 String으로 해석
```

## 사례

이제 사례로 하나하나씩 살펴보자 자세한 동작 원리는 코드 사례를 먼저 살펴보고 서술한다.

코드 한줄 한줄보다는 일단 전체적인 ViewModel에서 String이 필요한 다양한 사례에서 어떻게 우회했는지를 본다. `asText()`를 보면 일단 적당히 바꿨겠거니 생각해보자. 자세한 코드 원리는 마지막에 다룬다. 그리고 다시 코드부분을 보면 이해가 더욱 될 것이다.

### 사례 1 : `StartRegistrationViewModel`

```kotlin
@Parcelize
data class StartRegistrationState(
    // ...
    val dialog: StartRegistrationDialog?, // entry point for Text into the UI state
) : Parcelable

sealed class StartRegistrationDialog : Parcelable {
    @Parcelize
    data class Error(
        val title: Text?, // nullable: 타이틀 없는 오류도 표현 가능
        val message: Text,
        val error: Throwable? = null,
    ) : StartRegistrationDialog()
}
```

```kotlin
private fun handleContinueClick() = when {
    // 케이스 A: 이메일 미입력
    state.emailInput.isBlank() -> {
        mutableStateFlow.update {
            it.copy(
                dialog = StartRegistrationDialog.Error(
                    title = BitwardenString.an_error_has_occurred.asText(),
                    message = BitwardenString.validation_field_required
                        .asText(BitwardenString.email_address.asText()),  // ← Text 중첩!
                ),
            )
        }
    }
    // 케이스 B: 이메일 형식 오류
    !state.emailInput.isValidEmail() -> {
        mutableStateFlow.update {
            it.copy(
                dialog = StartRegistrationDialog.Error(
                    title = BitwardenString.an_error_has_occurred.asText(),
                    message = BitwardenString.invalid_email.asText(),
                ),
            )
        }
    }
    // ...
}
```


**왜 여기에서 Text가 사용되었어야만 했을까?**

`handleContinueClick`는 ViewModel 안에서 동기적으로 실행되는 순수 비즈니스 로직이다. 그런데 state 검증의 결과물에 사용자에게 보일 에러 메시지가 포함되게 된다.

따라서 `BitwardenString.validation_field.required.asText(...)`로 의도만 상태에 담고, 실제 String 변환은 Resource가 있는 Compose 계층으로 미룬다.

**중첩된 `asText`**

해당 코드는 다음과 같다.

```kotlin
BitwardenString.validation_field_required.asText(
    BitwardenString.email_address.asText()
)
```

각각의 BitwardenString을 뜯어보면 다음과 같다.

```xml
<string name="validation_field_required">The %1$s field is required.</string>
<string name="email_address">Email address</string>
```

인자인 `email_address`도 현지화 대상 리소스이고, `validation_field_required`도 인자가 필요한 포맷 문자열이므로 이와 같은 `.asText(...asText())` 형태가 등장한다

이는 아래와 같이 사용된다.

```kotlin
@Composable
private fun StartRegistrationDialogs(
    dialog: StartRegistrationDialog?,
    onDismissRequest: () -> Unit,
) {
    when (dialog) {
        is StartRegistrationDialog.Error -> {
            BitwardenBasicDialog(
                title = dialog.title?.invoke(), // nullable이므로 ?.invoke()
                message = dialog.message(), // invoke() → LocalResources로 해석
                throwable = dialog.error,
                onDismissRequest = onDismissRequest,
            )
        }
        // ...
    }
}
```

- `dialog.title?` : `Text?`
- `dialog.message` : `Text`

### 사례 2 : `CompleteRegistrationViewModel`

```kotlin
sealed class CompleteRegistrationDialog : Parcelable {
    @Parcelize
    data class HaveIBeenPwned(
        val title: Text,    // 유출된 비밀번호 경고 제목
        val message: Text,  // 경고 본문
    ) : CompleteRegistrationDialog()

    @Parcelize
    data class Error(
        val title: Text?,
        val message: Text,
        val error: Throwable? = null,
    ) : CompleteRegistrationDialog()
}
```

```kotlin
when (val registerAccountResult = action.registerResult) {
    // 케이스 A: API 오류 — 서버 메시지(String)와 로컬 리소스를 같은 Text 필드에
    is RegisterResult.Error -> {
        mutableStateFlow.update {
            it.copy(
                dialog = CompleteRegistrationDialog.Error(
                    title = BitwardenString.an_error_has_occurred.asText(),
                    message = registerAccountResult.errorMessage?.asText()  // 서버 String → Text
                        ?: BitwardenString.generic_error_message.asText(),  // 로컬 리소스 → Text
                ),
            )
        }
    }

    // 케이스 B: Have I Been Pwned 경고
    RegisterResult.DataBreachFound -> {
        mutableStateFlow.update {
            it.copy(
                dialog = CompleteRegistrationDialog.HaveIBeenPwned(
                    title = BitwardenString.exposed_master_password.asText(),
                    message = BitwardenString
                        .password_found_in_a_data_breach_alert_description.asText(),
                ),
            )
        }
    }
}
```

```kotlin
registerAccountResult.errorMessage?.asText()
    ?: BitwardenString.generic_error_message.asText()
```

`registerAccountResult.errorMessage`의 경우 아래와 같이 `String?` 이다

```kotlin
sealed class RegisterResult {
    ...
    data class Error(
        val errorMessage: String?,
        val error: Throwable?,
    ) : RegisterResult()
    ...
}
```

근데 왜 `asText()`가 붙을까?

간단하다, `CompleteRegistrationDialog.Error.message`의 타입이 `Text` 이기 때문이다.

Elvis 연산자(`?:`) 는 양쪽 타입이 같아야 결과 타입이 의도한 대로 나온다.

위와 같이 다루는 문자열의 성격이 달랐다.
- 서버 응답에 담겨온 `String?` - 이미 완성된 문자열, 현지화 불필요(서버가 결정)
- 서버가 메시지를 안 줬을 때의 폴백 - 클라이언트 리소스 ID, 표시 직전에 현지화 필요

결국 서로 다른 성격의 문자열을 한 곳에서 보여줘야 하므로 `asText()`로 한 곳으로 합쳐서 보여준다. 안그러면 필드가 2개가 되거나 `String`을 위해 context를 사용해야 했을 것이다.

```kotlin
when (val dialog = state.dialog) {
    is CompleteRegistrationDialog.Error -> {
        BitwardenBasicDialog(
            title = dialog.title?.invoke(),
            message = dialog.message(),   // 서버 문자열이든 로컬 리소스든 동일하게 처리
            throwable = dialog.error,
            onDismissRequest = handler.onDismissErrorDialog,
        )
    }
    is CompleteRegistrationDialog.HaveIBeenPwned -> {
        BitwardenTwoButtonDialog(
            title = dialog.title(),       // non-null이므로 ?.invoke() 불필요
            message = dialog.message(),
            // ...
        )
    }
}
```

### 사례 3 : `SendViewModel`

```kotlin
private fun handleInternetConnectionErrorReceived() {
    mutableStateFlow.update {
        it.copy(
            isRefreshing = false,
            dialogState = SendState.DialogState.Error(
                BitwardenString.internet_connection_required_title.asText(),
                BitwardenString.internet_connection_required_message.asText(),
            ),
        )
    }
}
```

- Flow 수집 중 네트워크 오류 이벤트를 받았을 때 호출된다.
- 함수 시그니처 어디에도 `Context`, `Resource`가 없다. 리소스 ID를 `Text`로 포장하는 것으로 충분하다.

데이터 계층의 Flow에서 비동기로 흘러온 이벤트에 반응해 실행된다. 근데 로직상 그 위치가 현재 ViewModel이다.
- `@Composable` 스코프가 아니기 때문에 `LocalResources.current`를 읽을 수 없다.
- Context를 들고 있을 수도 없다.

이를 해결하기 위해 `Text` 를 사용한다.
- 문자열을 만들지 않고 만들 방법(리소스 ID)만 담는다.
- `BitwardenString.xxx.asText()` : Resource를 전혀 건드리지 않음
  - 백그라운드 코루틴 안에서도 안전하게 만들 수 있다.
- 실제 해석은 나중에 Compose가 그릴 때 일어난다.

```kotlin
@Composable
private fun SendDialogs(
    dialogState: SendState.DialogState?,
    onDismissRequest: () -> Unit,
) {
    when (dialogState) {
        is SendState.DialogState.Error -> BitwardenBasicDialog(
            title = dialogState.title?.invoke(),
            message = dialogState.message(), // 여기서 비로소 Resources 접근
            onDismissRequest = onDismissRequest,
            throwable = dialogState.throwable,
        )
        is SendState.DialogState.Loading -> BitwardenLoadingDialog(
            text = dialogState.message(), // Loading 다이얼로그도 Text로 메시지 표현
        )
        // ...
    }
}
```

### 정리

결국 핵심은 
- 문자열이 필요한 시점(UI)
- 문자열을 결정하는 시점(ViewModel)을
Text로 분리한다.

## 동작 흐름

```
[생성 시점] asText() 호출
  → Text 구현체 인스턴스 생성 (문자열은 아직 없음, ID/원본만 보관)
            ↓
State에 담겨 ViewModel에 보관, Parcel로 저장됨
            ↓
[해석 시점] Composable에서 text() 호출
  → invoke(Resources)
  → 실제 String 반환
```

### 1단계 : 생성

```kotlin
fun @receiver:StringRes Int.asText(): Text = ResText(this)
```

**`Int.asText()` : 확장 함수 선언**

- `R.string.dark` 는 실제로 Int 값이다. 그러므로 이 확장 함수 덕분에 다음 처럼 쓸 수 있게 된다.

```kotlin
R.string.dark.asText()
BitwardenString.dark.asText()
```

**`this` : 수신자를 가리키는 키워드**

확장 함수 본문 앞에서 `this`는 수신자(receiver), 즉 점(.) 앞의 값을 가리킨다.

- `this`
- 수신자 `Int` 값
- `R.string.dark`의 실제 정수값

결국, 그 값이 그대로 `ResText(this)` 생성자에 전달되어, 정수 ID 하나만 품은 데이터 객체가 만들어 진다. 이 시점에 `Resources`는 등장하지 않는다.

**`@receiver:StringRes` : 어노테이션 사용 지점 타깃**

- `receiver:` : 이 어노테이션을 확장 함수의 수신자(int 값)에 적용하라는 지시
- 이로 인해 Lint가 다음을 정적으로 검사한다

```kotlin
BitwardenString.dark.asText() // O : R.string.* 정수 → 통과
42.asText() // X : 임의의 정수 → Lint 경고 발생
```

아래와 같이 `ResText` 생성자에도 동일한 사용지점 타깃 문법이 쓰인다.

```kotlin
@Parcelize
private data class ResText(@field:StringRes private val id: Int) : Text {
    override fun invoke(res: Resources): CharSequence = res.getText(id)
}
```

참고 : 예를 들어 `@receiver:` 없이

```kotlin
fun @StringRes Int.asText(): Text = ResText(this)
```
와 같이 되었다면, 반환값(return) 자리를 먼저 시도한다.

**두 오버로드 : 인자 유무로 구현체를 다르게 선택**

```kotlin
fun @receiver:StringRes Int.asText(): Text = ResText(this)

fun @receiver:StringRes Int.asText(vararg args: Any): Text = ResArgsText(this, args.asList())
```

Kotlin 컴파일러는 호출 시 인자 개수를 보고 자동으로 맞는 오버로드를 생성한다.

```kotlin
BitwardenString.invalid_email.asText()
// 인자 없음 → ResText(id) 생성

BitwardenString.validation_field_required.asText(BitwardenString.email_address.asText())
// 인자 있음 → ResArgsText(id, [emailAddressText]) 생성
```

**수신자 타입으로 구분되는 같은 이름의 함수**

```kotlin
fun String.asText(): Text = StringText(this)
fun @receiver:StringRes Int.asText(): Text = ResText(this)
```

### 2단계 : 보관

생성된 `Text`는 ViewModel의 State에 담겨 한동안 머문다.

- data class : `ResText(id = 123)`은 data class이므로 컴파일러가 다음 메소드를 자동으로 생성한다.
  - `equals()`
  - `hashCode()`
  - `toString()`
  - `copy()`
  - 객체가 달라도 내부 `id`를 통해 `equals`를 비교한다.
  - 테스트에서 `Resources` 없이 상태 비교가 가능하고, Compose에서 State가 바뀌었는지 판단시 이 값 동등성을 판단한다.
- `@Parcelize` : 컴파일 타임에 다음 구현을 생성
  - `writeToParcel()`
  - `createFromParcel()`
  - 즉 `RestText`는 자신의 `id`(정수)를 안드로이드의 `Parcel`(바이트 버퍼)에 직렬화할 수 있다.

```kotlin
@Parcelize
private data class ResArgsText(
    @field:StringRes private val id: Int,
    private val args: @RawValue List<Any>,
) : Text

@Parcelize
private data class PluralsText(
    @field:PluralsRes private val id: Int,
    private val quantity: Int,
    private val args: @RawValue List<Any>,
) : Text
```
- `@RawValue`가 필요한 이유
  - `@Parcelize`는 컴파일 타임에 각 필드 타입을 보고 직렬화 코드를 생성
  - `Any`는 컴파일러가 `Parcel`에 쓰는 방법을 알 수 없어 컴파일 에러를 낸다.
  - `@RawValue`는 타입 검사를 끄고 런타임에 `writeValue()`로 일반 직렬화를 시도하라고 지시.

### 3단계 : 해석 : invoke의 이중 구조와 LocalResources

`invoke`가 두 개의 오버로드로 존재한다.

```kotlin
@Immutable
interface Text : Parcelable {
    // (A)
    @Composable
    operator fun invoke(): String {
        return toString(LocalResources.current)
    }
    // (B)
    operator fun invoke(res: Resources): CharSequence
    
    ...
}
```

**(A) 인자 없는 invoke() - UI에서의 진입점**

`Text` 타입 변수가 있을 때, Compoable 안에서 `()`를 붙여 호출하면 이 버전이 실행된다.

예를 들어

```kotlin
// val message: Text = dialogState.message (Text 타입 변수)

message()  // operator fun invoke() 호출
message.invoke()  // 위와 동일
```

1. `@Composable`이므로 Compose 런타임 안에서만 호출 가능하다.
2. `LocalResources.current`로 현재 Composition에 흐르고 있는 `Resources`를 꺼낸다.
   - CompositionLocal : UI 트리를 따라 암묵적으로 전파되는 값, 어느 노드에서나 현재 환경값(여기서는 `Resources`)을 읽을 수 있다.
3. 꺼낸 `Resources`를 `toString(res)`에 넘긴다.

**(B) Resource를 받는 invoke(res) - 실제 해석 작업**

각 구현체가 자신만의 방식으로 `operator fun invoke(res: Resources): CharSequence`를 override한다.

```kotlin
@Parcelize
private data class ResText(@field:StringRes private val id: Int) : Text {
    override fun invoke(res: Resources): CharSequence = res.getText(id)
}
```
- Resource에 정수 ID를 건네 해당 언어 문자열 조회

```kotlin
@Parcelize
private data class StringText(private val string: String) : Text {
    override fun invoke(res: Resources): String = string
}
```

- 이미 완성된 문자열이라 Resource 불필요 -> 그냥 돌려줌

```kotlin
@Parcelize
private data class ResArgsText(
    @field:StringRes private val id: Int,
    private val args: @RawValue List<Any>,
) : Text {
    override fun invoke(res: Resources): String =
        res.getString(id, *convertArgs(res, args).toTypedArray())

    override fun toString(): String = "ResArgsText(id=$id, args=${args.contentToString()})"
}
```

- 리소스 조회 + 인자 치환을 한 번에 

```kotlin
@Parcelize
private data class ResText(@field:StringRes private val id: Int) : Text {
    override fun invoke(res: Resources): CharSequence = res.getText(id)
}
```

위 코드를 예시로 하나씩 살펴보자

1) id : 생성 시점, ViewModel이 제공

```kotlin
// ViewModel 코드
BitwardenString.internet_connection_required_title.asText()
```

```kotlin
fun @receiver:StringRes Int.asText(): Text = ResText(this)
```

- 코드 호출 -> `ResText(this)` 실행
- `this = R.string.internet_connection_required_title`의 정수값 (예: 2131820456)
- `asText()` 호출 -> `ResText(id = 2131820456)` 객체가 만들어짐.
- `id`는 생성자를 통해 객체 필드에 저장
- 이 시점에 `res`는 없다.

2) res : 호출 시점, Compose가 제공

`invoke(res: Resources)` 의 `res`는 생성자 파라미터가 아닌, 메서드 파라미터이므로, `invoke(res)`가 호출될 때마다 외부에서 넘겨받는다.

```kotlin
@Immutable
interface Text : Parcelable {
    @Composable
    operator fun invoke(): String {
        // Compose가 이 시점에 Activity의 Resources를 꺼내 직접 인자로 넘겨준다.
        return toString(LocalResources.current)
    }
}
```

1. `Composable` 안에서 `text()` 가 호출'
2. `LocalResources.current` 로 현재 Resources 를 꺼내 invoke(res) 에 인자로 전달
3. res를 건네주는 주체는 Compose 런타임

두 입력이 `invoke` 안에서 만남

```kotlin
override fun invoke(res: Resources): CharSequence = res.getText(id)
```

- `res` : Compose가 방금 넘겨준 Resources
  - 이 호출때만 존재
- `res.getText(id)`의 `id` : ViewModel이 넣어둔 정수 ID
  - 객체 생성 때부터 보관

```
[ViewModel — 생성 시점]

BitwardenString.xxx.asText()
  │
  └ ResText(id = 2131820456) 생성
        id = 2131820456  ← 필드에 저장
        (res는 없음, 필요도 없음)

            ... State에 담겨 보관, Parcel로 저장/복원 가능 ...

  
[Composable — 호출 시점]

text()
  └ invoke()    ← @Composable
    └ toString(LocalResources.current)
        │
        res = Activity의 Resources  ← Compose가 주입
        │
        └ invoke(res)
            │
            └ res.getText(id)
              ↑             ↑
            방금 받은 도구   오래전에 봉인된 ID
                │
                └ "An error has occurred"  (실제 문자열)

```

- 여기서 `toString`

```kotlin
@Immutable
interface Text : Parcelable {
    ...
    fun toString(res: Resources): String = invoke(res).toString()
}
```

위 함수를 오버라이드한

```kotlin
@Parcelize
private data class ResText(@field:StringRes private val id: Int) : Text {
    override fun invoke(res: Resources): CharSequence = res.getText(id)
}
```

이 호출된다.

## 마무리

나머지도 이런식으로 따라가다보면 호출 구조가 보일 것이다.