---
layout: single
title:  "EffectiveCpp 43:템플릿으로 만들어진 기본 클래스 안의 이름에 접근하는 방법을 알아두자"
categories: EffectiveCpp
tag: [cpp, c++]
toc: true
toc_sticky: true
---


## **ex 1**

다른 몇 개의 회사에 매시지를 전송할 수 있는 응용프로그램을 만든다, 전송용 메시지는 암호화될 수도 있다. 어떤 메시지가 어떤 회사로 전송될지를 컴파일 도중에 결정할 수 있는 충분한 정보가 있다면, 주저 없이 템플릿 기반의 방법을 쓸 수 있다.

```cpp
// [코드 43-1]
class CompanyA {
public:
    ...
    void sendClearText(const std::string& msg);
    void sendEncrrypted(const std::string& msg);
    ...
};
 
class CompanyB {
public:
    ...
    void sendClearText(const std::string& msg);
    void sendEncrrypted(const std::string& msg);
    ...
};
... // 다른 회사들을 나타내는 각각의 클래스
// 메시지 생성에 사용되는 정보를 담기 위한 클래스
class MsgInfo { ... };

template <typename Company>
class MsgSender {
public:
    ... // 생성자, 소멸자 등
    void sendClear(const MsgInfo& info)
    {
        std::string msg;
        
        (info로 부터 msg를 만든다.)
        
        Company c;
        c.sendClearText(msg);
    }
    // sendClear 함수와 비슷하다 c.sendEncrypted 함수 호출이 차이
    void sendSecret(const MsgInfo& info)
    { ... }
};
```
메시지를 보낼 때마다 관련 정보를 로그로 남기고 싶어, 파생 클래스를 사용하여 이 기능을 붙이려 한다.
```cpp
// [코드 43-2]
template<typename Company>
class LoggingMasgSender : public MsgSender<Company> {
public:
    ...
    void sendClearMsg(const MsgInfo& info)
    {
        ("메시지 전송 전" 정보를 로그에 기록)
        // 기본 클래스의 함수를 호출하는데
        // 이 코드는 컴파일 되지 않는다.
        sendClear(info);
        ("메시지 전송 후" 정보를 로그에 기록)
    }
    ...
};
```

파생 클래스의 메시지 전송 함수의 이름(`sendClearMsg`)이 기본 클래스에 있는 것(`sendClear`)과 다르다. 이는 기본클래스로부터 물려받은 이름을 파생 클래스에서 가리키는 문제, 상속받은 비가상 함수를 재정의하는 문제를 일으키지 않도록 한다.

하지만 '`sendClear` 함수가 존재하지 않는다' 는 이유로 인해 컴파일 되지 않는다.

문제는 간단하다 **컴파일러가 `LoggingMsgSender` 클래스 템플릿의 정의와 마주칠 때, 컴파일러는 대체 이 클래스가 어디서 파생된 것인지를 모른다는 것이다.**

`MsgSender<Company>` 인 것은 맞다. 하지만 `Company`는 템플릿 매개변수이고, 이 템플릿 매개변수는 나중(`LoggingMsgSender`가 인스턴스로 만들어질 때)까지 무엇이 될지 알 수 없다.

`Company`가 정확히 무엇인지 모르는 상황에서는 `MsgSender<Company>` 클래스가 어떤 형태인지 알 방법이 없고, 이러니 `sendClear` 함수가 들어 있는지 없는지 알아낼 방법이 없는 것도 당연하다.

## **ex 2**
`CompanyZ`라는 클래스가 있고, 이 클래스는 암호화된 통신만을 사용해야 한다.
```cpp
// [코드 43-3]
class CompanyZ {
public:
    ...
    void sendEndcrypted(const std::string& msg);
    ...
};
```
[코드 43-1]의 `MsgSender` 템플릿은 `CompanyZ` 객체의 설계 철학과 맞지 않는 `sendClear` 함수를 제공하기 때문에 그대로 사용할 수 없다. 이 부분을 바로 잡기 위해, `CompanyZ`를 위한 `MsgSender`의 특수화 버전을 만들 수 있다.

```cpp
// MsgSender 템플릿의 완전 특수화 버전.
// sendClear 함수가 빠진 것만 제외하면 일반형 템플릿과 같다.
template<>
class MsgSender<CompanyZ> {
public:
    ...
    void sendSecret(const MsgInfo& info)
    {...}
}
```

`template<>` : 이건 템플릿도 아니고 클래스도 아니다,

위의 코드는 `MsgSender` 템플릿을 템플릿 매개변수가 `CompanyZ`일 때 쓸 수 있도록 특수화한 버전, 이것을 **완전 템플릿 특수화(total template specialization)** 라고 한다.

`MsgSender` 템플릿이 `CompanyZ` 타입에 대해 특수화되었고, 이때 이 템플릿의 매개변수들이 하나도 빠짐없이 구체적인 타입으로 정해진 상태라는 뜻이다. 즉, 일단 타입 매개변수가 `CompanyZ`로 정의된 이상 이 템플릿(특수화된)의 매개변수로는 다른 것이 올 수 없게 된다는 이야기이다.

`MsgSender` 템플릿이 `CompanyZ`에 대해 특수화된 상태라고 가정하고, `LoggingMsgSender`로 다시 돌아와 보자
```cpp
// [코드 43-2]
template<typename Company>
class LoggingMasgSender : public MsgSender<Company> {
public:
    ...
    void sendClearMsg(const MsgInfo& info)
    {
        ("메시지 전송 전" 정보를 로그에 기록)
        // Company == CompanyZ라면
        // 이 함수는 있을 수 없다.
        sendClear(info);
        ("메시지 전송 후" 정보를 로그에 기록)
    }
    ...
};
```
기본클래스가 `MsgSender<CompanyZ>`이면 `MsgSender<CompanyZ>` 클래스에는 `sendClear` 함수가 없기 때문에 이 코드는 말이 안된다.

기본 템플릿은 언제라도 특수화될 수 있고, 이런 특수화 버전에서 제공하는 이넡페이스가 원래의 일반형 템플릿과 꼭 같으리라는 법은 없다는 것을 `C++`가 인식한다는 이야기이다.

이렇기 때문에 `C++` 컴파일러는 템플릿으로 만들어진 기본 클래스를 뒤져서 상속된 이름을 찾는 것을 거부한다. 어떤 의미로 보면 `C++`의 하위 언어들 중 한 부분인 객체지향 `C++`에서 템플릿 `C++`로 옮겨 갈 떄 상속 메커니즘이 끊기는 것이다.

## **템플릿화된 기본 클래스를 멋대로 뒤지지 않는 방법**
### **방법 1 : 기본 클래스 함수에 대한 호출문 앞에 "this->"를 붙인다.**
```cpp
// [코드 43-4]
template<typename Company>
class LoggingMasgSender : public MsgSender<Company> {
public:
    ...
    void sendClearMsg(const MsgInfo& info)
    {
        ("메시지 전송 전" 정보를 로그에 기록)
        // sendClear가 상속되는 것으로 가정한다.
        this->sendClear(info);
        ("메시지 전송 후" 정보를 로그에 기록)
    }
    ...
};
```
### **방법 2 : using 선언을 사용한다.**
*[Item 33] 가려진 기본 클래스의 이름을 파생 클래스의 유효범위에 끌어오는 용도로 `using`을 이용할 수 있다.*
```cpp
// [코드 43-5]
template<typename Company>
class LoggingMasgSender : public MsgSender<Company> {
public:
    // 컴파일러에게 sendClear 함수가 기본 클래스에 있다고
    // 가정하라고 알려준다.
    using MsgSender<Company>::sendClear;
    ...
    void sendClearMsg(const MsgInfo& info)
    {
        ("메시지 전송 전" 정보를 로그에 기록)
        // sendClear가 상속되는 것으로 가정한다.
        sendClear(info);
        ("메시지 전송 후" 정보를 로그에 기록)
    }
    ...
};
```
 하지만 이 문제에서는 기본 클래스의 이름이 파생 클래스에서 가려지는 것이 아니라, 기본 클래스(템플릿화된)의 유효범위를 뒤지라고 우리가 컴파일러에게 알려 주지 않으면 컴파일러가 알아서 찾는 일이 없다는 것이다.

### **방법 3 : 호출할 함수가 기본 클래스의 함수라는 점을 명시적으로 지정한다**
```cpp
// [코드 43-5]
template<typename Company>
class LoggingMasgSender : public MsgSender<Company> {
public:
    ...
    void sendClearMsg(const MsgInfo& info)
    {
        ("메시지 전송 전" 정보를 로그에 기록)
        // sendClear가 상속되는 것으로 가정한다.
        MsgSender<Company>::sendClear(info);
        ("메시지 전송 후" 정보를 로그에 기록)
    }
    ...
};
```
추천하지는 않는 방법이다. 호출되는 함수가 가상 함수인 경우에는, 이런 식으로 명시적 한정을 해 버리면 가상 함수 바인딩이 무시되기 때문이다.

### **정리**
이름에 대한 가시성을 조작한다는 면에서 보면 세 가지 방법은 모두 동작 원리가 같다. 기본 클래스 템플릿이 이후에 어떻게 특수화되더라도 원래의 일반형 템플릿에서 제공하는 인터페이스를 그대로 제공할 것이라고 컴파일러에게 약속을 하는 것이다.

이런 약속은 `LoggingMsgSender` 등의 파생 클래스 템플릿을 컴파일러가 구문분석하는 데 반드시 필요하지만, 그 약속이 거짓이었다는 것이 들통나면 컴파일 과정에서 에러가 발생한다.

예를 들어
```cpp
LoggingMsgSender<CompanyZ> zMsgSender;
MsgInfo msgData;
...             // msgData에 정보를 채운다
zMsgSender.sendClearMsg(msgData); // 에러, 컴파일되지 않음
```
위 예시에서 `sendClearMsg` 호출문은 컴파일 되지 않는다. 기본 클래스가 `MsgSender<CompanyZ>`(템플릿 특수화 버전)라는 사실을 컴파일러가 알고 있는데다가, `sendClearMsg` 함수가 호출하려고 하는 `sendClear` 함수는 `MsgSender<CompanyZ>` 클래스에 안 들어 있다는 사실도 컴파일러가 알아챈 후이기 때문이다.

**본질적인 논점:**

기본 클래스의 멤버에 대한 참조가 무효한지를 컴파일러가 진단하는 과정이 미리(파생 클래스 템플릿의 정의가 구문분석될 때) 들어가느냐, 아니면 나중에 (파생 클래스 템플릿이 특정한 템플릿 매개변수를 받아 인스턴스화될 때) 들어가느냐가 바로 이번 항목의 핵심이다. 여기서 `C++`는 '이른 진단(early diagnose)'을 선호하는 정책으로 결정한 것이다. 그러면 이제 파생 클래스가 템플릿으로부터 인스턴스화될 때 컴파일러가 기본 클래스의 내용에 대해 아무것도 모르는 것으로 가정하는 이유도 이해할 수 있을 것이다.

*Scott Meyers, 『Effective C++』, 곽용재 옮김, 프로텍 미디어(2015), p306-312*