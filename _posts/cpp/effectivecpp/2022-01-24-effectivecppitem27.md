---
layout: single
title:  "EffectiveCpp 27:항목 27: 캐스팅은 절약, 또 절약! 잊지 말자"
categories: EffectiveCpp
tag: [cpp, c++]
toc: true
toc_sticky: true
---

타입 에러가 생기지 않도록 보장하는것이 C++ 동작 규칙의 바탕 철학이나 캐스트(cast)가 사용될 경우 찾아내기 어려울 경우가 많다. C++ 캐스팅은 조심해서 써야하는 기능이다.

## **캐스팅**
### **구형 스타일 캐스트**
#### **C 스타일 캐스트**
```c
(T) expression // 표현식 부분을 T 타입으로 캐스팅합니다.
```

#### **함수 방식 캐스트**
```c
T(표현식)       // 표현식 부분을 T 타입으로 캐스팅합니다.
```

### **신형 스타일 캐스트(C++ 스타일 캐스트)**
#### **`const_cast`**
```cpp
const_cast<T>(expression)
```
객체의 상수성(constness)을 없애는 용도로 사용된다. 이런 기능을 가진 C++ 스타일의 캐스트는 이것밖에 없다

#### **`dynamic_cast`**
```cpp
dynamic_cast<T>(expression)
```
'안전한 다운캐스팅(safe downcasting)'을 할 때 사용하는 연산자, 주어진 객체가 어떤 클래스 상속 계통에 속한 특정 타입인지 아닌지를 결정하는 작업에 쓰인다. 런타임 비용이 높은 캐스트 연산자이기도 하다
#### **`reinterpret_cast`**
```cpp
reinterpret_cast<T>(expression)
```
포인터를 `int`로 바꾸는 등의 하부 수준 캐스팅을 위해 만들어진 연산자, 적용 결과는 구현 환경에 의존적이다.
#### **`static_cast`**
```cpp
static_cast<T>(expression)
```
암시적 변환(비상수 객체 > 상수 객체, `int` > `double`)을 강제로 진행할 때 사용.

타입 변환을 거꾸로 수행하는 용도(`void*` > 일반타입 포인터, 기본 클래스 포인터 > 파생 클래스의 포인터)로도 쓰인다. 상수 객체를 비상수 객체로 캐스팅하는데는 사용할 수 없다.

## **C++ 스타일 캐스트를 쓰는것이 바람직한 이유**

1. 코드를 읽을 때 알아보기 쉽다.(사람 눈, `grep` 검색도구)
2. 소스코드 어디에서 `C++`의 타입 시스템이 망가졌는지를 찾아보는 작업이 편해진다.
3. 캐스트를 사용한 목적을 더 좁혀서 지정하기 때문에 컴파일러쪽에서 사용 에러를 진단할 수 있다.

## **캐스팅에 대한 사실**
### **일단 타입 변환이 있으면 이로 말미암아 런타임에 실행되는 코드가 만들어지는 경우가 적지 않다.**

```cpp
int x, y;
...
// 부동 소수점 나눗셈을 사용하여 x를 y로 나눈다
double d = static_cast<double>(x)/y;
```
`int`타입의 `x`를 `double`타입으로 캐스팅한 부분에서 코드가 만들어진다. 그것도 거의 항상 그렇다. 왜냐하면 대부분의 컴퓨터 아키텍처에서 `int`의 표현구조와 `double`의 표현 구조가 아예 다르기 때문이다.

```cpp
class Base {...};
class Derived: public Base {...};
Derived d;
// Derived* -> Base*의 암시적 변환이 이루어진다.
Base *pb = &d;
```
파생 클래스 객체에 대한 기본 클래스 포인터를 만드는 코드이다. 그런데 두 포인터 값이 같지 않을 때는 포인터의 변위(offset)를 `Derived*` 포인터에 적용하여 실제의 `Base*` 포인터 값을 구하는 동작이 런타임에 이루어진다.

객체 하나(ex) `Dervied` 타입의 객체)가 가질 수 있는 주소가 오직 한 개가 아니라 그 이상이 될 수 있다.(`Base*`포인터로 가키리 때의 주소, `Derived*` 포인터로 가리킬 때의 주소)

이런 현상은 `C`, `Java`, `C#`에서는 생길 수 없으나 `C++`에서는 생긴다. 다중 상속이 사용되면 이런 일이 항상 생기지만, 단일 상속인데도 이렇게 되는 경우가 있다.

## **ex1) 캐스트 연산자가 당기면 뭔가 꼬여가는 징조다.**

캐스팅이 들어가면 보기엔 맞는 것 같지만, 실제로는 틀린 코드를 쓰고도 모르는 경우가 많아진다.

응용프로그램 프레임워크를 살펴보면 가상 함수를 파생 클래스에서 재정의해서 구현할 때 기본 클래스의 버전을 호출하는 문장을 가장 먼저 넣어달라는 요구사항을 보게된다.

```cpp
// 기본 클래스
class Window {
    public:
    // 기본 클래스의 onResize 구현 결과
    virtual void onResize() {...}
    ...
};
// 파생 클래스
class SpecialWindow: public Window {
    public:
    // 파생 클래스의 onResize 구현 결과 *this를 Window로 캐스팅하고
    // 그것에 대해 onResize를 호출한다. 동작이 안되어서 문제
    virtual void onResize() {
        static_cast<Window>(*this).onResize();
        // SpecialWindow에서만 필요한 작업을 여기서 수행한다.
        ...
    }
}
```
`*this`를 `Window`로 캐스팅하는 코드이다. 이에 따라 호출되는 `onResize` 함수는 `Window::onResize`가 된다. 

이상한 점 : 함수 호출이 이루어지는 객체가 현재의 객체가 아니다

이 코드에서는 캐스팅이 일어나면서 `*this`의 기본 클래스 부분에 대한 사본이 임시적으로 만들어지게 되어 있는데, 지금의 `onResize`는 바로 이 임시 객체에서 호출된 것이다.

`SpecialWindow`만의 동작을 현재 객체에 대해 수행하기도 전에 **기본 클래스 부분의 사본**에 대고 `Window::onResize`를 호출한다.

이 문제를 해결하기 위해 일단 캐스팅을 빼버려야 한다. 컴파일러에게 `*this`를 기본 클래스 객체로 취급하도록 하는 것은 생각하지 말자. 그냥 현재 객체에 대고 `onResize`의 기본 클래스 버전을 호출하도록 만들면 된다.

```cpp
class SpecialWindow: public Window {
    public:
    virtual void onResize() {
        Window::onResize();
        ...
    }
    ...
};
```

`dynamic_cast`가 당길 수 있다. 하지만 상당수의 구현 환경에서 이 연산자가 정말 느리게 구현되어있다.

## **ex2) `dynamic_cast` 연산자가 쓰고 싶어질 때 이를 피해가는 법**
파생 클래스 객체임이 분명한 것이 있어서 이에 대해 파생 클래스의 함수를 호출하고 싶은데, 그 객체를 조작할 수 있는 수단으로 기본 클래스의 포인터(혹은 참조자)밖에 없을 경우는 적지 않게 생긴다. 이런 문제를 피해가는 일반적인 방법으로는 두 가지를 들 수 있다.

### **방법 1**
파생 클래스 객체에 대한 포인터(혹은 스마트 포인터)를 컨테이너에 담아둠으로써 각 객체를 기본 클래스 인터페이스를 통해 조작할 필요를 아예 없애버리는 것.

`Window` 및 `SpecialWindow` 상속 계통에서 깜빡거리기(blink) 기능을 `SpecialWindow`객체만 지원하게 되어 있다면, **[Code 27-1]** 처럼 하지 말고 **[Code 27-2]** 처럼 해보라는 것이다.

```cpp
// [Code 27-1]
class Window {...};

class SpecialWindow: public Window {
    public:
    void blink();
    ...
};

typedef std::vector<std::tr1::shared_ptr<Window>> VPW;
VPW winPtrs;
...
for (VPW::iterator iter = winPtrs.begin(); iter != winPtrs.end(); ++iter) {
    if (SpecialWindow *psw = dynamic_cast<SpecialWindow*>(iter->get()))
    psw->blink();
}
```
```cpp
// [Code 27-2]
typedef std::vector<std::tr1::shared_ptr<SpecialWindow>> VPSW;
VPSW winPtrs;
...
// dynamic_cast가 없다
for (VPSW::itertor iter = winPtrs.begin(); iter != winPtrs.end(); ++iter)
    (*iter)->blink();
```
이 방법으로는 `Window`에서 파생될 수 있는 모든 녀석들에 대한 포인터를 똑같은 컨테이너에 저장할 수는 없다. 다른 타입의 포인터를 담으려면 타입 안전성을 갖춘 컨테이너 여러 개가 필요할 것이다.

### **방법 2**
원하는 조작을 가상 함수 집합으로 정리해서 기본 클래스에 넣어두면 `Window`에서 뻗어 나온 자손들을 전부 기본 클래스 인터페이스를 통해 조작할 수 있다.

지금은 `blink` 함수가 `SpecialWindow`에서만 가능하지만, 그렇다고 비노 클래스에 못 넣어둘 만한 것도 아니다. 그러니까, 아무것도 안 하는 기본 `blink`를 구현해서 가상 함수로 제공한다.

```cpp
class Window {
    public:
    // 기본 구현은 '아무 동작 안하기'
    // item 34에 가상함수의 기본 구현이 안좋은 아이디어인지 확인 가능
    virtual void blink() {}
    ...
};

class SpecialWindow: public Window {
    public:
    // 이 클래스에서는 blink 함수가 특정한 동작 수행
    virtual void blink() {...}
    ...
};

// 이 컨테이너는 Windows에서 파생된 모든 타입의 객체
// (에 대한 포인터) 들을 담는다.
typedef std::vector<std::tr1::shared_ptr<Windows>> VPW;
VPW winPtrs;
...

for (VPW::iterator iter = winPtrs.begin(); iter != winptrs.end(); ++iter)
    // dynamic_cast 가 없다.
    (*iter)->blink();
```

### **방법 정리**
1. 타입 안전성을 갖춘 컨테이너를 쓴다.
2. 가상 함수를 기본 클래스 쪽으로 올린다.

두가지 방법 모두 모든 상황에 적용할 수는 없지만, 상당히 많은 상황에서 `dynamic_cast`를 쓰는 방법 대신에 꽤 잘 쓸 수 있다.

### **폭포식(cascading) `dynamic_cast`는 피하자**

```cpp
class Window {...};
...     // 파생클래스가 정의됨
typedef std::vector<std::tr1::shared_ptr<Windows>> VPW;
VPW winPtrs;
...
for (VPW::iteraotr iter = winPtrs.begin(); iter != winPtrs.end(); ++iter)
{
    if (SpecialWindow *psw1 = dynamic_cast<SpecialWindow1*>(iter->get()))
    {...}
    else if (SpecialWindow *psw2 = dynamic_cast<SpecialWindow2*>(iter->get())) 
    {...}
    else if (SpecialWindow *psw3 = dynamic_cast<SpecialWindow3*>(iter->get())) 
    {...}
}
```
파생 클래스가 하나 추가되면 폭포식 코드에 계속해서 조건분기문을 우겨 넣어야 한다.

이런 형태의 코드를 보면 가상 함수 호출에 기반을 둔 어떤 방법이든 써서 바꿔 놓아야 한다.

## **정리**
**정말 잘 작성된 `C++` 코드는 캐스팅을 거의 쓰지 않는다.**

그냥 막 쓰기에는 꺼림칙한 문법 기능을 써야할 때 흔히 쓰이는 수단을 활용해서 처리하는 것이 좋다. (=최대한 격리시키자)
캐스팅을 해야 하는 코드는 내부 함수 속에 몰아 놓고, 그 안에서 일어나는 '천한' 일들은 이 함수를 호출하는 외부에서 알 수 없도록 인터페이스로 막아두는 식으로 해결하면 된다.
