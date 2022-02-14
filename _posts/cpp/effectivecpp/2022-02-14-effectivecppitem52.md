---
layout: single
title: "위치지정 new를 작성한다면 위치지정 delete도 같이 준비하자"
date: 2022-02-14 16:07:48
lastmod : 2022-02-14 16:07:51
categories: EffectiveCpp
tag: [cpp, c++, new, delete]
toc: true
toc_sticky: true
---
# 기본형 `operator new/delete`의 경우
```cpp
Widget *pw = new Widget;
```
위에서는 함수 두 개가 호출된다. 메모리 할당을 위해 `operator new`가 호출되고 그 뒤를 이어 `Widget`의 기본 생성자가 호출된다.

여기서 첫 번째 함수 호출은 무사히 지나갔는데 두 번째 함수 호출이 진행되다가 예외가 발생했다면 첫 단계에서 진행된 메모리 할당을 취소해야 한다. `Widget` 생성자에서 예외가 튀어나오면 `pw`에 포인터가 대입될 일은 절대로 안 생기기 때문에 사용자 코드에서는 이 메모리를 해제할 수 없다. 따라서 1단계 메모리 할당을 안전하게 되돌리는 것은 `C++` 런타임 시스템이 맡게 된다.

이는 `C++` 런타임 시스템이 해주어야 하는 일은 1단계에서 자신이 호출한 `operator new` 함수와 짝이 되는 버전의 `operator delete` 함수를 호출한다. 하지만 이게 제대로 되려면 `operator delete` 함수들 가운데 어떤 것을 호출해야 하는지를 런타임 시스템이 제대로 알고 있어야만 가능하다. 하지만 상대하고 있는 `new`, `delete` 가 기본형 시그니처로 되어 있는 한 이 부분은 그다지 대수로운 사안이 아니다. 

왜냐하면 기본형 `operator new`는 기본형 `operator delete`와 짝을 맞추기 때문이다.

```cpp
void* operator new(std::size_t) throw(std::bad_alloc);
```
```cpp
// 전역 유효범위에서의 기본형 시그니처
void operator delete(void *rawMemory) throw();
// 클래스 유효범위에서의 전형적인 기본형 시그니처
void operator delete(void *rawMemory, std::size_t size) throw();
```
# 기본형이 아닌 `operator new/delete`의 경우
**비기본형 : 다른 매개변수를 추가로 갖는 `operator new`**
## ex1
어떤 클래스에 대해 전용으로 쓰이는 `operaotr new`를 만들고 있는데, 메모리 할당 정보를 로그로 기록해 줄 `ostream`을 지정받는 꼴로 만든다고 가정하자, 그리고 클래스 전용 `operator delete`는 기본형으로 만든다.

```cpp
// [예제 52-1]
class Widget {
public:
    ...
    // 비표준 형태의 operator new
    static void* operator new(std::size_t size, 
                              std::ostream logStream)                 
      throw(std::bad_alloc);
    // 클래스 전용 operator delete의 표준 형태
    static void operator delete(void *pMemory,   
        size_t size) throw();                    
    ...
};
```

`operator new` 함수는 기본형과 달리 매개변수를 추가로 받는 형태로도 선언할 수 있다. 이런 형태의 함수를 위치지정(placement) `new` 라고 한다.

위치지정 `new` 중에 유용한 것이 있는데. 객체를 생성시킬 메모리 위치를 나타내는 포인터를 매개변수로 받는 것이 그것이다. 생김새는 다음과 같다

```cpp
// 위치지정 new
void* operator new(std::size_t, void *pMemory) throw();
```

*[예제 52-1]* 의 `Widget`를 사용한 코드를 보자 `Widget` 객체 하나를 동적 할당할 때 `cerr`에 할당 정보를 로그로 기록하는 코드다
```cpp
// operator new를 호출하는 데 cerr을 ostream 인자로 넘기는데
// 이 때 Widget 생성자에서 예외가 발생하면 메모리가 누출된다.
Widget *pw = new (std::cerr) Widget;
```

`Widget` 생성자에서 예외가 발생했을 경우 `operator new`에서 저지른 할당을 되돌리는 일은 `C++` 런타임 시스템이 책임지고 해야 한다. 그런데 런타임 시스템쪽에는 호출된 `operator new`가 어떻게 동작하는지를 알아낼 방법이 없으므로, 자신이 할당 자체를 되돌릴 수는 없다. 그 대신 런타임 시스템은 호출된 `operator new`가 받아들이는 **매개변수의 개수 및 타입의 똑같은 버전**의 `operator delete`를 찾고, 찾아냈으면 그것을 호출한다.

이 경우는
```cpp
void operator delete(void *, std::ostream&) throw();
```
과 같이 똑같은 시그너처를 가진 것이 마련되어 있어야 한다.

이런 형태의 `delete`를 가리켜 위치지정 `delete`라고 한다. 그런데 *[예제52-1]* 에서 `delete`의 위치지정 버전이 마련되어 있지 않기 때문에 결국 아무것도 하지 않는다. 즉 어떤 `operator delete`도 호출되지 않는다.

**추가 매개변수를 취하는 `operator new` 함수가 있는데 그것과 똑같은 추가 매개변수를 받는 `operator delete`가 짝으로 존재하지 않으면, 이 `new`에 해당 매개변수를 넘겨서 할당한 메모리를 해제해야 하는 상황이 오더라도 어떤 `operator delete`도 호출되지 않는다.**

```cpp
class Widget {
public:
    ...
    static void* operator new(std::size_t size, 
                              std::ostream logStream)                 
        throw(std::bad_alloc);
    static void operator delete(void *pMemory) throw();
    static void operator delete(void *pMemory, 
                                std::ostream& logStream) throw();
    ...
};
```

이렇게 해두면 아래의 문장이 실행되다가 `Widget` 생성자에서 예외가 발생되더라도
```cpp
// 이전과 같은 사용자 코드, 메모리 누출 없음
Widget *pw = new (std::cerr) Widget;
```
이젠느 위치지정 `new`와 짝이 되는 위치지정 `delete`가 (런타임 시스템에 의해) 자동으로 호출된다.

**위의 문장에서 `Widget` 생성자가 예외를 던지지 않았고, 사용자 코드의 `delete` 문까지 다다랐다고 하면 어떤 일이 생길까?**

```cpp
// 기본형의 operator delete가 호출된다.
delete pw;
```
이 경우에는 런타임 시스템이 기본형의 `operator delete`를 호출한다. 위치지정 버전을 호출하지 않는다. **위치지정 `delete`가 호출되는 경우는 위치지정 `new`의 호출에 '묻어서' 함께 호출도는 생성자에서 예외가 발생할 때 뿐이다. 그러니까, 포인터(위의 `pw`와 같은)에 `delete`를 적용했을 때는 절대로 위치지정 `delete`를 호출하는 쪽으로 가지 않는다.**

정리하면 이렇다

어떤 위치지정 `new` 함수와 조금이라도 연관된 모든 메모리 누출을 사전에 봉쇄하려면, 표준 형태의 `operator delete`를 기본으로 마련해 두어야(객체 생성 도중에 예외가 던져지지 않았을 경우 대비) 하고 그와 함께 위치지정 `new`와 똑같은 추가 매개변수를 받는 위치지정 `delete`도 빼먹지 많아야(예외가 던져졌을 때를 대비해서) 한다.

단 바깥쪽 유효범위에 있는 어떤 함수의 이름과 클래스 멤버 함수의 이름이 같으면 바깥쪽 유효범위의 함수가 '이름만 같아도' 가려지게 된다. 

예를 들어 달랑 위치지정 `new`만 선언된 기본 클래스가 버젓이 사용자에게 제공될 경우, 사용자 쪽에서는 표준 형태의 `new`를 써 보려다가 안 되는 것을 발견할 것이다.

```cpp
class Base {
public:
    ...
    /// 이 new가 표준 형태의 전역 new를 가린다.
    static void* operator new(std::size_t size, std::ostream& logStream)
        throw(std::bad_alloc);
    ...
};
// 에러! 표준 형태의 전역 operator new가 가려진다.
Base *pb = new Base;
// 문제 없음! 문제 없이 Base의 위치지정 new를 호출한다.
Base *pb = new (std::cerr) Base;
```
파생 클래스의 경우 전역 `operator new`는 물론이고 자신이 상속받은 기본 클래스의 `operator new`까지 가려 버린다.

```cpp
// 위의 Base로부터 상속받은 클래스
class Derived: public Base {
public:
    ...
    // 기본형 new를 클래스 전용으로 다시 선언한다.
    static void* operator new(std::size_t size)
        throw(std::bad_alloc);
    ...
};
// 에러! Base의 위치지정 new가 가려져 있기 때문이다.
Derived *pd = new (std::clog) Derived;
// 문제 없음! Derived의 operator new를 호출한다.
Derived *pd = new Derived;
```

기본적으로 `C++`가 전역 유효범위에서 제공하는 `operator new`의 형태는 다음 세 가지가 표준이다
```cpp
// 기본형 new
void* operator new(std::size_t) throw(std::bad_alloc);
// 위치지정 new
void* operaotr new(std::size_t, void*) throw();
// 예외불가 new
void* operator new(std::size_t, const std::nothrow_t&) throw();
```
어떤 형태이든 간에 `operator new`가 클래스 안에 선언되는 순간, 위의 표준 형태들이 가려진다. 표준 형태를 막는 것이 원래 의도가 아니라면, 사용자 정의 `operator new`외에 표준 형태들도 사용자가 접근할 수 있도록 해준다. 물론 `opeartor new`를 만들었다면 `operator delete`도 만들어주는 것도 잊지 않는다.

클래스의 울타리 안에서 이런 저런 할당, 해제 함수들이 여느때와 똑같은 방식으로 동작했으면 하는 경우에는, 그냥 클래스 전용 버전이 전역 버전을 호출하도록 구현한다.

이것을 쉽게 하고 싶다면 기본 클래스 하나를 만들고, 이 안에 `new` 및 `delete`의 기본 형태를 전부 넣어준다.
```cpp
class StandardNewDeleteForms 
{
public: 
  // 기본형 new/delete
  static void* operator new(std::size_t size) throw(std::bad_alloc)
  { return ::operator new(size); }
  
  static void operator delete(void* pMemory) throw()
  { ::operator delete(pMemory); }
  
  //위치지정 new/delete
  static void* operator new(std::size_t size, void* ptr) throw()
  { return ::operator new(size, ptr); }

  static void operator delete(void* pMemory, void* ptr) throw()
  { ::operator delete(pMemory, ptr); }
  
  //예외불가 new/delete
  static void* operator new(std::size_t size, const std::nothrow_t& nt) throw() 
  { return ::operator new(size, nt); }
  static void operator delete(void* pMemory, const std::nothrow_t& nt) throw()
  { ::operator delete(pMemory); }
};
```

표준 형태에 덧붙여 사용자 정의 형태를 추가하고 싶다면 이 기본 클래스를 축으로 넓혀가면 된다. 상속과 `using` 선언을 사용해서 표준 형태를 파생 클래스 쪽으로 끌어와 외부에서 사용할 수 있게 만든 후에, 원하는 사용자 정의 형태를 선언한다.

```cpp
// 표준 형태를 물려 받는다.
class Widget : public StandardNewDeleteForms 
{
public: 
    // 표준 형태가 (Widget 내부에) 보이도록 만든다.
    using StandardNewDeleteForms::operator new; 
    using StandardNewDeleteForms::operator delete;
    // 사용자 정의 위치지정 new를 추가한다.
    static void* operator new(std::size_t size, std::ostream& logStream)
        throw(std::bad_alloc);
    // 앞의 것과 짝이 되는 위치지정 delete를 추가한다.
    static void operator delete(void* pMemory, std::ostream& logStream)
        throw();
};
```