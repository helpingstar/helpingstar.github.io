---
layout: single
title: "new 및 delete를 작성할 때 따라야 할 기존의 관례를 잘 알아 두자"
date: 2022-02-14 13:59:47
lastmod : 2022-02-14 15:14:40
categories: EffectiveCpp
tag: [cpp, c++, new, delete]
toc: true
toc_sticky: true
---

# `operator new`
## 반환 값이 제대로 되어 있어야 하고, 가용 메모리가 부족할 경우에는 `new` 처리자 함수를 호출해야 한다.

요청된 메모리를 마련해 줄 수 있으면 그 메모리에 대한 포인터를 반환하는 것으로 끝이다. 메모리를 마련해 줄 수 없는 경우가 문제인데, 이 경우에는 항목 49에서 이야기한 규칙을 따라서 `bad_alloc` 타입의 예외를 던지게 하면 된다. 

간단하지는 않다. `operator new`는 메모리 할당이 실패할 때마다 `new` 처리자 함수를 호출하는 식으로 메모리 할당을 2회 이상 시도하기 때문이다. 즉, 어떤 메모리를 해제하는 데 실마리가 되는 동작을 `new` 처리자 함수 쪽에서 할 수 있을 것으로 가정한다. `operator new`가 예외를 던지게 되는 경우는 오직 `new` 처리자 함수에 대한 포인터기 `null`일 때뿐이다.

## 크기가 없는(0 바이트) 메모리 요청에 대한 대비책을 갖춰 두어야 한다.

0바이트가 요구되었을 때조차도 `operator new` 함수는 적법한 포인터를 반환해야 한다. 
```cpp
// 비멤버 버전의 operator new 함수 의사코드

// 다른 매개변수를 추가로 가질 수 있다.
void * operator new(std::size_t) throw(std::bac_alloc) 
{
    using namespace std;
    if (size == 0) {
        // 0 바이트 요청이 들어오면 1 바이트 요구로 간주하고 처리
        size = 1; 
    }

    while(true) {
        size 바이트를 할당한다.
        if (할당 성공시)
          return (할당된 메모리에 대한 포인터);

        // 할당 실패시 현재 new 처리자 함수가 어느 것으로 설정되어 있는지 찾는다.
        new_handler globalHandler = set_new_handler(0);
        set_new_handler(globalHandler);

        if(globalHandler) (*globalHandler)();
        else throw std::bac_alloc();
  }
}
```

### **`operator new` 함수의 무한 루프**

루프를 빠져나오는 조건
* 메모리 할당이 성공한다.
* 가용 메모리를 늘린다.
* 다른 `new` 처리자를 설치한다
* `new` 처리자의 설치를 제거한다
* `bad_alloc` 혹은 `bad_alloc`에서 파생된 타입의 예외를 던진다.
* 아예 함수 복귀를 포기하고 도중 중단을 시킨다.

### **할당을 시도하는 메모리의 크기가 `size`바이트로 되어 있다.**
`operator new`멤버 함수는 파생 클래스 쪽으로 상속이 되는 함수이다. 그러나 상속으로 인하여 파생 클래스 객체를 담을 메모리를 할당하는 데 기본 클래스의 `operator new` 함수가 호출될 수 있다.

```cpp
class Base
{
public:
    static void * operator new(std::size_t size) throw(std::bad_alloc);
    ...
};
// Derived에서는 `operator new가 선언되지 않았다.
class Derived : public Base {...};
// Base::operator new가 호출된다.
derived * p = new Derived;
```

이에 대한 해결 방법은 "틀린" 메모리 크기가 들어왔을 때를 시작부분에서 확인한 후에 표준 `operator new`를 호출하는 쪽으로 살작 비껴가게 하여 해결할 수 있다.

```cpp
void * Base::operator new (std::size_t size) throw(std::bad_alloc)
{
    if (size != sizeof(Base))
    {
      // 틀린 크기가 들어오면 표준 operator new 쪽에서 메모리 할당 요구를 처리
      return ::operator new(size); 
    }

    // 맞는 크기가 들어오면 메모리 할당 요구를 여기서 처리한다.
}
```
C++에는 모든 독립구조(freestanding)의 객체는 반드시 크기가 0이 넘어야 한다는 금기사항이 있다. 그 덕분에 `sizeof(Base)` 가 0이 될 일은 없다. 따라서 `size`가 0이면 `if` 문이 거짓이 되어 메모리 처리 요구가 `::operator new` 쪽으로 넘어간다.

배열에 대한 메모리 할당을 클래스 전용 방식으로 하고 싶다면 `operator new[]`를 구현하면 된다. `operator new[]` 안에서 해줄 일은 단순히 원시 메모리 덩어리를 할당하는 것 밖에 없다. 이 시점에서는 배열 메모리에 아직 생기지도 않은 클래스 객체에 대해서 아무것도 할 수 없다. 배열 안에 몇 개의 객체가 들어갈지 계산하는 것조차도 안된다.
1. 객체 하나가 얼마나 큰지를 확정할 방법이 없다. 상속때문에 파생 클래스 객체의 배열을 할당하는 데 기본 클래스 `operator new[]` 함수가 호출될 수 있다.
파생 클래스 객체는 대체적으로 기본 클래스 객체보다 더 크다는 것도 문제이다.
그렇기 때문에 `Base::operator new[]` 안에서조차도 배열에 대해 들어가는 객체 하나의 크기가 `sizeof(Base)`라는 가정을 할 수 없다.
2. `operator new[]` 에 넘어가는 `size_t` 타입의 인자는 객체들을 담기에 딱 맞는 메모리 양보다 더 많게 설정되어 있을 수도 있다.

## 실수로 "기본(normal)" 형태의 `new`가 가려지지 않도록 한다.


# `operator delete`
`C++`는 널 포인터에 대한 `delete` 적용이 항상 안전하도록 보장한다는 사실을 잊지 말자. 우리가 할 일은 이 보장을 유지하는 것이다.

```cpp
// 비멤버 버전 operaotr delete의 의사코드

void operator delete(void* rawMemory) throw() {
    //rawMemory가 delete되려고 할 경우 아무것도 하지 않게 한다.
	if (rawMemory == 0) 
		return;

	rawMemory가 가리키는 메모리 해제
}
```

`operator delete`의 클래스 전용 버전도 단순하기는 매한가지이다. 삭제될 메모리의 크기를 점검하는 코드를 넣어 주면 된다. 클래스 전용의 `operator new`가 "틀린" 크기의 메모리 요청을 `::operator new` 쪽으로 구현되었다고 가정하면, 클래스 전용의 `operator delete` 역시 "틀린 크기로 할당된" 메모리의 삭제 요청을 `:: operaotr delete` 쪽으로 전달하는 식으로 구현하면 된다.
```cpp
// operator delete의 클래스 전용 버전

class Base
{
public:
	static void operator delete(void* rawMemory, std::size_t size) throw(std::bad_alloc);
};

void Base::operator delete(void* rawMemory, std::size_t size) throw(std::bad_alloc)
{
    // NULL 포인터 점검
	if (rawMemory == 0)
		return;

	// 크기가 "틀린" 경우 표준 operator delete가 
    // 메모리 삭제 요청을 맡도록 한다.
	if (size != sizeof(Base))
	{
		::operator delete(rawMemory);
		return;
	}

	rawMemroy가 가르키는 메모리 해제

	return;
}
```
가상 소멸자가 없는 기본 클래스로부터 파생된 클래스의 객체를 삭제하려고 할 경우에는 `operaotr delete`로 `C++`가 넘기는 `size_t` 값이 엉터리일 수 있다. 이것만으로도 기본 클래스에 가상 소멸자를 꼭 두어야 하는 충분한 이유가 선다.

기본 클래스에서 가상 소멸자를 빼먹으면 `operator delete` 함수가 똑바로 동작하지 않을 수 있다는 사실만 머리에서 놓치지 말자.