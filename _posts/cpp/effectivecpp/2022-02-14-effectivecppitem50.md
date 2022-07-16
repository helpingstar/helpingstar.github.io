---
layout: single
title: "EffectiveCpp 50:new 및 delete를 언제 바꿔야 좋은 소리를 들을지를 파악해 두자"
date: 2022-02-14 13:59:47
lastmod : 2022-02-14 13:59:52
categories: EffectiveCpp
tag: [cpp, c++, new, delete]
toc: true
toc_sticky: true
---
# `new` 및 `delete`의 기본 제공 버전을 다른 것으로 대체하는 작업을 언제 해야 의미가 있는가

## 잘못된 힙 사용을 탐지하기 위해
1. `new`한 메모리에 `delete`를 하는 것을 잃어버리면 메모리가 누출되며, 한 번 `new`한 메모리르 두 번 이상 `delete`하면 미정의 동작이 발생하고 만다. 만일 할당된 메모리 주소의 목록을 `operator new`가 유지해 두고 `operator delete`가 그 목록으로부터 주소를 하나씩 제거해 주게 만들어져 있다면 이런 식의 실수는 쉽게 잡아낼 수 있다.

2. 프로그래밍을 하다가 실수를 하다 보면 데이터 오버런(overrun, 할당된 메모리 블록의 끝을 넘어 뒤에 기록하는 것) 및 언더런(underrun, 할당된 메모리 블록의 시작을 넘어 앞에 기록하는 것)이 발생할 수 있다. 이런 경우 사용자 정의 `operator new`를활용한다면, 요구된 크기보다 약간 더 메모리를 할당한 후에 사용자가 실제로 사용할 메모리의 앞과 뒤에 오버런/언더런 탐지용 바이트 패턴(경계표지, signature)을 적어두도록 만들 수 있다. `operator delete`는 누군가가 이 경계표지에 손을 댔는지 안 댔는지 점검하도록 만든다. 만일 이 경계표지부분에 원래와 다른 정보가 적혀 있다면 할당된 메모리 블록을 사용하는 도중에 오버런/언더런이 발생한 것이므로, `operator delte`는 이 사실을 로그로 기록함으로써 문제를 일으킨 포인터 값을 남겨놓을 수 있다.

## 효율을 향상시키기 위해
컴파일러가 제공하는 기본 버전의 `operator new` 및 `operator delete` 함수는 대체적으로 일반적인 쓰임새에 맞추어 설계되었다. 즉 여러 가지 할당 유형도 소화할 수 있어야 하며 힙 단편화(fragmentation)에 대한 대처방안도 없으면 안 된다. 만일 개발자가 자신의 프로그램이 동적 메모리를 어떤 성향으로 사용하는지를 제대로 이해하고 있다면, 사용자 정의 `operator new`와 `operator delete`를 자신이 만들어서 쓰는 편이 기본제공 버전을 썼을 때보다 더 우수한 성능(빠른 실행속도, 적은 메모리 차지)을 낼 확률이 높다.

## 동적 할당 메모리의 실제 사용에 관한 정보를 수집하기 위해
사용자 정의 `operator new` 및 `operator delete`를 사용하면 메모리 블록의 크기, 할당 해제 순서, 사용패턴 동적 할당 메모리의 최대량 등의 정보를 아주 쉽게 수집할 수 있다.

### 오버런 및 언더런을 탐지하기 쉬운 형태로 만들어 주는 전역 `operator new`
```cpp
static const int signature = 0xDEADBEEF;
typedef unsigned char Byte;
 
// 이 코드는 고쳐야 할 부분이 몇 개 있습니다.
void* operator new(std::size_t size) throw(std::bad_alloc)
{
    using namespace std;
    // 경계표지 2개를 앞뒤에 붙일 수 있을만큼만 메모리 크기를 늘립니다
    size_t realSize = size + 2 * sizeof(int);  

    // malloc을 호출하여 실제 메모리를 얻어냅니다.
    void *pMem = malloc(realSize);             
    if (!pMem) throw bad_alloc();             
 
    // 메모리 블록의 시작 및 끝부분에 경계표지를 기록합니다.
    *(static_cast<int*>(pMem)) = signature;
    *(reinterpret_cast<int*>(static_cast<Byte*>(pMem)+realSize-
        sizeof(int))) = signature;
 
    // 앞쪽 경계표지 바로 다음의 메모리를 가리키는 포인터를 반환합니다.
    return static_cast<Byte*>(pMem) + sizeof(int);
}
```

위 예시의 틀린 점
* `operator new`에는 `new` 처리자를 호출하는 루프가 반드시 들어 있어야 하는데 없다.
* 바이트 정렬(alignment) 

아키텍처적으로 특정 타입의 데이터가 특정 종류의 메모리 주소를 시작으로 하여 저장될 것을 요구사항으로 두고 있다. 포인터는 4바이트 단위로 정렬되거나 `double`은 8바이트 단위로 정렬되어야 한다.

모든 `operator new` 함수는 어떤 데이터 타입에도 바이트 정렬을 만족하는 포인터를 반환해야 한다는 것이 `C++`의 요구사항이다. 표준 `malloc` 함수는 이를 만족한다.

하지만 예시의 `operator new` 에서는 `malloc`에서 나온 포인터를 반환하지 않는다. 이런 경우 안전하다는 보장을 할 수가 없다.

**꼭 만들어 쓸 이유가 없다면 굳이 들이댈 필요는 없다.**

`Boost`의 `Pool` 라이브러리에서 제공하는 메모리 할당자는 사용자 정의 메모리 관리 루틴으로 도움을 얻을 수 있는 가장 흔한 경우들 중 하나에 맞추어 튜닝되어 있는데, 크기가 작은 객체(소형 객체)를 많이 할당할 경우이다.

# `new` 및 `delete`를 "언제" 대체하는 것인가
## 잘못된 힙 사용을 탐지하기 위해
## 동적 할당 메모리의 실제 사용에 관한 통계 정보를 수집하기 위해
## 할당 및 해제 속력을 높이기 위해
사용자 정의 버전이 특정 타입의 객체에 맞추어 설계되어 있으면 더 빠른 경우가 많다. `Boost`의 `Pool` 라이브러리에서 자공하는 할당자처럼 고정된 크기의 객체만 만들어주는 할당자의 전형적인 응용 예가 바로 클래스 전용(class-specific) 할당자다. 응용 프로그램이 단일 스레드로 동작하는데, 기본 컴파일러가 제공하는 메모리 관리 루틴이 다중 스레드에 맞게 만들어져 있다면, 스레드 안전성이 없는 할당자를 직접 만들어 씀으로써 상당한 속력 이득을 볼 수 있다.
## 기본 메모리 관리자의 공간 오버헤드를 줄이기 위해
범용 메모리 관리자는 사용자 정의 버전과 비교해서 속력이 느리고 메모리를 많이 잡아먹는 사례가 많다. 할당된 각각의 메모리 블록에 대해 전체적으로 지우는 부담이 꽤 되기 때문이다. 크기가 작은 객체에 대해 튜닝된 할당자(ex `Boost-Pool`)를 사용하면 이러한 오버헤드를 실질적으로 제거할 수 있다.
## 적당히 타협한 기본 할당자의 바이트 정렬 동작을 보장하기 위해
`x86` 아키텍처에서는 `double`이 8바이트 단위로 정렬될 때 `R/W` 속도가 가장 빠르다. 기본 제공 `operaotr new` 대신에 8바이트 정렬을 보장하는 사용자 정의 버전으로 바꿈으로써 성능을 끌어올릴 수 있다.
## 임의의 관계를 맺고 있는 객체들을 한 군데에 나란히 모아 놓기 위해
한 프로그램에서 자료구조 몇 개가 대개 한 번에 동시에 쓰이고 있다는 사실을 알고 있고, 앞으로 이들에 대해서는 페이지 부재(page fault: an exception that the memory management unit (MMU) raises when a process accesses a memory page without proper preparations.) 발생 횟수를 최소화하고 시픙 ㄹ경우, 해당 자료구조를 담을 별도의 힙을 생성하믕로써 이들이 가능한 적은 페이지를 차지하도록 하면 좋은 효과를 볼 수 있다. 이러한 메모리 군집화는 위치지정(placement) `new` 및 위치지정 `delete`를 통해 쉽게 구현할 수 있다.
## 그때그때 원하는 동작을 수행하도록 하기 위해
1. 메모리 할당과 해제를 공유 메모리에다 하고 싶은데 공유 메모리를 조작하는 일은 C API로밖에 할 수 없을 때 사용자 정의 버전을 만드는 것이 좋다.(위치지정 `new`/`delete`가 적당하다) 
2. 응용 프로그램 데이터의 보안 강화를 위해 해제한 메모리 블록에 `0`을 덮어쓰는 사용자 정의를 만드는 경우
