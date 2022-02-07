---
layout: single
title: "EffectiveCpp 44:매개변수에 독립적인 코드는 템플릿으로부터 분리시키자"
date: 2022-02-04 16:01:14
lastmod : 2022-02-07 23:41:23
categories: EffectiveCpp
tag: [cpp, c++]
toc: true
toc_sticky: true
---

아무 생각 없이 템플릿을 사용하면 코드 비대화(code bloat)가 초래될 수 있다. 똑같은 내용의 코드와 데이터가 여러 벌로 중복되어 이진 파일로 구워진다는 뜻이다.

# **템플릿에 의한 이진 코드 비대화 방지 방법**

## **공통성 및 가변성 분석**
지금 만들고 있는 클래스의 어떤 부분이 다른 클래스의 어떤 부분과 똑같다는 사실을 발견한다면, 그 공통 부분을 양쪽에 두지 않는 것이 맞는 코딩이다. 즉 공통 부분을 별도의 새로운 클래스에 옮긴 후, 클래스 상속 혹은 객체 합성을 사용해서 원래의 클래스들이 공통 부분을 공유하도록 한다. 원래의 두 클래스가 제각기 갖고 있는 다른 부분은 원래의 위치에 남아 있게 된다.

템플릿을 작성할 경우에도 똑같이 코드 중복을 막으면 된다.

## **템플릿의 경우**
템플릿이 아닌 코드에서는 코드 중복이 명시적이다. 두 함수 혹은 두 클래스 사이에 똑같은 부분이 있으면 눈으로 찾아낼 수 있다. 반면, 태블릿 코드에서는 코드 중복이 암시적이다. 소스 코드에는 템플릿이 하나밖에 없기 때문에, 어떤 템플릿이 여러 번 인스턴스화될 떄 발생할 수 있는 코드 중복을 감각으로 알아채야 한다.

### **ex1)**
```cpp
template<typename T, std::size_t n>
class SquareMatrix {
public:
    ...
    void invert();
};
```

이 탬플릿은 `T`라는 타입 매개변수도 받지만, `size_t` 타입의 비 타입 매개변수(non-type parameter)인 `n`도 받도록 되어 있다.

다음의 코드를 보자
```cpp
SquareMatrix<double, 5>, sm1;
...
sm1.invert();
SquareMatrix<double, 10> sm2;
...
sm2.invert();
```
`invert`의 사본이 인스턴스화되는데 각각 다른 행렬에 동작할 함수이기 때문에 만들어지는 사본의 개수가 두 개이다.

그렇지만 행과 열의 크기를 나타내는 상수만 빼면 두 함수는 완전히 똑같다. 이런 현상이 템플릿을 포함한 프로그램이 코드 비대화를 일으키는 일반적인 형태이다.

### **별도의 함수를 만든다.**
```cpp
// 정방행렬에 대해 쓸 수 있는 크기에 독립적인 기본 클래스
template<typename T>
class SquareMatrixBase {
protected:
    // 주어진 크기의 행렬을 역행렬로 만든다.
    void invert(std::size_t matrixSize);
};

template<typename T, std::size_t n>
class SquareMatrix: private SquareMatrixBase<T> {
private:
    // 기본 클래스의 invert가 가려지는 것을 막기 위한 문장
    using SquareMatrixBase<T>::invert;
public:
    // invert의 기본 클래스 버전에 대해 인라인 호출 수행
    void invert() {this->invert(n);}
};
```

행렬의 크기를 매개변수로 받도록 바뀐 `invert` 함수가 기본 클래스인 `SquareMatrix`에 들어 있다. 행렬의 원소가 갖는 타입에 대해서만 템플릿화되어 있을 뿐이고 행렬의 크기는 매개변수로 받지 않는다는 것은 `SquareMatrix`와 다르다. 따라서 같은 타입의 객체를 원소로 갖는 모든 정방행렬은 오직 한 가지의 `SquareMatrixBase` 클래스를 공유하게 되는 것이다.

다시 말해, 같은 원소 타입의 정방행렬이 사용하는 기본 클래스 버전의 `invert` 함수도 오직 한 개의 사본이다.

`SquareMatrixBase::invert` 함수는 파생 클래스에서 코드 복제를 피할 목적으로만 마련한 장치이기 때문에, `public` 멤버가 아니라 `protected` 멤버로 되어 있다.

기본 클래스의 `invert` 함수를 호출하도록 구현된 파생 클래스의 `invert` 함수가 바로 인라인 함수이기 때문에 함수의 호출에 드는 추가 비용은 하나도 없어야 한다.

기본 클래스를 사용한 데는 순전히 파생 클래스의 구현을 돕기 위한 것 외에는 아무 이유도 없다는 사실을 드러내는 부분이 `private`키워드이기 떄문에 `SquareMatrix`와 `SquareMatrixBase` 사이의 상속 관계가 `private`이다.

### **메모리 할당 방법의 결정 권한을 파생클래스로 넘김**
`SquareMatrixBase::invert` 함수가 자신이 상대할 데이터가 어떤 것인지 알아야 한다. 이것은 파생 클래스만 알고 있다. 이 정보를 알려주기 위해 정방 행렬의 메모리 위치를 파생 클래스가 기본 클래스로 넘겨준다.
```cpp
template<typename T>
class SquareMatrixBase {
protected:
    // 행렬 크기를 저장하고 행렬 값에 대한 포인터를 저장
    SquareMatrixBase(std::size_t m, T *pMem) : size(n), pData(pMem){}

    // pData에 다시 대입
    void setDataPtr(T * ptr) {pData = ptr;}
private:
    // 행렬의 크기
    std::size_t size;
    
    // 행렬 값에 대한 포인터
    T *pData;
};

template<typename T, std::size_t n>
class SquareMatrix: private SquareMatrixBase<T> {
public:
    // 행렬의 크기 및 데이터 포인터를 기본 클래스로 올려 보낸다.
    SquareMatrix() : SquareMatrixBase<T>(n, data) {}
private:
    T data[n*n];
}
```
이렇게 설계해 두면, 메모리 할당 방법의 결정 권한이 파생 클래스 쪽으로 넘어가게 된다.
이렇게 파생클래스를 만들면 동적 메모리 할당이 필요 없는 객체가 되지만, 객체 자체의 크기가 좀 커질 수 있다. 

### **데이터를 힙에 둔다.**
```cpp
template<typename T, std::size_t n>
class SquareMatrix: private SquareMatrixBase<T> {
public:
  SquareMatrix() : SquareMatrixBase<T>(n, 0), pData(new T[n*n]) {
    this->setDataPtr(pdData.get());
  }
private:
  boost::scoped_arrayt<T> pData;
};
```
`SquareMatrix`에 속해 있는 멤버 함수 중 상당수가 기본 클래스 버전을 호출하는 단순 인라인 함수가 될 수 있다.

똑같은 타입의 데이터를 원소로 갖는 모든 정방행렬들이 행렬 크기에 상관없이 이 기본 클래스 버전의 사본 하나를 공유한다. 이와 동시에 행렬 크기가 다른 `SquareMatrix` 객체는 저마다 고유의 타입을 갖고 있다. 

ex)

`SqaureMatrix<double, 5>` 객체와 `SquareMatrix<douboe, 10>` 객체가 똑같이 `SquareMatrixBase<double>` 클래스의 멤버 함수를 갖고 있다 하더라도 타입이 다르기 떄문에 다른 타입의 함수 호출을 컴파일러가 막아준다.

1. 행렬 크기가 미리 녹아든 상태로 별도의 버전이 만들어지는 `invert` (크기별 고정 버전)
2. 행렬 크기가 함수 매개변수로 넘여지거나 객체에 저장된 형태로 다른 파생 클래스들이 공유하는 버전의 `invert`함수. (크기 독립형 버전)

**크기별 고정 버전의 경우**, 행렬 크기가 컴파일 시점에 투입되는 상수이기 때문에 상수 전파(constant propagation) 등의 최적화가 먹혀 들어가기 좋다. 생성되는 기계 명령어에 대해 이 크기 값이 즉치 피연산자(immediate operand)로 바로 박히는 것도 이런 종류의 최적화 중 하나이다. 이는 크기 독립형 버전(후자)에서는 얻어낼 수 없다.

**한가지 버전의 `invert`를 두는 버전의 경우**, 

**코드의 크기**


여러 행렬 크기에 대해 한 가지 버전의 `invert`를 두도록 만들면 실행 코드의 크기가 작아진다 이는 **실행 코드 크기 작아짐 -> 프로그램의 작업 세트 크기 감소, 명령어 캐시 내의 참조 지역성 향상 -> 실행속도 향상**을 일으키는데 이는 크기별 고정 버전의 최적화 효과를 얻지 못한 것에 대해 보상을 얻고도 남을 수 있다.

**객체의 크기**

`invert` 비슷한 크기의 독립형 버전의 함수를 기본 클래스 쪽으로 아무 생각 없이 옮겨 놓다 보면. 객체의 전체 크기가 슬그머니 늘어날 수 있다. `SquareMatrix` 객체는 메모리에 생길 때마다 파생 클래스 자체에 이미 이 데이터에 접근할 수 있는 수단이 있는데도 불구하고 `SquareMatrixBase` 클래스에 들어 있는 데이터를 가리키는 포인터를 하나씩 떠안고 있다. 이것은 포인터 하나 크기만큼 낭비하게 된다. 이런 포인터를 제거할 수 있도록 설계할 수는 있지만 보통 골치아픈 일이 아니다.

## **타입 매개변수에 의한 비대화**

`int`, `long`의 경우 상당수의 플랫폼에서 이진 표현 구조가 동일하기 때문에 `vector<int>`, `vector<long>`의 멤버 함수는 똑같게 나올 수 있다. 

**포인터 타입**의 경우도 같다. 상당수의 플랫폼에서 포인터 타입은 똑같은 이진 표현구조를 갖고 있기 때문에 `list<int*>`, `list<const int*>`, `list<SquareMatrix<long, 3>*>` 은 이진 수준에서만 보면 멤버 함수 집합을 달랑 한번만 써도 되어야 한다. 이 말을 기술적으로 풀어 보면, 타입 제약이 엄격한 포인터를 써서 동작하는 멤버 함수를 구현할 때는 하단에서 타입미정(untyped) 포인터(즉, `void*` 포인터)로 동작하는 버전을 호출하는 식으로 만든다.