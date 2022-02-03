---
layout: single
title:  "EffectiveCpp 28:내부에서 사용하는 객체에 대한 '핸들'을 반환하는 코드는 되도록 피하자"
categories: EffectiveCpp
tag: [cpp, c++]
toc: true
toc_sticky: true
---

## **ex1)**
사각형을 추상화한 `Rectangle` 클래스를 만들었는데, 이 클래스의 객체를 썼을 때의 메모리 부담을 최대한 줄이고 싶다.

사각형 영역을 정의하는 꼭짓점을 `Rectangle` 자체에 넣으면 안될 것 같고 이것들을 별도의 구조체에 넣은 후에 `Rectangle`이 이 구조체를 가리키도록 하면 어떨까?

```cpp
class Point {
pulibc:
    Point(int x, int y);
    ...
    void setX(int newVal);
    void setY(int newVal);
    ...
};

struct RectData {   // Rectangle에 쓰는 점 데이터
    Point ulhc;     // upper left-hand corner
    Point lrhc;     // lower right-hand corner
};

class Rectangle {
    ...
private:
    std::tr1::shared_ptr<RectData> pData;
};
```
`Point`가 사용자 정의 타입이니 값에 의한 전달보다 참조에 의한 전달 방식을 쓰는 편이 더 효율적이라고 생각한다. 그래서 이들 두 멤버 함수는 (스마트) 포인터로 물어둔 `Point`객체에 대한 참조자를 반환하는 형태로 만든다.
```cpp
class Rectangle {
public:
    ...
    Point& upperLeft() const {return pData->ulhc;}
    Point& lowerRight() const {return pData->lrhc;}
}
```

위의 코드들은 컴파일은 잘 되나 틀렸다. 

`Rectangle`의 꼭짓점 정보를 알아낼 수 있는 방법은 사용자에게 제공하고, `Rectangle` 객체를 수정하는 일은 할 수 없도록 설계했으므로, `upperLeft`, `lowerRight` 함수는 상수 멤버 함수이다.

**하지만** 이 함수들은 `private` 멤버인 내부 데이터에 대한 참조자를 반환한다.

그러니 다음과 같이 쓰면
```cpp
Point coord1(0, 0);
Point coord2(100, 100);

// rec : (0, 0) -> (100, 100) 영역의 Rectangle객체
const Rectangel rec(coord1, coord2);
// 이제 이 rec은 (50, 0) -> (100, 100)의 영역에 있게 된다
rec.upperLeft().setX(50);
```
### **교훈 1**
**클래스 데이터 멤버는 아무리 숨겨봤자 그 멤버의 참조자를 반환하는 함수들의 최대 접근도에 따라 캡슐화 정도가 정해진다.**

`ulhc`와 `lrhc`는 `private`로 선언되어 있으나, 이들의 참조자를 반환하는 `upperLeft` 및 `lowerRight` 함수가 `public` 멤버 함수기 때문에 실직적으로 `public` 멤버이다.

### **교훈 2**
**어떤 객체에서 호출한 상수 멤버 함수의 참조자 반환 값이 실제 데이터가 그 객체의 바깥에 저장되어 있다면, 이 함수의 호출부에서 그 데이터의 수정이 가능하다.**

**ex1)** 의 경우도 있지만 포인터나 반복자를 반환하도록 되어 있다고 해도 마찬가지 이유로 인해 같은 문제가 생긴다. **참조자**, **포인터** 및 **반복자**는 어쨌든 모두 핸들(handle: 다른 객체에 손을 댈 수 있게 하는 매개자)이고, 어떤 객체의 내부요소에 대한 핸들을 반환하게 만들면 언제든지 그 객체의 캡슐화를 무너뜨리는 위험을 무릎쓸 수밖에 없다.

일반적인 수단으로 접근이 불가능한(`protected`, `private`) 멤버 함수도 객체의 내부 요소에 들어간다. 즉, 외부 공개가 차단된 멤버 함수에 대해 이들의 포인터를 반환하는 멤버 함수를 만드는 일이 절대로 없어야 한다.

## **해결방법**

**반환 타입에 `const` 키워드를 붙인다.**
```cpp
class Rectangle {
public:
    ...
    const Point& upperLeft() const {return pData->ulhc;}
    const Point& lowerRight() const {return pData->lrhc;}
    ...
};
```
이렇게 하면 꼭짓점 쌍을 읽을 수는 있지만 쓸 수는 없게 된다. 클래스를 구성하는 요소들을 들여다보도록 하자는 것은 처음부터 설계할 수 있고 이는 의도적인 캡슐화 완화라고 할 수 있다. 더 중요한 부분은 느슨하게 만든 데에도 **제한**을 두었다는 것이다. (읽기 접근O, 쓰기 접근 X)

## **무효 참조 핸들**
핸들이 있기는 하지만 그 핸들을 따라갔을 때 실제 객체의 데이터가 없는 것, 핸들이 물고 있는 객체가 없어지는 현상은 함수가 객체를 값으로 반환할 경우에 가장 흔하게 발생한다

### **ex1)**
```cpp
class GUIObject {...};
// Rectangle 객체를 값으로 반환한다.
const Rectangle boundingBox(const GUIObject& obj);
```

이 상태에서 어떤 사용자가 이 함수를 사용한다 생각해보자
```cpp
// pgo를 써서 임의의 GUIObject를 가리키도록 한다.
GUIObject *pgo;
...
// pgo가 가리키는 GUIObject의 사각 테두리 영역으로부터 
// 좌측 상단 꼭짓점의 포인터를 얻는다
const Point *pUpperLeft = &(boundingBox(*pgo).upperLeft());
```
마지막 문장에서 `boundingBox` 함수를 호출하면 `Rectangle` 임시 객체가 새로 만들어진다. (이 객체를 `temp`라 하자)

다음엔 이 `temp`에 대해 `upperLeft`가 호출될 텐데, 이 호출로 인해 `temp`의 내부 데이터, 정확히 말하면 두 `Point` 객체 중 하나에 대한 참조자가 나온다.

마지막으로 이 참조자에 `&` 연산자를 건 결과 값(주소) 이 `pUpperLeft` 포인터에 대입된다.

이 문장이 끝날 무렵 `boundingBox` 함수의 반환 값(임시 객체`temp`)이 소멸된다.

`temp`가 소멸되니 그 안에 들어있는 `Point` 객체들도 덩달아 없어진다.

그럼 `pUpperLeft` 포인터가 가리키는 객체는 이제 날아가고 없게 된다.

이 문장은 `pUpperLeft`에게 객체를 달아 줬다가 주소 값만 남기고 몽땅 빼앗아 간 것이다.

**객체의 내부에 대한 핸들을 반환하는 함수는 어떻게든 위험하다**

핸들을 반환하는 멤버 함수를 절대로 두지 말라는 말은 아니다

`operator[]` 연산자는 `string`, `vector` 등의 클래스에서 개개의 원소를 참조할 수 있게 만드는 용도로 제공되고 있는데, 실제로 이 연산자는 내부적으로 해당 컨테이너에 들어 있는 개개의 원소 데이터에 대한 참조자를 반환하는 식으로 동작한다. 물론 이 원소 데이터는 컨테이너가 사라질 때 같이 사라지는 데이터이다. 하지만 이런 함수는 예외적인 것이다.