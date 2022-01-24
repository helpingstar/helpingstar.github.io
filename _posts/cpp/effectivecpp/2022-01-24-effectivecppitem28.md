---
layout: single
title:  "EffectiveCpp 28:내부에서 사용하는 객체에 대한 '핸들'을 반환하는 코드는 되도록 피하자"
categories: EffectiveCpp
tag: [cpp, c++]
toc: true
---
# **항목 27: 내부에서 사용하는 객체에 대한 '핸들'을 반환하는 코드는 되도록 피하자**

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
    struct RectData {   // Rectangle에 쓰는 점 데이터
        Point ulhc;     // upper left-hand corner
        Point lrhc;     // lower right-hand corner
    };
    class Rectangle {
        ...
    private:
        std::tr1::shared_ptr<RectData> pData;
    }
}
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

## **ex2)**
**ex1)** 의 경우도 있지만 포인터나 반복자를 반환하도록 되어 있다고 해도 마찬가지 이유로 인해 같은 문제가 생긴다. **참조자**, **포인터** 및 **반복자**는 어쨌든 모두 핸들(handle: 다른 객체에 손을 댈 수 있게 하는 매개자)이고, 어떤 객체의 내부요소에 대한 핸들을 반환하게 만들면 언제든지 그 객체의 캡슐화를 무너뜨리는 위험을 무릎쓸 수밖에 없다.