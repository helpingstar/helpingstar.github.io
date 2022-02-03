---
layout: single
title:  "EffectiveCpp 38:has-a 혹은 is-implemented-in-terms-of 를 모형화할 때는 객체 합성을 사용하자"
categories: EffectiveCpp
tag: [cpp, c++]
toc: true
---


## **합성**
**합성**이란, 어떤 타입의 객체들이 그와 다른 타입의 객체들을 포함하고 있을 경우에 성립하는 그 타입들 사이의 관계를 말한다.

이를테면 다음과 같다.
```cpp
// [코드 38-1]
class Address {...};
class PhoneNumber {...};
class Person {
public:
    ...
private:
    std::string name;
    Address address;
    PhoneNumber voiceNumber;
    PhoneNumber faxNumber;
}
```
### **합성의 의미**
`public` 상속의 의미가 `is-a`(~는 ~의 일종이다) 라고 공부했다.

합성 역시 의미를 갖고 있는데

`has-a` (~는 ~를 가짐) 을 뜻할 수도 있고 `is-implemented-in-terms-of`(~는 ~를 써서 구현됨)을 뜻할 수도 있다.

뜻이 두 개인 이유는 소프트웨어 개발에서 대하는 영역(domain)이 두 가지이기 때문이다.

객체 중에는 일상생활에서 볼 수 있는 사물을 본 뜬 것들이 있는데 (ex. 사람, 이동수단, 비디오 프레임) 이런 객체는 **소프트웨어의 응용 영역(application domain)** 에 속한다. 객체 합성이 응용 영역의 객체들 사이에서 일어나면 `has-a` 관계이다.

응용 영역에 속하지 않는 나머지들은 버퍼, 뮤텍스, 탐색 트리 등 순수하게 시스템 구현만을 위한 인공물이다. 이런 종류의 객체가 속한 부분은 **소프트웨어의 구현 영역(implementation domain)** 이라고 한다. 객체 합성이 구현 영역에서 일어나면 `is-implemented-in-terms-of` 관계를 나타낸다.

[코드 38-1]의 경우는 `has-a`관계이다. (Person has a name...)

## `is-a` vs `is-implemented-in-terms-of`

### **`is-a`**
중복 원소가 없는 집합체를 나타내고 저장 공간도 적게 차지하는 클래스의 템플릿이 하나 필요하다고 가정하자.

표준 라이브러리의 `set` 템플릿이 적당하지 않아 새로 만드려고 한다.

연결 리스트를 활용하기 위해 표준 라이브러리의 `list` 템플릿을 
재사용하려고 한다. 그리하여 `Set<T>` 는 `list<T>` 로부터 상속을 받는다. 실제로 `Set` 객체는 `list`의 일종(`is-a`)이 되고 다음 코드를 선언한다.
```cpp
template<typename T> // Set을 만든답시고 list를 잘못쓰는 방법
class Set: public std::list<T> {...};
```
하지만 이 코드는 틀렸다, `D`, `B` 사이에 `is-a` 관계가 성립하면 `B`에서 참인 것들이 전부 `D`에서도 참이어야 한다. 하지만 `list` 객체는 중복 원소를 가질 수 있지만 `Set` 객체는 원소가 중복되면 안된다.

따라서 `Set`과 `list`는 `is-a` 관계가 아니다.
### **`is-implemented-in-terms-of`**
`Set` 객체는 `list` 객체를 써서 구현되는(`is implemented in terms of`) 형태의 설계가 가능하다.

```cpp
template<class T>   // Set을 만드는 데 list를 제대로 쓰는 방법
class Set {
public:
    bool member(const T& item) const;
    void insert(const T& item);
    void remove(const T& item);
    std::size_t size() const;
private:
    std::list<T> rep;   // Set 데이터의 내부 표현부
}
```
```cpp
template<typename T>
bool Set<T>::member(const T& item) const
{
    return std::find(rep.begin(), rep.end(), item) != rep.end();
}
 
template<typename T>
void Set<T>::insert(const T& item)
{
    if (!member(item)) rep.push_back(item);
}
 
template<typename T>
void Set<T>::remove(const T& item)
{
    typename std::list<T>::iterator it =
        std::find(rep.begin(), rep.end(), item);
 
    if (it !+ rep.end()) rep.erase(it);
}
 
template<typename T>
std::size_t Set<T>::size() const
{
    return rep.size();
}
```



*Scott Meyers, 『Effective C++』, 곽용재 옮김, 프로텍 미디어(2015), p275-278*