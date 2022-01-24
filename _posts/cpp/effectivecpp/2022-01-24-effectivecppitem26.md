---
layout: single
title:  "EffectiveCpp item 26"
categories: EffectiveCpp
tag: [cpp, c++]
toc: true
---
# **항목 26: 변수 정의는 늦출 수 있는 데까지 늦추는 근성을 발휘하자**
생성자 혹은 소멸자를 끌고 다니는 타입으로 변수를 정의하면 반드시 물게 되는 비용

1. 프로그램 제어 흐름이 변수의 정의에 닿을 때 생성자가 호출되는 비용
2. 그 변수가 유효범위를 벗어날 때 소멸자가 호출되는 비용
## **ex1)**

```cpp
// 이 함수는 "encrypted" 변수를 너무 일찍 정의해 버립니다.
std::string encryptPassword(const std:: string& password)
{
    using namespace std;
    string encrypted;
    if (password.length() < MinimumPasswordLength) {
        throw logic_error("Password is too short");
    }
    ...
    return encrypted;
}
```
예외가 발생하면 `enrypted` 객체는 사용하지 않게 된다. 그리하여 `encrypted` 변수를 정의하는 일은 꼭 필요해지기 전까지로 미루는 편이 낫다는 생각이 들 것이다.

```cpp
// 이 함수는 encrypted 변수가 진짜로 필요해질 때까지 정의를 미룬다.
std::string encryptPassword(const std:: string& password)
{
    using namespace std;
    if (password.length() < MinimumPasswordLength) {
        throw logic_error("Password is too short");
    }
    string encrypted;
    ...
    return encrypted;
}
```

## **ex2)**

객체를 기본 생성하고 나서 값을 대입하는 방법은 원하는 값으로 직접 초기화하는 방법보다 효율이 좋지 않다.

```cpp
void encrypt(std::string& s); // 이 자리에서 s를 바로 암호화한다.
```

```cpp
// 이 함수는 encrypted 변수가 필요해질 때까지 이 변수의 정의를
// 늦추긴 했지만, 여전히 쓸데없이 비효율적입니다.
std::string encryptPassword(const std::string& password)
{
    ...                     // 길이를 점검하는 부분은 똑같으므로 생략
    std::string encrypted   // 기본 생성자에 의해 만들어지는 encrypted
    encrypted = password;   // encrypted에 password를 대입

    encrypt(encrypted);
    return encrypted;
}
```
비용이 만만치 않은 기본 생성자 호출을 건너뛰어야 하기 위해 `encrypted`를 `password`로 초기화한다.

```cpp
// encrypted를 정의하고 초기화하는 가장 좋은 방법
std::string encryptPassword(const std::string& password)
{
    ...
    std::string encrypted(password);    // 변수를 정의함과 동시에 초기화
                                        // 이때 복사 생성자가 쓰인다
    encrypt(encrypted);
    return encrypted;
}
```
## **ex3)**

루프에 대해서는 어떨까, 변수가 루프 안에서만 쓰이는 경우
```cpp
// 방법 A : 루프 바깥쪽에 정의
Widget w;
for (int i = 0; i < n; ++i) {
    w = i;   // i에 따라 달라지는 값
    ...
}
```
```cpp
// 방법 B : 루프 안쪽에 정의
for (int i = 0; i < n; ++i) {
    Widget w(i); // i에 따라 달라지는 값
    ...
}
```

**A 방법**

해당 변수를 루프 바깥에서 미리 정의함

생성자 1번 + 소멸자 1번 + 대입 n번

 - 대입에 들어가는 비용이 생성자-소멸자 쌍보다 적게 나오는 경우 더 좋다
 - `w` 이름의 유효범위가 [B 방법]보다 넓어져 프로그램의 이해도와 유지보수성이 안좋아질 수 있다.

**B 방법**

루프 안에서 변수를 정의함

생성자 n번 + 소멸자 n번

- 대입이 생성자-소멸자 쌍보다 비용이 덜 들고, 전체 코드에서 수행 성능에 민감한 부분을 건드리는 중이라고 생각하지 않는다면 [B 방법] 이 더 좋다.


## 정리

어떤 변수를 사용해야 할 때가 오기 전까지 변수의 정의를 늦춘다.

초기화 인자를 손에 넣기 전까지 정의를 늦출 수 있는지도 둘러본다.

이렇게 해야 쓰지도 않을 객체가 만들어졌다 없어지는 일이 생기지 않으며, 불필요한 기본 생성자 호출도 일어나지 않는다.

덤으로 누가 보아도 그 변수의 의미가 명확한 상황에서 초기화가 이루어지기 때문에 변수의 쓰임새를 문서화하는데도 도움이 된다.


*Scott Meyers, 『Effective C++』, 곽용재 옮김, 프로텍 미디어(2015), p183-187*