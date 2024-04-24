---
layout: single
title: "파이썬 헷갈리는 것 정리"
date: 2024-04-24 19:49:37
lastmod : 2024-04-24 19:49:37
categories: python
tag: [python]
use_math: false
---


```python
class Test:

    step: int = 0

    def fun1(self):
        self.step += 1

    @classmethod
    def fun2(cls):
        cls.step += 2


if __name__ == "__main__":
    test1 = Test()
    print(id(Test.step) == id(test1.step))  # True
```

파이썬의 인스턴스는 변수 참조시 인스턴스 변수에 없으면 클래스 변수를 참조한다.

```python
class Test:

    step: int = 0

    def fun1(self):
        self.step += 1

    @classmethod
    def fun2(cls):
        cls.step += 2


if __name__ == "__main__":
    # None : 인스턴스 변수가 없고 클래스 변수를 참조한다는 뜻
    # [Test.step, test1.step, test2.step]
    # [0, None, None]
    test1 = Test()  # -> [0, None, None]
    test2 = Test()  # -> [0, None, None]
    Test.fun2()  # -> [2, None, None]
    test1.fun1()  # -> [2, 3, None]
    test1.fun2()  # -> [4, 3, None]
    test2.fun2()  # -> [6, 3, None]
    Test.fun2()  # -> [8, 3, None]
    test2.fun1()  # -> [8, 3, 9]
    test2.fun1()  # -> [8, 3, 10]
    test2.fun2()  # -> [10, 3, 10]
    test1.fun1()  # -> [10, 4, 10]
```

* 각 인스턴스의 `step`은 클래스 변수를 참조하다가 `self.step += 1` 같은 코드가 나오면 해당 인스턴스의 인스턴스 변수를 만들고 값을 할당한다.
  1. `self.step = self.step + 1` 이므로 `self.step + 1`에서는 클래스 변수를 참조한다.
  2. `self.step = ...` 에서 인스턴스 변수를 새로 선언하고 값을 저장한다.