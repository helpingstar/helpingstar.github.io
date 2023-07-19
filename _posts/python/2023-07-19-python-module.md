---
layout: single
title: "파이썬의 모듈과 패키지"
date: 2023-07-19 11:30:17
lastmod : 2023-07-19 11:30:17
categories: python
tag: [python, module, package]
use_math: false
---

쓸 때마다 헷갈려서 정리하는 글이다.

예제 실습에 사용한 코드는 [**python_module_example**](https://github.com/helpingstar/python_module_example) 에서 확인할 수 있다.

## 모듈

**모듈은 파이썬 정의와 문장들을 담고 있는 파일**을 의미한다.

모듈의 이름은 모듈 이름에 확장자 `.py`를 붙인다. 모듈의 이름은 전역 변수 `__name__`으로 제공된다.

```
example1
  add.py
  exec1.py
  exec2.py
  exec3.py
  exec4.py
```

```python
# add.py
"""This is __doc__"""

# __annotations__
v_int: int = 3
v_str: str

def add1(a: int, b: int) -> int:
    return a + b

def add2(a: int, b: int) -> int:
    result = a + b
    return result
```

`import`로 선언한 모듈은 선언한 namespace에서만 유효하다

`import <module>`은 `<module> = __import__("<module>")`과 같다.

```python
# exec1.py
import add

print(add.add1(2, 3))
# 5
print(add.add2(2, 3))
# 5
print(add.__name__)
# add
print(add.__file__)
# ~~\module_example\example1\add.py
print(add.__annotations__)
# {'v_int': <class 'int'>, 'v_str': <class 'str'>}
print(add.__doc__)
# This is __doc__
```

`from <module> import <item>`을 통해 `<item>`으로 바로 모듈의 전역변수에 접근할 수 있다. 그냥 `import`만 했다면 `<module>.<item>` 으로 접근해야 했을 것이다.

```python
# exec2.py
from add import add1, add2, _add3

print(add1(3, 11))
# 14
print(add2(3, 11))
# 14
print(_add3(3, 11))
# 14
```

`from <module> import *`를 통해 밑줄(`_`)로 시작하는 것을 제외한 모든 이름을 임포트할 수 있다. 그러나 최대한 쓰지 않도록 하는 것이 좋다.

```python
# exec3.py
from add import *

print(add1(3, 11))
# 14
print(add2(3, 11))
# 14
# print(_add3(3, 11))
# NameError: name '_add3' is not defined.
```

모듈 이름 다음에 `as`가 올 경우, `as` 다음의 이름을 임포트한 모듈에 직접 연결한다

`import <module> as temp`는 `temp = __import__("<module>")`와 같다.

```python
# exec4.py
import add as add_numbers

print(add_numbers.add1(2, 3))
# 5

from add import add1 as add_numbers

print(add_numbers(2, 3))
# 5
```

파이썬 모듈을 직접 실행하면 모듈의 `__name__`은 `"__main__"`으로 설정된다.

```
example2
  exec.py
  name.py
```

```python
# name.py
print("outside")
if __name__ == "__main__":
    print("inside")
```

(`[output]`아래가 출력된 내용)

```bash
python name.py

# outside
# inside
```

```python
# exec.py
import name
```

```bash
python exec.py

# outside
```

`import mong` 라인이 실행되면 인터프리터는 다음과 같이 실행된다.

1. built-in module에서 `mong`이라는 이름을 찾는다. (`sys.builtin_module_names`: `tuple`)
2. `sys.path`에서 `mong.py`라는 파일을 찾는다. `sys.path`는 다음과 같이 초기화된다. (`sys.path`: `list`)
   * 입력 스크립트를 포함하는 디렉터리 (파일이 지정되지 않았을 때는 현재 디렉터리)
   * `PYTHONPATH` (환경변수)
   * 설치에 따라 달라지는 기본값(`site` 모듈에서 처리하는 `site-packages` 디렉터리를 포함하는 규칙에 따라)

`dir()`**는 모듈이 정의하는 변수, 모듈, 함수 등등의 이름들을 찾는 데 사용된다.** 문자열들의 정렬된 리스트를 돌려준다. 인자가 없으면 현재 정의한 이름들을 나열한다. 내장 함수와 변수들 (ex. `all`, `sum`, `ZeroDivisionError`, ...) 은 표준 모듈 `builtins`에 정의되어 있다.

```
example3
  exec1.py
  funcs.py
```

```python
# funcs.py
var: int
var_real = "real"


def func1() -> None:
    pass


def func2() -> None:
    pass
```

```python
# exec1.py
import sys
import funcs

print(dir(funcs))
# [... 'func1', 'func2', 'var_real']
print(dir(sys))
# ['__breakpointhook__', '__displayhook__', '__doc__', ...]
funcs_exec = 1
print(dir())
# [..., 'funcs', 'funcs_exec', 'sys']
```

## 패키지

**패키지는 "점으로 구분된 모듈 이름"을 써서 파이썬의 모듈 이름 공간을 구조화하는 방법**을 의미한다.

`A.B`는 `A`라는 이름의 패키지에 있는 `B`라는 이름의 서브 모듈을 가리킨다.

```
sound/              Top-level package
  __init__.py       Initialize the sound package
  formats/          Subpackage for file format conversions
    __init__.py
    wavread.py
    wavwrite.py
    aiffread.py
    aiffwrite.py
    auread.py
    auwrite.py
    ...
  effects/          Subpackage for sound effects
    __init__.py
    echo.py
    surround.py
    reverse.py
    ...
  filters/          Subpackage for filters
    __init__.py
    equalizer.py
    vocoder.py
    karaoke.py
    ...
```

`__init__.py` 파일은 파이썬이 해당 파일이 포함된 디렉터리를 패키지로 취급하도록 하는 데 필요하다. 이렇게 하면 `string`과 같은 일반적인 이름을 가진 디렉터리가 모듈 검색 경로에서 나중에 발생하는 유효한 모듈을 의도치 않게 숨기는 것을 방지할 수 있다. 가장 간단한 경우 `__init__.py`는 그냥 빈 파일일 수도 있지만 패키지에 대한 초기화 코드를 실행하거나 나중에 설명하는 `__all__` 변수를 설정할 수도 있다.

다음은 모두 같은 함수를 실행한다.

```python
import sound.effects.echo
sound.effects.echo.echofilter(input, output, delay=0.7, atten=4)

from sound.effects import echo
echo.echofilter(input, output, delay=0.7, atten=4)

from sound.effects.echo import echofilter
echofilter(input, output, delay=0.7, atten=4)
```

`from package import item`을 사용할 때 item은 패키지의 서브 모듈(또는 패키지) 일 수도 있고 함수, 클래스, 변수 등 패키지에 정의된 다른 이름들일 수도 있음에 유의하자. `import` 문은 먼저 item이 패키지에 정의되어 있는지 검사하고, 그렇지 않으면 모듈이라고 가정하고 로드를 시도한다. 찾지 못한다면. `ImportError` 예외를 일으킨다.

`import item.subitem.subsubitem`과 같은 문법을 사용할 때 마지막 것을 제외한 각 항목은 반드시 패키지여야 한다. 마지막 항목은 모듈이나 패키지가 될 수 있지만, 앞의 항목에서 정의된 클래스, 함수, 변수 등이 될 수는 없다.

`from sound.effects import *` 로 `sounds/effects`의 모든 모듈을 임포트할 수 없다. ('from example4.high.middle import *' used; unable to detect undefined names)

이 때 `sounds/effects/__init__.py`에 정의된 `__all__`를 이용할 수 있는데 `import` 문은 패키지의 `__init__.py` 코드가 `__all__` 이라는 이름의 목록을 제공하면 이것을 `from package import *`를 만날 때 임포트 해야만 하는 모듈 이름들의 목록으로 받아들인다.

`__all__` 이 정의되지 않으면, 문장 `from sound.effects import *` 은 패키지 `sound.effects` 의 모든 서브 모듈들을 현재 이름 공간으로 임포트 하지 않는다;

```
example4
  high
    __init__.py
    middle
      class1.py
      class2.py
      class3.py
      __init__.py
```

```python
# high/__init__.py
print("high/__init__.py")

# high/middle/__init__.py
print("high/middle/__init__.py")
__all__ = ["class1", "class2"]

# high/middle/class1.py
print("class 1")

# high/middle/class2.py
print("class 2")

# high/middle/class3.py
print("class 3")
```

아래 실행은 모두 각각 실행한 것이다.

```python
>>> from example4.high.middle import *
high/__init__.py
high/middle/__init__.py
class 1
class 2
```

```python
>>> from example4.high.middle import class2
high/__init__.py
high/middle/__init__.py
class 2
```

```python
>>> from example4.high.middle import class2
high/__init__.py
high/middle/__init__.py
class 2
>>> from example4.high.middle import *
class 1
```

모듈간의 상대적인 경로를 이용하여 임포트 문을 사용할 수도 있다. **상대 경로는 현재의 모듈의 이름(`__name__`)에 기반한다.** 메인 모듈의 이름은 항상 `"__main__"` 이기 때문에, 파이썬 응용 프로그램의 메인 모듈로 사용될 목적의 모듈들은 반드시 절대 임포트를 사용해야 한다.

```
/example5
  /red_folder
    __init__.py
    red.py
    /blue_folder
      __init__.py
      blue.py
      inblue.py
      /black_folder
        __init__.py
        black.py
    /green_folder
      __init__.py
      green.py
```

```python
# example5/red_folder/blue_folder/inblue.py
print("red_folder/blue_folder/inblue.py")

from ..green_folder import green
from .. import red
from .black_folder import black
```

```python
>>> import example5.red_folder.blue_folder.inblue
red_folder/__init__.py
red_folder/blue_folder/__init__.py
red_folder/blue_folder/inblue.py    # print
red_folder/green_folder/__init__.py
red_folder/green_folder/green.py    # green
red_folder/red.py                   # red
red_folder/blue_folder/black_folder/__init__.py
red_folder/blue_folder/black_folder/black.py  # black
```

### 참고

* <https://docs.python.org/ko/3/tutorial/modules.html>
