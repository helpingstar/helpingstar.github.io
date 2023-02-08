---
layout: single
title: "파이썬 멀티스레딩, multithreading"
date: 2023-02-08 15:37:43
lastmod : 2023-02-08 15:37:40
categories: python
tag: [python, multithreading]
toc: true
toc_sticky: true
---

[파이썬 3 바이블](https://product.kyobobook.co.kr/detail/S000001019529)을 참고하였다.

하나의 프로세스 안에 있는 스레드들은 각각 독립적으로 스택을 가지고 실행되지만, 코드와 데이터는 공유한다. 스레드의 실행은 어느 시점에서라도 중단되고(Preemptive) 다른 스레드로 실행구너이 넘어갈 수 있다. 스레드의 실행은 독립적이어서 다른 스레드와의 실행 순서에 관한 어떠한 가정도 할 수 없다.

파이썬은 내부적으로 전역 인터프리터 록(Global Interpreter Lock, GIL)을 사용한다. 이것은 시스템 하나에서 스레드 하나만 실행되도록 제한한다. 따라서 여러개의 코어가 있어도 파이썬 스레드는 하나의 코어에서만 실행된다.

# Multithreading

## threading 모듈

### 스레드 객체 생성

* **[threadex1.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/threadex1.py)**
* **[threadex2.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/threadex2.py)**

### Lock/RLock 객체

스레드가 좋은 점은 전역 변수를 공유할 수 있다는 것이다. 하지만, 여러 스레드에서 동시에 공유하는 변수를 수정하려고 하면 경쟁 조건(Race Condition) 문제가 발생한다. 따라서 상호 배제(Mutual Exclusion)를 구현하여 공유하는 변수가 올바르게 수정되는 것을 보장해야 한다. `Lock` 클래스를 이용하는 것이 대표적이다.

* **[threadex3.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/threadex3.py)**

`RLock` 클래스 객체는 `Lock` 클래스 객체와 같으나, 록을 소유하고 있는 스레드가 한 번 이상 `acquire()` 메서드를 호출할 수 있다. 록을 획득(acquire)만큼 해제(release)해야 록이 해제된다.

### Condition 객체

조건변수에 대한 내용은 일반적으로 운영 체제의 모니터에서 다룬다. 조건 변수는 내부에 하나의 스레드 대기 큐(Queue)를 가진다. `wait()` 메서드를 호출하는 스레드는 이 대기 큐에 넣어지고 대기(Sleep) 상태가 된다. `notify()` 메서드를 호출하는 스레드는 이 대기 큐에서 하나의 스레드를 깨운다.

전형적으로 `wait()`와 `notify()` 메서드는 록을 획득한 상태에서 호출된다. 다음과 같이 모든 스레드가 참조하는 공유변수 `cv`가 있다고 하자.

```python
cv = threading.Condition()
```

`wait()` 메서드를 호출하는 스레드는 다음과 같다.

```python
cv.acquire()    # 록을 얻는다.
# A
while ...:      # 적절한 조건이 주어진다.
cv.wait()       # 잠시 록을 해제하고, cv 내부 대기 큐에서 기다린다.
# B
cv.release()    # 록을 해제한다.
```

또 다른 스레드는 `notify()` 메서드로 대기중인 스레드를 깨운다.

```python
cv.acquire()    # 록을 얻는다.
# C
cv.notify()     # cv 내부 대기 큐에서 기다리고 있는 스레드 하나를 깨운다.
# D
cv.release()    # 록을 해제한다.
```

A, B, C, D 모두 록을 얻은 상태에서만 실행 가능한 코드이다. A, B, C, D에는 하나의 스레드만이 존재할 수 있다.

`cv.notifyAll()` : 대기 큐에서 기다리고 있는 스레드 모두를 깨운다.

P1(A) -> P2(A) -> P3(C) -> P1(wait) -> P2(wait) -> P3(notify) -> P1(wakeup) -> P3(release) -> P1(B) -> P2(...wait for notify)

* **[threadex4.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/threadex3.py)**

### Semaphore 객체

가장 오래된 동기화 프리미티브(Primitive)이다. 내부에 정수형의 카운터 변수(`_value`)를 가지고 있으며 이것은 세마포어 변수를 만들 때 초기화된다. `acquire()`에 의해 1씩 감소하고 `release()`에 의해 1씩 증가한다. 0보다 작을 수 없다.

`acquire()` 실행시 카운터 값이 0이면 스레드는 세마포어 변수의 대기 큐에 넣어져 블록상태(Block, 실행을 멈추고 어떤 사건이 일어나기를 대기하는 상태, CPU를 점유하지 않음)로 들어간다.

`release()` 는 대기 스레드가 있는지 검사하고 있으면 가장 오래된 스레드를 깨우고 없으면 카운터 값이 1만큼 증가한다.

* **[threadex5.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/threadex5.py)**

### Event 객체

`set()`, `clear()`, `wait()`, `isSet()` 의 4개의 메서드를 가지고 있다. 이벤트 객체는 내부에 하나의 이벤트 플래그(`_flag`)를 가진다. 초깃값은 0이다.

* `set()` : 내부 플래그를 1로 만듬
* `clear()` : 내부 플래그를 0으로 만듬
* `wait()` : 내부 플래그가 1이면 즉시 반환하며, 0이면 다른 스레드에 의해서 1이 될 때까지 블록(대기) 상태에 들어간다. `wait()` 메서드는 내부 플래그 값을 바꾸지 않는다.
* `isSet()` : 내무 플래그의 상태를 넘겨준다. `_flag`를 직접 참조하지 말자.

---

* **[threadex6.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/threadex6.py)**

## queue 모듈

파이썬은 멀티스레드 환경에서 사용할 수 있는 `queue` 모듈을 제공한다.

* `queue.Queue(n)` : `n` -> 최대크기
* `put(item, block=True, timeout=None)` : 큐에 item을 넣습니다. 선택적 인자 block=`True`이고 timeout이 `None`(기본값)이면, 사용 가능한 슬롯이 확보될 때까지 필요하면 블록합니다. timeout이 양수면, 최대 timeout 초 동안 블록하고 그 시간 내에 사용 가능한 슬롯이 없으면 Full 예외가 발생합니다. 그렇지 않으면 (block=`False`), 빈 슬롯이 즉시 사용할 수 있으면 큐에 항목을 넣고, 그렇지 않으면 Full 예외를 발생시킵니다 (이때 timeout은 무시됩니다).
* `put_nowait(item)` : `put(item, False)`와 동일하다 바로 시도해서 없으면 기다리지 않고 바로 예외 발생
* `qsize()` : 큐의 크기
* `empty()` : 비어있으면 `True`, 아니면 `False`
* `full()` : 차 있으면 `True`, 아니면 `False`
* `get()` : 큐에서 항목을 제거하고 반환합니다. 선택적 인자 block=`True`이고 timeout이 None(기본값)이면, 항목이 사용 가능할 때까지 필요하면 블록합니다. timeout이 양수면, 최대 timeout 초 동안 블록하고 그 시간 내에 사용 가능한 항목이 없으면 Empty 예외가 발생합니다. 그렇지 않으면 (block=`False`), 즉시 사용할 수 있는 항목이 있으면 반환하고, 그렇지 않으면 Empty 예외를 발생시킵니다 (이때 timeout은 무시됩니다).
* `get_nowait()` : `get(False)` 와 같다.

---

* **[threadex7.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/threadex7.py)**
* **[threadex8.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/threadex8.py)**
