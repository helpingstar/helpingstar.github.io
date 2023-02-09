---
layout: single
title: "파이썬 멀티프로세싱, python multiprocessing"
date: 2023-02-09 16:52:54
lastmod : 2023-02-09 16:52:51
categories: python
tag: [python, multiprocessing]
toc: true
toc_sticky: true
---

[파이썬 3 바이블](https://product.kyobobook.co.kr/detail/S000001019529)을 참고하였다.

파이썬은 여러 개의 코어에서 동시에 프로세스를 실행하게 할 수 있는 `multiprocessing` 모듈을 제공한다. 이 모듈은 `threading` 모듈과 거의 동일한 인터페이스를 제공한다.

# Multiprocessing

## 프로세스 객체 생성

**호출 가능한 객체(함수 등)를 생성자에 직접 전달**

`Process` 클래스를 이용하여 프로세스 객체를 생성한다.

```python
class multiprocessing.Process(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
```

* **[process01.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process01.py)**

`Process` 클래스가 프로세스를 생성하기 위하여 내부에서는 `fork()` 함수를 사용한다. 따라서 새로운 프로세스 객체를 생성할 때 전달되는 인수들은 전체가 복사되어 전달된다.

프로세스 객체를 생성하고 실행하는 코드는 반드시 `__main__`만 실행하는 코드 안에 적어야 한다. 즉, 이 코드가 `if __name__ == '__main__'`: 바깥으로 나오면 에러가 발생한다.

---
**하위 클래스에서 `run()` 메서드를 중복**

* **[process02.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process02.py)**

## 로그 기록

프로세스에 대한 정보를 확인하는 것은 병행 프로그램에 도움이 될 수 있다. 파이썬에서는 로그 정보를 위해서 `log_to_stderr()` 함수를 제공한다. 표준 에러 출력으로 로그 정보를 출력하도록 해준다. `log_to_stderr()` 함수는 로거 객체를 너멱주는데, 이것을 이용하여 로그 레벨을 설정할 수 있다. 로그 레벨을 설정할 때는 `logging` 모듈의 `DEBUG`와 `INFO`, `WARNING`, `ERROR`, `CRITICAL` 중 하나로 한다.

* **[process_logging.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process_logging.py)**

## 데몬 프로세스

기본적으로 메인 프로그램은 자식 프로세스가 종료될 때까지는 종료되지 않는다.

아래 예시에서는 메인 프로세스는 `non_daemon()` 프로세스가 종료될 때까지 프로세스로 남아 있다.

* **[process_non_daemon.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process_non_daemon.py)**

데몬 프로세스로 선언하는 것은 `Process()` 객체를 생성할 때 `daemon = True`로 설정하면 된다. 그러면 메인 프로세스는 `daemon()` 프로세스의 종료 여부에 관계 없이 종료해 버리고 `daemon()` 프로세스는 백그라운드 프로세스에 남게 된다.

## 프로세스간 통신

프로세스는 스레드와 같이 메모리 공유가 가능하지 않다. 그래서 프로세스 간 통신 방법이 필요하다.

### Queue 클래스
세 개 이상의 프로세스가 통신해야 한다면 `Queue` 클래스를 사용하는 것이 좋다. 공유 메모리보다 동기화, 록, 기타 문제를 자체적으로 해결하기 때문이다.

`multiprocessing` 모듈의 `Queue` 클래스는 스레드와 멀티프로세싱에 안전하게 사용된다.

* `Queue(x)` : 큐 객체 생성, `x` : 최대 크기
* `put(item, block=True, timeout=None)` : 데이터 추가
* `put_nowait(item)` : `put(item, False)`와 같다.
* `get(block=True, timeout=None)` : 데이터 읽기
* `get_nowait()` : `get(False)`와 동일.
* `qsize()` : 큐의 크기
* `empty()` : 큐가 비어있는 지 여부
* `full()` : 큐가 꽉 차있는 지 여부
* `task_done()` : 소비자 측에서 `get()` 메서드로 얻은 데이터 항목에 대해 처리를 완료했다는 것을 알리기 위해서이다.
* `join()` 큐의 모든 데이터 항목이 소비될 때까지(카운트가 0이 될 때까지) 기다린다. `put()` 메서드로 카운트가 올라가며 `task_done()` 메서드로 카운트가 내려간다.

---

* **[process03.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process03.py)**

### Pipe 클래스

입력과 출력 두 종단점을 가지고 있으므로 두 개의 프로세스가 데이터를 주고받는 경우에 적당하다.

파이프 객체를 생성하면 파이프의 양쪽 끝에서 통신에 사용하는 두 개의 `multiprocessing.connection.Pipeconnection` 객체를 얻는다. 파이프는 양방향 통신(Duplex)이 가능하다.

멀티프로세싱에서 파이프 연결 객체가 복사되어 전달되므로 사용하지 않는 연결 객체는 각각 닫아 주어야 한다. 이것이 잘못되면 블로킹 상태에서 빠져나오지 않을 수 있다.

* **[process04.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process03.py)**

### 공유 메모리

프로세스들 사이에 공유 메모리인 `Value`와 `Array` 객체를 이용하여 데이터를 공유할 수 있다. `Value` 객체는 값 하나를 저장하는 객체이고 `Array` 객체는 같은 자료형의 데이터 여러 개를 저장하는 객체이다.

다음 예와 같이 타입 코드와 초깃값으로 객체를 생성한다.

```python
s = multiprocessing.Value('d', 0.0)
a = Array('i', (1, 2, 3, 4, 5))
```

* **[process05.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process05.py)**
* **[process05_1.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process05_1.py)**

### 서버 프로세스

`Manager()` 함수로 생성되는 매니저 객체는 별도의 서버 프로세스를 만든다. 이 서버 프로세스는 파이썬 객체들을 가지고 관리하면서 다른 프로세스들이 프락시를 통해서 이 값들을 조작할 수 있도록 해준다. 매니저 객체는  list, dict, Namespace, Lock, RLock, Semaphore, BoundedSemaphore, Condition, Event, Barrier, Queue, Value 그리고 Array 형을 지원합니다. 매니저 서버 프로세스는 데이터를 가지고 있으면서 값이 변경되면 연관된 다른 프로세스들에 그 값을 전달하는 방식으로 동작한다.

네임스페이스 안의 list 값의 변화는 감지할 수 없다. 감지하고 싶다면 매니저 객체의 리스트를 사용해야 한다.


* **[process06.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process06.py)**


### Listener와 Client 클래스

같은 머신에서 동작하는 프로세스들뿐만 아니라 네트워크에서 동작하는 프로세스 간에도 메시지를 주고받을 수 있다. `multiprocessing.connection` 모듈의 `Client`와 `Listener` 클래스를 이용하여 이런 일을 할 수 있다. `Listener` 클래스는 소켓이나 윈도우 파이프를 감싸는 역할을 하고 `Client` 객체로부터 연결을 기다린다.

```python
class multiprocessing.connection.Listener([address[, family[, backlog[, authkey]]]])

multiprocessing.connection.Client(address[, family[, authkey]])
```

* `address` : 소켓의 주소나 이름 있는 파이프의 주소.
  * ex) `('localhost', 6000)`
* `family` : 소켓이나 파이프의 형식
  * ex) `AF_INET`, `AF_UNIX`, `AF_PIPE`
* `authkey` : 주어지지 않으면 `current_process().authkey` 바이트열이 인증키로 사용되며, 주어지면 다이제스트 인증이 사용된다.

---

* **[process09.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process09.py)**
* **[process10.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process10.py)**

### 동기화 문제

`multiprocessing` 모듈의 동기화는 `threading` 모듈의 동기화와 동일하다. 예를 들어 RLock, Condition, Semaphore, Event 객체 등이 동일하게 존재한다.

예시) [파이썬 멀티스레딩, multithreading](https://helpingstar.github.io/python/python_multithreading/)

### Pool 함수

`Pool` 함수는 작업 프로세스의 풀 객체를 생성하고 필요에 따라서 병렬로 함수를 실행할 수 있게 한다. 예를 들어, 다음은 4개의 작업 프로세스를 만든다. 풀 객체는 이 프로세스 풀을 관리하는 기능을 담당한다.

```python
pool = Pool(processes=4)
```

* `apply_async()` : 풀에서 하나의 작업 프로세스를 선택해 작업을 수행하고는 ApplyResult 객체를 즉시 반환한다(r1, r2, r3, r4) 연산이 종료되었는지는 `r1.ready()` 메서드로 파악할 수 있다.
* `r1.get(timeout=2)` : 2초 이내에 결과가 나타나지 않으면 TimeoutError 예외를 발생시킨다.
* `r1.get()` : 결과가 나타날 때까지 기다린다.
* `apply()` : 풀에서 하나의 작업 프로세스를 선택해 작업을 수행하고 결과가 나타날 때까지 기다린다.
* `map()` : 풀에서 모든 작업 프로세스를 선택해 작업을 수행하고 결과가 나타날 때까지 기다린다.(병렬형)
* `map_async()` : `map()` 함수와 같으나 결과 객체를 즉시 반환한다.
* `ready()` : 메서드를 통해 연산이 끝났는지 알 수 있으며 `get()` 메서드를 통해서 결과를 얻어낸다.

---

* **[process07.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process07.py)**
* **[process08.py](https://github.com/helpingstar/multi-python/blob/main/python3_bible/process08.py)**
