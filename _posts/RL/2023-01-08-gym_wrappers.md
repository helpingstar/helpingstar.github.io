---
layout: single
title: "gym Wrappers 정리"
date: 2023-01-08 13:58:15
lastmod : 2023-01-13 00:14:13
categories: RL
tag: [gymnaisum, gym, wrappers, vector]
toc: true
toc_sticky: true
---

`gym`이 `gymnasium`으로 바뀌었으나 서술의 편의를 위하여 `gym`으로 서술하겠다.

강화학습 학습시 `gym`이나 `wrappers`를 사용할 일이 참 많다. `Vector`와 함께 사용될때도 많은데 자주 사용하는 것 위주로 정리해본다. 자세한 설명은 [공식 문서](https://gymnasium.farama.org/)를 참고하기 바란다.

기본 사용 방법은 간단하다. 아래와 같이 환경을 Wrapper클래스 안에 인자로 넘겨주고 반환된 환경을 사용하면 된다. 기타 설정이 필요할 경우 파라미터도 같이 적어주면 된다.
```python
env = gym.make(env_id)
env = gym.wrappers.YourWrapper(env, param1=param1, param2=param2, ...)
```

# RecordVideo

```python
class gymnasium.wrappers.RecordVideo(env, video_folder, episode_trigger, step_trigger, video_length, disable_logger = False)
```

에피소드를 저장하는 wrapper이다. 결과 영상을 출력할 때나 디버그할 때 유용하다.

Parameter
* `env` : Wrappers를 적용할 환경
* `video_folder` : 영상을 저장할 폴더
* `episode_trigger` : 영상을 녹화할 조건이다. 200개의 episode마다 저장하고 싶다면 아래와 같이 하면 된다
  * `episode_trigger=lambda x: x % 200 == 0`
* `name_prefix` : 파일의 접두사이다. 귀찮아도 해주는게 편하다.
* `disable_logger` : `moviepy`의 logger의 출력여부를 결정한다. `False`로 하여 출력할 경우 아래와 같은 문구가 출력되는데 영상이 길거나 용량이 클 경우 진행상황을 알려준다. 보통 안하는 게 낫다.
```
Moviepy - Building video <PATH>
Moviepy - Writing video <PATH>
Moviepy - Done !
Moviepy - video ready <PATH>
```

# AutoResetWrapper

```python
class gymnasium.wrappers.AutoResetWrapper(env)
```

`step` 호출시 `episode`가 끝날 때마다 자동으로 `reset`을 호출하는 Wrapper이다.

`env.step`이 호출될 때  `terminated=True` 또는 `truncated=True` 이 반환될 경우 `env.reset()`이 자동으로 호출된다. 이때 `env.step()`의 반환은 `(new_obs, final_reward, final_terminated, final_truncated, info)`이 된다. 여기서 `info`는 다음 내용을 반환한다.
```
info
 ├ 'final_observation' : final_observation
 └ 'final_info' : final_info
```

이때 `new_obs`, `final_observation`, `final_reward` ... 등등의 순서가 헷갈리는데 풀어쓰면 다음과 같다.

```python
    # termination == True
    final_observation, final_reward, final_terminated, final_truncated, info = env.step(action)
    if final_terminated or final_truncated:
        new_obs, _ = env.reset()
```

# RecordEpisodeStatistics

```python
class gymnasium.wrappers.RecordEpisodeStatistics(env, deque_size = 100)
```

Record 라고 써있어서 영상을 저장하는 것인가 싶었지만 아니다. 에피소드가 끝날 때마다 각 에피소드의 누적 보상, 에피소드 길이, 에피소드 경과 시간 `info`에 반환한다

```
info
 └ 'episode'
     ├ 'r' : <array of cumulative reward>
     ├ 'l' : <array of episode length>
     └ 't' : <array of elapsed time since beginning of episode>
```

`AutoResetWrapper`, `RecordEpisodeStatistics` 두 개를 모두 하면 다음과 같이 된다.

```
info
 ├ 'final_observation' : final_observation
 ├ 'final_info'
 └ 'episode'
    ├ 'r' : <array of cumulative reward>
    ├ 'l' : <array of episode length>
    └ 't' : <array of elapsed time since beginning of episode>
```
