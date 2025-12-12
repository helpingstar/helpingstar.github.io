---
layout: single
title: "Android build-logic"
date: 2025-12-08 23:00:00
lastmod : 2025-12-08 23:00:00
categories: android
tag: [Android, build-logic]
toc: true
toc_sticky: true
published: true
---

## `root/settings.gradle.kts`

```kotlin
pluginManagement {
  includeBuild("build-logic")
}
```

- 이 설정으로 `build-logic` 디렉토리가 독립적인 Gradle 프로젝트로 포함되며, 여기서 정의한 플러그인들을 메인 프로젝트에서 사용할 수 있게 된다.

## `build-logic/gradle.properties`

### `org.gradle.parallel=true`

- 의미 : 여러 project(모듈)의 task를 병렬로 실행한다.
- 동작방식 : 기본적으로 Gradle은 project를 하나씩 차례로 실행하지만 parallel을 켜면, 서로 의존성이 없는 project들의 task를 여러 worker thread에서 동시에 실행한다.

### `org.gradle.caching=true`

- 의미 : Gradle build cache를 활성화한다.
- 동작 방식 : 각 task는 input과 output이 있다. 동일한 입력으로 task를 이미 실행한 적이 있다면, 다시 실행하지 않고 이전에 만든 결과물(output)을 재사용한다. 캐시는 로컬 디스크(local cache) 또는 원격 서버(remote cache)에 저장될 수 있다.

### `org.gradle.configureondemand=true`

- 의미 : build시에 필요한 project만 구성(configure) 하겠다는 설정이다.
- 기본 동작과의 차이
  - 기본 : `:app:assembleDebug`를 실행하더라도 모든 project를 모두 configure 한다.
  - configure on demand : 실제로 이 task에 직접/간접적으로 관련된 project만 configure 한다.

### `org.gradle.configuration-cache=true`

- 의미 : Gradle의 configuration-cache를 활성화한다.
- Configuration 단계 : Gradle이 `build.gradle.kts`, plugins, settings 등을 읽고, 전체 task/graph를 구성하는 단계
- 동작 방식 : 한 번 configuration을 수행하면, 그 결과("어떤 task가 어떤 설정으로 존재하는지" 등)를 파일로 저장한다. 다음에 동일/유사한 빌드를 실행할 때, configuration 단계를 다시 하지 않고 저장된 configuration 상태를 그대로 복원한다.

### `org.gradle.configuration-cache.parallel=true`

- 의미 : configuration-cache를 저장/복원할 때도 병렬 처리를 허용하겠다는 설정
- 동작 방식 : configuration-cache 사용 시, 캐시를 만들 때, 읽어올 때 내부적으로 여러 작업을 병렬로 처리하여 속도를 끌어올린다.


## 기타 잡것들

### [`org.gradle.api.Project`](https://docs.gradle.org/current/dsl/org.gradle.api.Project.html)

This interface is the main API you use to interact with Gradle from your build file. From a `Project`, you have programmatic access to all of Gradle's features.

이 인터페이스는 build 파일에서 Gradle과 상호작용할 때 사용하는 주요 API입니다. `Project`를 통해 Gradle의 모든 기능에 프로그래밍 방식으로 접근할 수 있습니다.

Dependencies

dependency 관리를 위해서는 `Project.getDependencies()` 메서드가 반환하는 `DependencyHandler`를 사용한다.
- Properties
  - `pluginManager` : The plugin manager for this plugin aware object.
- Method
  - `void dependencies(Closure configureClosure)` : project의 dependencies를 구성합니다. 이 메서드는 전달된 closure를 해당 project의 `DependencyHandler`에 대해 실행합니다. `DependencyHandler`는 closure의 delegate로 전달됩니다.
- Method inherited from interface org.gradle.api.plugins.PluginAware
  - [`void apply(Map<String, ?> options)`](https://docs.gradle.org/current/dsl/org.gradle.api.Project.html#org.gradle.api.Project:apply(java.util.Map)) : 주어진 옵션을 map 형태로 사용하여 plugin 또는 script를 적용합니다. 해당 plugin이 이미 적용된 경우에는 아무 작업도 수행하지 않습니다.


### [`org.gradle.api.Plugin`](https://docs.gradle.org/current/javadoc/org/gradle/api/Plugin.html)

A `Plugin` represents an extension to Gradle. A plugin applies some configuration to a target object. Usually, this target object is a `Project`, but plugins can be applied to any type of objects.

`Plugin`은 Gradle을 확장하는 기능을 나타냅니다. Plugin은 target 객체에 특정 구성을 적용합니다. 일반적으로 이 target 객체는 `Project`이지만, plugin은 어떤 타입의 객체에도 적용될 수 있습니다.

- Method
  - `apply`(T target) : Apply this plugin to the given target object.

### [`org.gradle.api.plugins.PluginManager`](https://docs.gradle.org/current/dsl/org.gradle.api.plugins.PluginManager.html)

Facilitates applying plugins and determining which plugins have been applied to a `PluginAware` object.

plugin을 적용하고, 특정 plugin이 `PluginAware` 객체에 적용되었는지 여부를 확인하는 기능을 제공합니다.

- Methods
- [`void withPlugin(String id, Action<? super AppliedPlugin> action)`](https://docs.gradle.org/current/dsl/org.gradle.api.plugins.PluginManager.html#org.gradle.api.plugins.PluginManager:withPlugin(java.lang.String,%20org.gradle.api.Action)) : 지정된 plugin이 적용될 때, 주어진 action을 실행합니다. 



## 출처
- https://docs.gradle.org/