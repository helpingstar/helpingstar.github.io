---
layout: single
title: "Android DesignSystem : Best Practice"
date: 2025-10-17 11:00:00
lastmod : 2025-10-17 11:00:00
categories: android
tag: [android, coroutine]
toc: true
toc_sticky: true
published: false
---

# NowInAndroid

## Color

```kotlin
// core.designsystem.theme.Color.kt
internal val Blue10 = Color(0xFF001F28)
...
```

## Type

```kotlin
internal val NiaTypography = Typography(
    displayLarge = TextStyle(
        fontWeight = FontWeight.Normal,
        fontSize = 57.sp,
        lineHeight = 64.sp,
        letterSpacing = (-0.25).sp,
    ),
    ...
    headlineSmall = TextStyle(
        fontWeight = FontWeight.Normal,
        fontSize = 24.sp,
        lineHeight = 32.sp,
        letterSpacing = 0.sp,
        lineHeightStyle = LineHeightStyle(
            alignment = Alignment.Bottom,
            trim = Trim.None,
        ),
    ),
    ...
```