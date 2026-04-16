---
layout: privacy
title: "2048 coach Privacy Policy"
permalink: /app/2048coach/privacy/
author_profile: false
sitemap: false
---

# Privacy Policy

Effective date: April 17, 2026

2048 Coach ("the App"), published on Google Play as "2048 Coach - AI Solver," is designed to help you recreate 2048 board states and receive move recommendations on your device. This Privacy Policy describes how the current version of the App processes information based on the current codebase.

## Summary

- The App does not require account registration.
- The App does not currently include advertising, analytics, crash reporting, or in-app purchase SDKs.
- The App does not request the Android `INTERNET` permission.
- Move analysis is performed on-device using a model packaged with the App.
- The App stores a small amount of local state on your device to remember onboarding and settings.

## Information We Process

### 1. Information stored locally on your device

The App currently stores limited data in private app storage, including:

- whether onboarding has been completed; and
- gameplay preferences such as tile spawning, auto analysis, and animation settings.

The App may also preserve your current board state, score, undo history, and recent recommendation results for local session restoration when Android recreates the App after interruption.

### 2. Information used for on-device analysis

When you use analysis features, the App processes the 4x4 board values you enter. This analysis runs locally on your device using the bundled `game2048_policy_value_float32.tflite` model.

Based on the current implementation, the model is loaded from the App's packaged assets and your board data is not uploaded to our servers for analysis.

## Information We Do Not Currently Collect

Based on the current codebase, the App does not currently collect or request:

- your name, email address, phone number, or account credentials;
- precise location, contacts, photos, camera, or microphone access;
- payment card information or billing details; or
- advertising identifiers or analytics profiles.

## Permissions and Device Features

The App's own manifest does not declare sensitive permissions such as camera, microphone, contacts, or location access.

However, the current release build includes the following permissions through bundled ML/runtime dependencies:

- `android.permission.ACCESS_NETWORK_STATE`
- `android.permission.WAKE_LOCK`
- `android.permission.RECEIVE_BOOT_COMPLETED`
- `android.permission.FOREGROUND_SERVICE`
- `android.permission.FOREGROUND_SERVICE_DATA_SYNC`

These entries are introduced by transitive libraries used to support the on-device ML runtime. Based on the current application code, the App does not use them to upload your gameplay data, deliver ads, create user accounts, or perform analytics.

## Third-Party Libraries

The App currently includes Google AI Edge LiteRT and related Google Play support libraries so it can run the recommendation model on-device. Based on the current code, these libraries are used for local model runtime support, not for advertising, analytics, or user profiling.

If you choose to open an external link from the App, your browser or operating system may connect to that destination under the privacy policy of the destination service.

## Backup and Device Transfer

The App currently enables Android backup and data extraction features in the manifest. The backup configuration files do not currently define specific exclusions.

As a result, depending on your Android version, device settings, and platform backup services, some local app data may be included in device backup, restore, or device-to-device transfer. We do not separately control those platform backup services.

## Data Retention

Local data stored by the App generally remains on your device until you:

- clear the App's storage;
- uninstall the App; or
- change or overwrite the stored settings and local state.

If Android backup or device transfer is enabled on your device, some local app data may persist through those platform features.

## Data Sharing

Based on the current codebase, the App does not currently sell your personal information and does not intentionally share your gameplay data with our own servers.

## Your Choices

You can:

- use the App without creating an account;
- disable App features such as auto analysis, tile spawning, or animations in settings;
- clear App storage through Android settings to remove locally stored App data; and
- disable Android backup or device transfer features in your device settings if you do not want platform-level backup of local app data.

## Changes to This Privacy Policy

We may update this Privacy Policy if the App's features or data practices change. When we do, we will update the effective date above.

## Contact

If you have questions about this Privacy Policy, please contact:

iamhelpingstar@gmail.com
