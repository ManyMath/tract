# tract_flutter

Flutter integration for [`tract`](../tract): cross-platform neural network
inference with no system dependencies for users to install.

This package re-exports the full `tract` API and adds helpers for loading
models bundled as Flutter assets. The native library comes from `tract`
itself: its build hook compiles the bundled `tract_ffi` Rust crate for each
platform you build for, and Flutter bundles the result with your app.

```dart
import 'package:tract_flutter/tract_flutter.dart';

await TractFlutter.ensureInitialized();
final model = await TractFlutter.loadOnnxAsset('assets/mobilenet.onnx',
    inputShape: [1, 3, 224, 224]);
final outputs = model.run([Tensor.float32(pixels, [1, 3, 224, 224])]);
model.dispose();
```

The native crate wraps the Rust [`tract`](https://github.com/sonos/tract) engine,
which is pure Rust, so it cross-compiles to Android, iOS, macOS, Windows and
Linux without extra toolchains. Web is not supported (no `dart:ffi`).

Building requires Rust (`rustup`); the first build installs the pinned
toolchain and its targets. Android builds also need NDK 27 or newer.
