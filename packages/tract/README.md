# `tract`

Cross-platform neural network inference for Dart and Flutter with the goal of
no system dependencies that need to be installed.  Pure Dart (works in CLI
tools) with model inference from the Rust [`tract`](https://github.com/sonos/tract)
engine over FFI.

The Rust crate ships inside this package (`native/`), and a build hook
compiles it for the platform you run or build for, so `dart run`, `dart test`,
`dart build` and Flutter builds pick it up automatically. The only build-time
requirement beyond Dart/Flutter is Rust (`rustup`); the first build installs the
pinned toolchain. For Flutter apps, `tract_flutter` adds helpers for loading
models from assets.

```dart
import 'package:tract/tract.dart';

// Load an ONNX model (pin input 0's shape to resolve a dynamic batch size).
final model = await Tract.loadOnnxFile('mobilenet.onnx',
    inputShape: [1, 3, 224, 224]);

final input = Tensor.float32(pixels, [1, 3, 224, 224]);
final outputs = model.run([input]);
print(outputs.single.asFloat32List());

model.dispose();
```

- [`Tract`] picks the best available backend and loads models
  ([`loadOnnxFile`]/[`loadOnnxBytes`]/[`loadNnef`]).
- [`Tensor`] carries a [`DatumType`], a shape and native-endian bytes, with
  `Tensor.float32(...)` constructors and `asFloat32List()` views.
- [`Model.run`] takes one [`Tensor`] per input and returns the outputs.

## Status

| Target | Status |
|---|---|
| Linux | Working |
| Android | Working |
| macOS | Should work (built by the same hook, not yet run) |
| Windows | Should work (built by the same hook, not yet run) |
| iOS | Should work (built by the same hook, not yet run) |
| Web | Not supported by this Dart package (no `dart:ffi`); the repository's JavaScript package `@manymath/tract-wasm` runs the same engine as WebAssembly |

[`Tract`]: https://pub.dev/documentation/tract/latest/tract/Tract-class.html
[`Tensor`]: https://pub.dev/documentation/tract/latest/tract/Tensor-class.html
[`DatumType`]: https://pub.dev/documentation/tract/latest/tract/DatumType.html
[`Model.run`]: https://pub.dev/documentation/tract/latest/tract/Model/run.html
[`loadOnnxFile`]: https://pub.dev/documentation/tract/latest/tract/Tract/loadOnnxFile.html
[`loadOnnxBytes`]: https://pub.dev/documentation/tract/latest/tract/Tract/loadOnnxBytes.html
[`loadNnef`]: https://pub.dev/documentation/tract/latest/tract/Tract/loadNnef.html
