# tract

Neural network inference for Dart and Flutter across platforms. Native builds
require Rust in addition to Dart or Flutter.

It wraps the Rust [`tract`](https://github.com/sonos/tract) inference engine,
which loads and runs ONNX and NNEF models. tract is pure Rust with no system
dependencies, so it cross-compiles to every native target.

This is a [melos](https://melos.invertase.dev) monorepo:

| Package | What it is |
|---|---|
| [`packages/tract`](packages/tract) | Pure Dart (no Flutter); works in CLI tools. Loads and runs models over `dart:ffi`, with a pluggable backend registry. Its build hook compiles the bundled Rust crate. |
| [`packages/tract/native`](packages/tract/native) | The `tract_ffi` Rust crate (the C ABI behind the FFI backend). Wraps `tract-onnx`/`tract-nnef`: load an ONNX/NNEF model, then run it on input tensors. |
| [`packages/tract_flutter`](packages/tract_flutter) | Flutter package. Re-exports the `tract` API and adds helpers for loading models from Flutter assets. |

## Quick start (Dart CLI)

```dart
import 'package:tract/tract.dart';

final model = await Tract.loadOnnxFile('mobilenet.onnx',
    inputShape: [1, 3, 224, 224]);
final outputs = model.run([Tensor.float32(pixels, [1, 3, 224, 224])]);
print(outputs.single.asFloat32List());
model.dispose();
```

There is no separate native build step: `tract`'s build hook (via
[`native_toolchain_rust`](https://pub.dev/packages/native_toolchain_rust))
compiles the crate the first time you `dart run`, `dart test`, `dart build` or
build a Flutter app, and bundles it for the target platform.

### Runnable demos

Both examples run the real MNIST classifier from the ONNX Model Zoo - not a toy
or echo, so the predictions are checkable against the digits you see.

```sh
# CLI: classify the bundled real MNIST digits, rendering one as text art.
dart run packages/tract/example/tract_example.dart

# Flutter: draw a digit with the mouse or a finger, then recognize it.
cd packages/tract_flutter/example && flutter run
```

## Status

| Host | Status |
|---|---|
| Linux | Working |
| Android | Working |
| macOS | Expected to work (built by the same hook, not yet run) |
| Windows | Expected to work (built by the same hook, not yet run) |
| iOS | Expected to work (built by the same hook, not yet run) |
| Web | Not supported yet (no `dart:ffi`; tract is not compiled to WASM here) |

Linux runs the CLI, the tests and the Flutter example end to end; Android runs
the Flutter example end to end (on an API 33 emulator).
Web inference would require compiling tract to WASM and calling it via JS
interop, which the build hook does not do - it is a known gap.

## Development

Driven by [melos](https://melos.invertase.dev) (config lives in the root
`pubspec.yaml`). After `dart pub get` (or `flutter pub get`):

```sh
dart run melos run analyze        # analyze tract + tract_flutter
dart run melos run format         # check Dart formatting (format:fix to apply)
dart run melos run test           # Dart + Flutter tests
dart run melos run native:test    # Rust crate tests
```

The crate's `rust-toolchain.toml` pins the Rust version (at least 1.91, tract's
MSRV) and every target the hook may build for; rustup installs them on first
use. Android builds also need NDK 27 or newer.
