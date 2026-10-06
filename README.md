# tract

Neural network inference for Dart and Flutter across platforms. Native builds
require Rust in addition to Dart or Flutter. Precompiled binaries are planned
to remove that build requirement.

It wraps the Rust [`tract`](https://github.com/sonos/tract) inference engine,
which loads and runs ONNX and NNEF models. tract is pure Rust with no system
dependencies, so it cross-compiles to every native target.

This is a [melos](https://melos.invertase.dev) monorepo:

| Package | What it is |
|---|---|
| [`packages/tract`](packages/tract) | Pure Dart (no Flutter); works in CLI tools. Loads and runs models over `dart:ffi`, with a pluggable backend registry. |
| [`packages/tract_flutter`](packages/tract_flutter) | Flutter plugin. Builds the native library automatically (via cargokit) and re-exports the `tract` API. |
| [`native/tract_ffi`](native/tract_ffi) | Rust crate (the C ABI behind the FFI backend). Wraps `tract-onnx`/`tract-nnef`: load an ONNX/NNEF model, then run it on input tensors. |

## Quick start (Dart CLI)

```dart
import 'package:tract/tract.dart';

final model = await Tract.loadOnnxFile('mobilenet.onnx',
    inputShape: [1, 3, 224, 224]);
final outputs = model.run([Tensor.float32(pixels, [1, 3, 224, 224])]);
print(outputs.single.asFloat32List());
model.dispose();
```

Build the native library first: `cargo build --manifest-path native/tract_ffi/Cargo.toml`.

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
| macOS | Expected to work (pure-Rust engine) |
| Windows | Expected to work (pure-Rust engine) |
| Android | Expected to work (pure-Rust engine) |
| iOS | Expected to work (pure-Rust engine) |
| Web | Not supported yet (no `dart:ffi`; tract is not compiled to WASM here) |

The native crate, its FFI bindings and ONNX inference are verified on Linux.
The other native targets build from the same pure-Rust engine; bundling/loading
on each is set up but not yet exercised end to end. Web inference would require
compiling tract to WASM and calling it via JS interop, which cargokit does not
build - it is a known gap.

## Development

Driven by [melos](https://melos.invertase.dev) (config lives in the root
`pubspec.yaml`). After `dart pub get` (or `flutter pub get`):

```sh
dart run melos run native:build   # build the Rust shared library
dart run melos run analyze        # analyze tract + tract_flutter
dart run melos run format         # check Dart formatting (format:fix to apply)
dart run melos run test           # Dart + Flutter tests
dart run melos run native:test    # Rust crate tests
```

Building the Rust crate requires Rust 1.91 or newer (tract's MSRV, newer than
some distros ship); `rustup toolchain install stable` and build with it. cargokit
does not update an installed toolchain, so run `rustup update stable` if a
Flutter build reports an old rustc.
