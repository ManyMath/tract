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
| Web | JavaScript only, via `tract_ffi.wasm` and [`@manymath/tract-wasm`](#webassembly-javascript); the Dart package has no web backend (no `dart:ffi`) |

Linux runs the CLI, the tests and the Flutter example end to end; Android runs
the Flutter example end to end (on an API 33 emulator). The example's
integration test checks a platform in one command:
`cd packages/tract_flutter/example && flutter test integration_test -d <device>`.
Web inference goes through the separate WebAssembly build described below; the
Dart package does not call it.

## WebAssembly (JavaScript)

`packages/tract/native` also builds for `wasm32-unknown-unknown` as a module
with no imports (no `wasm-bindgen`), wrapped by the ESM package
`packages/tract_wasm` (`@manymath/tract-wasm`, private, not published).

```sh
packages/tract/native/tool/build_wasm.sh   # -> packages/tract_wasm/tract_ffi.wasm
cd packages/tract_wasm && npm test
```

The script builds cargo's size-oriented `wasm` profile, runs `wasm-opt -Oz`
(binaryen), asserts zero imports and enforces a size budget. The result is about
7.8 MB (2.4 MB gzipped). The module uses WebAssembly SIMD (simd128, plus bulk
memory and the other features rustc enables by default), so the browser or
Node must support it: Chrome 91+, Firefox 89+, Safari 16.4+, Node 16.4+.

```js
import { load } from '@manymath/tract-wasm';

const tract = await load(await fetch('tract_ffi.wasm'));
const model = tract.loadModel(onnxBytes, { normalization: 'mnist' });
const { label, probs } = model.classify28x28(pixels); // 784 Uint8 or Float32
const logits = model.run(new Float32Array(784), [1, 1, 28, 28]);
model.free();
```

`loadModel` parses and optimizes the model once and returns a handle; run it as
often as needed. The underlying C ABI is `tract_model_load[_with_options]`,
`tract_model_run_f32` / `tract_model_run_u8` / `tract_infer_28x28` and
`tract_model_free`, with results read through `tract_buffer_*`. The one-shot
`tract_infer_bytes` takes the model bytes on every call and keeps a one-entry
cache that compares them in full, so it is correct but slow for large models.

Pixel inputs can be normalized at load time as
`((invert ? max - v : v) * scale - mean) / std`. Presets:

| `normalization` | Formula |
|---|---|
| `none` (default) | values passed through unchanged |
| `unit` | `x / 255` |
| `mnist` | `(x / 255 - 0.1307) / 0.3081` |

or pass `{ scale, mean, std, max, invert }`. The bundled `mnist-8.onnx`
classifies equally well with `unit` and `mnist`, and worse with raw 0-255.

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
