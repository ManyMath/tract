# tract_flutter

Flutter integration for [`tract`](../tract): cross-platform neural network
inference with no system dependencies for users or devs to install.

This package builds the shared `tract_ffi` Rust crate automatically (via
[cargokit](https://github.com/ManyMath/cargokit)) and bundles it with your app,
then re-exports the full `tract` API.

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

Building requires Rust (`rustup`). To remove even that for consumers, the repo
is set up for cargokit precompiled binaries:

- `native/tract_ffi/cargokit.yaml` holds the public signing key and the
  release URL prefix.
- `.github/workflows/precompile.yml` builds and uploads signed binaries to
  `precompiled_<crate-hash>` releases on tag pushes. It needs a
  `CARGOKIT_PRIVATE_KEY` repository secret: the hex private key printed by
  `dart run build_tool gen-key` (keep it secret; never commit it).

Once a release exists, builds on machines without `rustup` download the
matching signed binary instead of compiling. Where `rustup` is installed,
cargokit builds from source unless the app opts in with
`use_precompiled_binaries: true` in a `cargokit_options.yaml` at its root. When
no matching binary has been published, cargokit builds from source, which
needs `rustup`.
