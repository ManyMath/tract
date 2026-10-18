## 0.0.1

- Flutter integration for `tract`: re-exports the `tract` API, whose build hook
  compiles and bundles the `tract_ffi` Rust crate (the sonos/tract engine), and
  adds `TractFlutter.ensureInitialized()`.
- `TractFlutter.loadOnnxAsset`/`loadNnefAsset` to load models bundled as Flutter
  assets (with an optional `package`/`bundle`).
- Example app: draw a digit on a canvas and recognize it with the MNIST
  ONNX model (plus a "Random sample" button that classifies bundled real MNIST
  test digits).
