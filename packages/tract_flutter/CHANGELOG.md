## 0.0.1

- Flutter integration for `tract`: builds the shared `tract_ffi` Rust crate
  (the sonos/tract engine) via cargokit and bundles it. Re-exports the `tract`
  API and adds `TractFlutter.ensureInitialized()`.
- `TractFlutter.loadOnnxAsset`/`loadNnefAsset` to load models bundled as Flutter
  assets (with an optional `package`/`bundle`).
- Example app: draw a digit on a canvas and recognize it live with the MNIST
  ONNX model (plus a "Random sample" button that classifies bundled real MNIST
  test digits).
