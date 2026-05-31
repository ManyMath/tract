# Changelog

## 0.0.1

- Initial scaffolding.
- Core API: `Tract`, `ModelSource`, `Tensor`, `DatumType`, `Model`, pluggable
  `TractBackend`s.
- FFI backend that loads and runs ONNX and tract NNEF models via the `tract_ffi`
  Rust library (the sonos/tract engine). Web-safe via conditional imports (no
  native inference on web yet).
- ONNX loading from a file or in-memory bytes, with an optional `inputShape`
  override to pin input 0 and resolve symbolic dimensions before optimization.
- `Tensor.float32`/`float64`/`int32`/`int64`/`uint8` constructors and matching
  typed-list views.
- Pure-Dart CLI example that classifies real MNIST digits with the ONNX Model
  Zoo MNIST network, rendering each digit as ASCII next to its prediction.
