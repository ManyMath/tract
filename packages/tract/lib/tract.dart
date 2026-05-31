/// Cross-platform neural network inference for Dart and Flutter.
///
/// `tract` is pure Dart and has no Flutter dependency, so it works in CLI tools
/// as well as Flutter apps. Models are loaded and run over FFI against the Rust
/// [tract](https://github.com/sonos/tract) engine; see [TractBackend] for the
/// pluggable backend contract and [Tract] for the entry point.
library;

export 'src/datum_type.dart';
export 'src/exceptions.dart';
export 'src/model.dart';
export 'src/model_source.dart';
export 'src/tensor.dart';
export 'src/tract.dart';
export 'src/tract_backend.dart';
