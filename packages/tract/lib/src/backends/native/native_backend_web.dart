import '../../tract_backend.dart';

/// There is no native inference backend on the web yet (`dart:ffi` is
/// unavailable and tract is not compiled to WASM here), so report none. Loading
/// a model then throws [NoBackendAvailableException] unless the app registers
/// its own [TractBackend].
TractBackend? createNativeBackend() => null;
