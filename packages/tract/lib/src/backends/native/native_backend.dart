import '../../tract_backend.dart';

// Selects the FFI implementation on native platforms and a no-op on the web,
// so importing `tract` never pulls in `dart:ffi` where it does not exist.
import 'native_backend_ffi.dart'
    if (dart.library.js_interop) 'native_backend_web.dart'
    as impl;

/// Returns the default native backend for this platform, or `null` when none
/// applies (e.g. the web, where there is no native inference backend).
TractBackend? createNativeBackend() => impl.createNativeBackend();
