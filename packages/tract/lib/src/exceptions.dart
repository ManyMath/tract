/// Base class for all errors thrown by `tract`.
class TractException implements Exception {
  const TractException(this.message);

  final String message;

  // Spelled out rather than taken from runtimeType, which is minified in
  // release web builds.
  String get _name => 'TractException';

  @override
  String toString() => '$_name: $message';
}

/// Thrown when no registered backend can run in the current environment.
///
/// The native backend needs the `tract_ffi` library that tract's build hook
/// bundles, which happens for `dart run`, `dart test`, `dart build` and Flutter
/// builds, but not `dart compile` or the web. Elsewhere, register a custom
/// [TractBackend].
class NoBackendAvailableException extends TractException {
  const NoBackendAvailableException([
    super.message =
        'No tract backend is available. Native inference needs the tract_ffi '
        "library, which tract's build hook bundles for `dart run`, "
        '`dart test`, `dart build` and Flutter builds (not `dart compile`, '
        'and not the web).',
  ]);

  @override
  String get _name => 'NoBackendAvailableException';
}

/// Thrown when a model fails to load (bad bytes/path, unsupported op, or an
/// unresolved symbolic shape).
class ModelLoadException extends TractException {
  const ModelLoadException(super.message);

  @override
  String get _name => 'ModelLoadException';
}

/// Thrown when running a model fails (e.g. an input shape/type mismatch).
class InferenceException extends TractException {
  /// Creates an [InferenceException] with the given [message].
  const InferenceException(super.message);

  @override
  String get _name => 'InferenceException';
}

/// Thrown when a backend is asked to load a [ModelSource] kind or option it
/// does not support.
class UnsupportedSourceException extends TractException {
  const UnsupportedSourceException(super.message);

  @override
  String get _name => 'UnsupportedSourceException';
}
