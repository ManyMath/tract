/// Base class for all errors thrown by `tract`.
class TractException implements Exception {
  const TractException(this.message);

  final String message;

  @override
  String toString() => 'TractException: $message';
}

/// Thrown when no registered backend can run in the current environment.
///
/// On the web there is currently no native inference backend; load and run
/// models on a native platform, or register a custom [TractBackend].
class NoBackendAvailableException extends TractException {
  const NoBackendAvailableException([
    super.message =
        'No tract backend is available in this environment. '
        'Native inference is unavailable (e.g. on the web).',
  ]);
}

/// Thrown when a model fails to load (bad bytes/path, unsupported op, or an
/// unresolved symbolic shape).
class ModelLoadException extends TractException {
  const ModelLoadException(super.message);
}

/// Thrown when running a model fails (e.g. an input shape/type mismatch).
class InferenceException extends TractException {
  /// Creates an [InferenceException] with the given [message].
  const InferenceException(super.message);
}

/// Thrown when a backend is asked to load a [ModelSource] kind or option it
/// does not support.
class UnsupportedSourceException extends TractException {
  const UnsupportedSourceException(super.message);
}
