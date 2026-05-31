import 'datum_type.dart';
import 'model.dart';
import 'model_source.dart';

/// A pluggable provider of inference for a given platform/technique.
///
/// `tract` is backend-agnostic: native platforms reach the Rust engine over
/// FFI. Several backends can be registered at once; the active one is chosen by
/// [isAvailable] and [priority] (see [Tract]).
abstract class TractBackend {
  /// A short, stable identifier (e.g. `ffi`).
  String get name;

  /// Whether this backend can actually run in the current environment.
  ///
  /// Should be cheap and side-effect free; it is consulted during backend
  /// selection. The FFI backend reports `false` when the native library cannot
  /// be located/loaded.
  bool get isAvailable;

  /// Selection weight when multiple backends are available; higher wins.
  int get priority;

  /// Prepares the backend for use. Called once before the first load.
  ///
  /// Must be idempotent.
  Future<void> initialize();

  /// Loads an ONNX model from [source].
  ///
  /// When [inputShape] is given, input 0 is pinned to that shape (with element
  /// type [inputType], default [DatumType.f32]) before the model is optimized,
  /// resolving symbolic dimensions such as a dynamic batch size.
  Future<Model> loadOnnx(
    ModelSource source, {
    List<int>? inputShape,
    DatumType inputType,
  });

  /// Loads a tract NNEF model from [source] (a directory or `.nnef.tgz`).
  Future<Model> loadNnef(ModelSource source);

  /// Releases any global resources held by the backend.
  Future<void> dispose();
}
