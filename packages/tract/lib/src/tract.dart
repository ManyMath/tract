import 'dart:typed_data';

import 'backends/native/native_backend.dart';
import 'datum_type.dart';
import 'exceptions.dart';
import 'model.dart';
import 'model_source.dart';
import 'tract_backend.dart';

/// Entry point for loading and running models.
///
/// `tract` keeps a registry of [TractBackend]s and picks the best available one
/// for the current environment (highest [TractBackend.priority] among those
/// whose [TractBackend.isAvailable] is true). Apps can register extra backends
/// or pin a specific one.
///
/// ```dart
/// final model = await Tract.loadOnnxFile('mobilenet.onnx',
///     inputShape: [1, 3, 224, 224]);
/// final out = model.run([Tensor.float32(pixels, [1, 3, 224, 224])]);
/// model.dispose();
/// ```
class Tract {
  Tract._();

  static final List<TractBackend> _backends = [];
  static final Set<TractBackend> _initialized = Set.identity();
  static TractBackend? _active;
  static bool _defaultsRegistered = false;

  /// Registers the backends that ship with `tract`.
  static void _ensureDefaults() {
    if (_defaultsRegistered) return;
    _defaultsRegistered = true;
    // The platform's native backend: FFI on native, none on the web.
    final native = createNativeBackend();
    if (native != null) registerBackend(native);
  }

  /// All registered backends, in registration order.
  static List<TractBackend> get backends {
    _ensureDefaults();
    return List.unmodifiable(_backends);
  }

  /// Adds [backend] to the registry.
  ///
  /// Pass [makeActive] to select it immediately regardless of priority.
  /// Re-registering an instance is a no-op.
  static void registerBackend(TractBackend backend, {bool makeActive = false}) {
    _ensureDefaults();
    if (!_backends.contains(backend)) _backends.add(backend);
    if (makeActive) {
      _active = backend;
    } else {
      // Force re-selection so a newly added, higher-priority backend wins.
      _active = null;
    }
  }

  /// The backend currently in use, selecting one on first access.
  ///
  /// Throws [NoBackendAvailableException] if nothing is available.
  static TractBackend get backend {
    _ensureDefaults();
    return _active ??= _select();
  }

  /// Pins a specific registered backend by [name].
  static void useBackend(String name) {
    _ensureDefaults();
    final match = _backends.where((b) => b.name == name);
    if (match.isEmpty) {
      throw NoBackendAvailableException(
        'No backend named "$name" is registered.',
      );
    }
    _active = match.first;
  }

  static TractBackend _select() {
    final available = _backends.where((b) => b.isAvailable).toList()
      ..sort((a, b) => b.priority.compareTo(a.priority));
    if (available.isEmpty) throw const NoBackendAvailableException();
    return available.first;
  }

  static Future<TractBackend> _ready() async {
    final b = backend;
    if (!_initialized.contains(b)) {
      await b.initialize();
      _initialized.add(b);
    }
    return b;
  }

  /// Loads an ONNX model from [source] with the active backend.
  ///
  /// When [inputShape] is given, input 0 is pinned to that shape (element type
  /// [inputType], default [DatumType.f32]) before optimization, resolving
  /// symbolic dimensions such as a dynamic batch size.
  static Future<Model> loadOnnx(
    ModelSource source, {
    List<int>? inputShape,
    DatumType inputType = DatumType.f32,
  }) async {
    final b = await _ready();
    return b.loadOnnx(source, inputShape: inputShape, inputType: inputType);
  }

  /// Loads a tract NNEF model from [source] with the active backend.
  static Future<Model> loadNnef(ModelSource source) async {
    final b = await _ready();
    return b.loadNnef(source);
  }

  /// Convenience for [loadOnnx] with a [FileSource].
  static Future<Model> loadOnnxFile(
    String path, {
    List<int>? inputShape,
    DatumType inputType = DatumType.f32,
  }) => loadOnnx(
    ModelSource.file(path),
    inputShape: inputShape,
    inputType: inputType,
  );

  /// Convenience for [loadOnnx] with a [BytesSource].
  static Future<Model> loadOnnxBytes(
    Uint8List bytes, {
    List<int>? inputShape,
    DatumType inputType = DatumType.f32,
  }) => loadOnnx(
    ModelSource.bytes(bytes),
    inputShape: inputShape,
    inputType: inputType,
  );

  /// Disposes all initialized backends and clears selection/registry.
  ///
  /// Primarily for tests; after this the defaults are re-seeded on next use.
  static Future<void> reset() async {
    for (final b in _initialized) {
      await b.dispose();
    }
    _initialized.clear();
    _backends.clear();
    _active = null;
    _defaultsRegistered = false;
  }
}
