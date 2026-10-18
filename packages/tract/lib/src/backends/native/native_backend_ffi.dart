import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../../datum_type.dart';
import '../../exceptions.dart';
import '../../model.dart';
import '../../model_source.dart';
import '../../tensor.dart';
import '../../tract_backend.dart';
import 'tract_ffi_bindings.dart';

/// The default native backend on platforms with `dart:ffi`.
TractBackend? createNativeBackend() => FfiBackend();

String _lastError() {
  final ptr = tractLastError();
  return ptr == nullptr ? 'unknown error' : ptr.toDartString();
}

/// Runs models by calling the native `tract_ffi` library over FFI.
///
/// The library is built from `native/` and bundled by the package's build
/// hook, so it is present wherever the hook ran for the target platform.
class FfiBackend extends TractBackend {
  bool? _available;

  @override
  String get name => 'ffi';

  @override
  int get priority => 100;

  @override
  bool get isAvailable => _available ??= _probe();

  // Resolving a symbol throws (rather than crashing) when no library was
  // bundled for this target, e.g. when the build hook did not run.
  static bool _probe() {
    try {
      tractVersion();
      return true;
    } on Object {
      return false;
    }
  }

  @override
  Future<void> initialize() async {
    if (!isAvailable) {
      throw const NoBackendAvailableException();
    }
  }

  @override
  Future<Model> loadOnnx(
    ModelSource source, {
    List<int>? inputShape,
    DatumType inputType = DatumType.f32,
  }) async {
    if (inputShape != null && inputShape.any((d) => d < 0)) {
      throw ArgumentError.value(
        inputShape,
        'inputShape',
        'dimensions must be non-negative',
      );
    }
    await initialize();
    final handle = switch (source) {
      FileSource(:final path) => _withUtf8(path, (cPath) {
        if (inputShape == null) return tractOnnxModelLoadPath(cPath);
        return _withShape(
          inputShape,
          (ptr, rank) =>
              tractOnnxModelLoadPathWithShape(cPath, inputType.code, ptr, rank),
        );
      }),
      BytesSource(:final bytes) => _withBytes(bytes, (data, len) {
        if (inputShape == null) return tractOnnxModelLoadBytes(data, len);
        return _withShape(
          inputShape,
          (ptr, rank) => tractOnnxModelLoadBytesWithShape(
            data,
            len,
            inputType.code,
            ptr,
            rank,
          ),
        );
      }),
    };
    if (handle == nullptr) throw ModelLoadException(_lastError());
    return FfiModel._(handle);
  }

  @override
  Future<Model> loadNnef(ModelSource source) async {
    await initialize();
    final handle = switch (source) {
      FileSource(:final path) => _withUtf8(path, tractNnefModelLoadPath),
      BytesSource(:final bytes) => _withBytes(bytes, tractNnefModelLoadBytes),
    };
    if (handle == nullptr) throw ModelLoadException(_lastError());
    return FfiModel._(handle);
  }

  @override
  Future<void> dispose() async {}
}

/// A model loaded into the native library.
class FfiModel implements Model {
  FfiModel._(this._handle) {
    _inputCount = tractModelInputCount(_handle);
    _outputCount = tractModelOutputCount(_handle);
    _inputNames = _readNames(tractModelInputName, _inputCount);
    _outputNames = _readNames(tractModelOutputName, _outputCount);
  }

  Pointer<Void> _handle;
  late final int _inputCount;
  late final int _outputCount;
  late final List<String> _inputNames;
  late final List<String> _outputNames;

  List<String> _readNames(
    Pointer<Utf8> Function(Pointer<Void>, int) fn,
    int count,
  ) => List<String>.generate(count, (i) {
    final ptr = fn(_handle, i);
    return ptr == nullptr ? '' : ptr.toDartString();
  });

  @override
  int get inputCount => _inputCount;

  @override
  int get outputCount => _outputCount;

  @override
  List<String> get inputNames => List.unmodifiable(_inputNames);

  @override
  List<String> get outputNames => List.unmodifiable(_outputNames);

  @override
  List<Tensor> run(List<Tensor> inputs) {
    if (_handle == nullptr) {
      throw const InferenceException('model has been disposed');
    }
    final handles = <Pointer<Void>>[];
    final outArray = calloc<Pointer<Void>>(
      _outputCount == 0 ? 1 : _outputCount,
    );
    final inArray = calloc<Pointer<Void>>(inputs.isEmpty ? 1 : inputs.length);
    try {
      for (final t in inputs) {
        handles.add(_makeValue(t));
      }
      for (var i = 0; i < handles.length; i++) {
        inArray[i] = handles[i];
      }
      final n = tractModelRun(
        _handle,
        inArray,
        inputs.length,
        outArray,
        _outputCount,
      );
      if (n < 0) throw InferenceException(_lastError());
      try {
        return [for (var i = 0; i < n; i++) _readValue(outArray[i])];
      } finally {
        // We own every output handle, including those after one that failed
        // to read.
        for (var i = 0; i < n; i++) {
          tractValueFree(outArray[i]);
        }
      }
    } finally {
      for (final h in handles) {
        tractValueFree(h);
      }
      calloc.free(inArray);
      calloc.free(outArray);
    }
  }

  Pointer<Void> _makeValue(Tensor t) {
    final shapePtr = t.rank == 0 ? nullptr : calloc<Size>(t.rank);
    final Pointer<Void> dataPtr = t.bytes.isEmpty
        ? nullptr
        : malloc<Uint8>(t.bytes.length).cast<Void>();
    try {
      for (var i = 0; i < t.rank; i++) {
        shapePtr[i] = t.shape[i];
      }
      if (t.bytes.isNotEmpty) {
        dataPtr.cast<Uint8>().asTypedList(t.bytes.length).setAll(0, t.bytes);
      }
      final v = tractValueFromBytes(
        t.datumType.code,
        shapePtr,
        t.rank,
        dataPtr,
        t.bytes.length,
      );
      if (v == nullptr) {
        throw InferenceException(_lastError());
      }
      return v;
    } finally {
      if (shapePtr != nullptr) calloc.free(shapePtr);
      if (dataPtr != nullptr) malloc.free(dataPtr);
    }
  }

  Tensor _readValue(Pointer<Void> v) {
    final dt = DatumType.fromCode(tractValueDatumType(v));
    final rank = tractValueRank(v);
    final shapePtr = calloc<Size>(rank == 0 ? 1 : rank);
    final shape = <int>[];
    try {
      tractValueShape(v, shapePtr, rank);
      for (var i = 0; i < rank; i++) {
        shape.add(shapePtr[i]);
      }
    } finally {
      calloc.free(shapePtr);
    }
    final byteLen = tractValueByteLen(v);
    final dataPtr = tractValueData(v);
    final bytes = Uint8List.fromList(dataPtr.asTypedList(byteLen));
    return Tensor.fromBytes(datumType: dt, shape: shape, bytes: bytes);
  }

  @override
  void dispose() {
    if (_handle != nullptr) {
      tractModelFree(_handle);
      _handle = nullptr;
    }
  }
}

// --- small allocation helpers ------------------------------------------------

T _withUtf8<T>(String s, T Function(Pointer<Utf8>) body) {
  final p = s.toNativeUtf8();
  try {
    return body(p);
  } finally {
    malloc.free(p);
  }
}

T _withBytes<T>(Uint8List bytes, T Function(Pointer<Void>, int) body) {
  final Pointer<Void> p = bytes.isEmpty
      ? nullptr
      : malloc<Uint8>(bytes.length).cast<Void>();
  try {
    if (bytes.isNotEmpty) {
      p.cast<Uint8>().asTypedList(bytes.length).setAll(0, bytes);
    }
    return body(p, bytes.length);
  } finally {
    if (p != nullptr) malloc.free(p);
  }
}

T _withShape<T>(List<int> shape, T Function(Pointer<Size>, int) body) {
  final p = shape.isEmpty ? nullptr.cast<Size>() : calloc<Size>(shape.length);
  try {
    for (var i = 0; i < shape.length; i++) {
      p[i] = shape[i];
    }
    return body(p, shape.length);
  } finally {
    if (p != nullptr) calloc.free(p);
  }
}
