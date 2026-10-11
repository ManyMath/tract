import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../../datum_type.dart';
import '../../exceptions.dart';
import '../../model.dart';
import '../../model_source.dart';
import '../../tensor.dart';
import '../../tract_backend.dart';

/// The default native backend on platforms with `dart:ffi`.
TractBackend? createNativeBackend() => FfiBackend();

// --- C ABI signatures, mirroring native/tract_ffi/src/lib.rs -----------------

typedef _PtrC = Pointer<Void>;

typedef _ValueFromBytesNative =
    _PtrC Function(Int32, Pointer<Size>, Size, Pointer<Void>, Size);
typedef _ValueFromBytes =
    _PtrC Function(int, Pointer<Size>, int, Pointer<Void>, int);
typedef _FreeNative = Void Function(_PtrC);
typedef _Free = void Function(_PtrC);
typedef _IntQueryNative = Int32 Function(_PtrC);
typedef _IntQuery = int Function(_PtrC);
typedef _SizeQueryNative = Size Function(_PtrC);
typedef _SizeQuery = int Function(_PtrC);
typedef _ShapeNative = Size Function(_PtrC, Pointer<Size>, Size);
typedef _Shape = int Function(_PtrC, Pointer<Size>, int);
typedef _DataNative = Pointer<Uint8> Function(_PtrC);
typedef _Data = Pointer<Uint8> Function(_PtrC);
typedef _LoadPathNative = _PtrC Function(Pointer<Utf8>);
typedef _LoadPath = _PtrC Function(Pointer<Utf8>);
typedef _LoadPathShapeNative =
    _PtrC Function(Pointer<Utf8>, Int32, Pointer<Size>, Size);
typedef _LoadPathShape = _PtrC Function(Pointer<Utf8>, int, Pointer<Size>, int);
typedef _LoadBytesNative = _PtrC Function(Pointer<Void>, Size);
typedef _LoadBytes = _PtrC Function(Pointer<Void>, int);
typedef _LoadBytesShapeNative =
    _PtrC Function(Pointer<Void>, Size, Int32, Pointer<Size>, Size);
typedef _LoadBytesShape =
    _PtrC Function(Pointer<Void>, int, int, Pointer<Size>, int);
typedef _NameNative = Pointer<Utf8> Function(_PtrC, Size);
typedef _Name = Pointer<Utf8> Function(_PtrC, int);
typedef _RunNative =
    IntPtr Function(_PtrC, Pointer<_PtrC>, Size, Pointer<_PtrC>, Size);
typedef _Run = int Function(_PtrC, Pointer<_PtrC>, int, Pointer<_PtrC>, int);
typedef _LastErrorNative = Pointer<Utf8> Function();
typedef _LastError = Pointer<Utf8> Function();

/// Locates and opens the `tract_ffi` shared library.
class NativeLibrary {
  /// Explicit path override, highest priority. Set before first use.
  static String? overridePath;

  static DynamicLibrary? _opened;
  static bool _attempted = false;

  static String get _fileName => switch (Platform.operatingSystem) {
    'windows' => 'tract_ffi.dll',
    'macos' || 'ios' => 'libtract_ffi.dylib',
    _ => 'libtract_ffi.so',
  };

  /// Opens the library, caching the result. Returns `null` if it cannot be
  /// found/loaded - never throws.
  static DynamicLibrary? open() {
    if (_attempted) return _opened;
    _attempted = true;
    for (final load in _loaders()) {
      try {
        return _opened = load();
      } on Object {
        // Try the next candidate.
      }
    }
    return null;
  }

  static Iterable<DynamicLibrary Function()> _loaders() sync* {
    final override = overridePath;
    if (override != null) yield () => DynamicLibrary.open(override);
    final fromEnv = Platform.environment['TRACT_LIB'];
    if (fromEnv != null && fromEnv.isNotEmpty) {
      yield () => DynamicLibrary.open(fromEnv);
    }
    if (Platform.isMacOS || Platform.isIOS) {
      // In a Flutter app the library is -force_load'd into the plugin's
      // framework, or straight into the app binary when pods are linked
      // statically. Try those before the bare file name, which the loader
      // could resolve to an unrelated copy (e.g. in /usr/local/lib).
      yield () => DynamicLibrary.open('tract_flutter.framework/tract_flutter');
      yield () {
        final p = DynamicLibrary.process();
        p.lookup<NativeFunction<_LastErrorNative>>('tract_last_error');
        return p;
      };
    }
    // Resolved via the OS loader path / Flutter app bundle.
    yield () => DynamicLibrary.open(_fileName);
    // Developer builds of the in-repo crate, searched from the cwd upward.
    var dir = Directory.current.absolute;
    for (var i = 0; i < 6; i++) {
      for (final profile in ['release', 'debug']) {
        final path = '${dir.path}/native/tract_ffi/target/$profile/$_fileName';
        yield () => DynamicLibrary.open(path);
      }
      final parent = dir.parent;
      if (parent.path == dir.path) break;
      dir = parent;
    }
  }
}

/// Bound entry points into the native library.
class _Bindings {
  _Bindings(DynamicLibrary lib)
    : valueFromBytes = lib
          .lookupFunction<_ValueFromBytesNative, _ValueFromBytes>(
            'tract_value_from_bytes',
          ),
      valueFree = lib.lookupFunction<_FreeNative, _Free>('tract_value_free'),
      valueDatumType = lib.lookupFunction<_IntQueryNative, _IntQuery>(
        'tract_value_datum_type',
      ),
      valueRank = lib.lookupFunction<_SizeQueryNative, _SizeQuery>(
        'tract_value_rank',
      ),
      valueShape = lib.lookupFunction<_ShapeNative, _Shape>(
        'tract_value_shape',
      ),
      valueByteLen = lib.lookupFunction<_SizeQueryNative, _SizeQuery>(
        'tract_value_byte_len',
      ),
      valueData = lib.lookupFunction<_DataNative, _Data>('tract_value_data'),
      onnxLoadPath = lib.lookupFunction<_LoadPathNative, _LoadPath>(
        'tract_onnx_model_load_path',
      ),
      onnxLoadPathShape = lib
          .lookupFunction<_LoadPathShapeNative, _LoadPathShape>(
            'tract_onnx_model_load_path_with_shape',
          ),
      onnxLoadBytes = lib.lookupFunction<_LoadBytesNative, _LoadBytes>(
        'tract_onnx_model_load_bytes',
      ),
      onnxLoadBytesShape = lib
          .lookupFunction<_LoadBytesShapeNative, _LoadBytesShape>(
            'tract_onnx_model_load_bytes_with_shape',
          ),
      nnefLoadPath = lib.lookupFunction<_LoadPathNative, _LoadPath>(
        'tract_nnef_model_load_path',
      ),
      nnefLoadBytes = lib.lookupFunction<_LoadBytesNative, _LoadBytes>(
        'tract_nnef_model_load_bytes',
      ),
      modelFree = lib.lookupFunction<_FreeNative, _Free>('tract_model_free'),
      inputCount = lib.lookupFunction<_SizeQueryNative, _SizeQuery>(
        'tract_model_input_count',
      ),
      outputCount = lib.lookupFunction<_SizeQueryNative, _SizeQuery>(
        'tract_model_output_count',
      ),
      inputName = lib.lookupFunction<_NameNative, _Name>(
        'tract_model_input_name',
      ),
      outputName = lib.lookupFunction<_NameNative, _Name>(
        'tract_model_output_name',
      ),
      run = lib.lookupFunction<_RunNative, _Run>('tract_model_run'),
      lastError = lib.lookupFunction<_LastErrorNative, _LastError>(
        'tract_last_error',
      );

  final _ValueFromBytes valueFromBytes;
  final _Free valueFree;
  final _IntQuery valueDatumType;
  final _SizeQuery valueRank;
  final _Shape valueShape;
  final _SizeQuery valueByteLen;
  final _Data valueData;
  final _LoadPath onnxLoadPath;
  final _LoadPathShape onnxLoadPathShape;
  final _LoadBytes onnxLoadBytes;
  final _LoadBytesShape onnxLoadBytesShape;
  final _LoadPath nnefLoadPath;
  final _LoadBytes nnefLoadBytes;
  final _Free modelFree;
  final _SizeQuery inputCount;
  final _SizeQuery outputCount;
  final _Name inputName;
  final _Name outputName;
  final _Run run;
  final _LastError lastError;
}

/// Runs models by calling the native `tract_ffi` library over FFI.
class FfiBackend extends TractBackend {
  _Bindings? _bindings;
  bool _available = false;
  bool _probed = false;

  @override
  String get name => 'ffi';

  @override
  int get priority => 100;

  @override
  bool get isAvailable {
    if (_probed) return _available;
    _probed = true;
    final lib = NativeLibrary.open();
    if (lib == null) return _available = false;
    try {
      _bindings = _Bindings(lib);
      _available = true;
    } on Object {
      _available = false;
    }
    return _available;
  }

  @override
  Future<void> initialize() async {
    if (!isAvailable) {
      throw const NoBackendAvailableException(
        'The tract_ffi library could not be loaded.',
      );
    }
  }

  _Bindings get _b {
    if (!isAvailable) {
      throw const NoBackendAvailableException(
        'The tract_ffi library could not be loaded.',
      );
    }
    return _bindings!;
  }

  String lastError() {
    final ptr = _b.lastError();
    return ptr == nullptr ? 'unknown error' : ptr.toDartString();
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
    final handle = switch (source) {
      FileSource(:final path) => _withUtf8(path, (cPath) {
        if (inputShape == null) return _b.onnxLoadPath(cPath);
        return _withShape(
          inputShape,
          (ptr, rank) => _b.onnxLoadPathShape(cPath, inputType.code, ptr, rank),
        );
      }),
      BytesSource(:final bytes) => _withBytes(bytes, (data, len) {
        if (inputShape == null) return _b.onnxLoadBytes(data, len);
        return _withShape(
          inputShape,
          (ptr, rank) =>
              _b.onnxLoadBytesShape(data, len, inputType.code, ptr, rank),
        );
      }),
    };
    if (handle == nullptr) throw ModelLoadException(lastError());
    return FfiModel._(this, handle);
  }

  @override
  Future<Model> loadNnef(ModelSource source) async {
    final handle = switch (source) {
      FileSource(:final path) => _withUtf8(path, _b.nnefLoadPath),
      BytesSource(:final bytes) => _withBytes(
        bytes,
        (data, len) => _b.nnefLoadBytes(data, len),
      ),
    };
    if (handle == nullptr) throw ModelLoadException(lastError());
    return FfiModel._(this, handle);
  }

  @override
  Future<void> dispose() async {}
}

/// A model loaded into the native library.
class FfiModel implements Model {
  FfiModel._(this._backend, this._handle) {
    final b = _backend._b;
    _inputCount = b.inputCount(_handle);
    _outputCount = b.outputCount(_handle);
    _inputNames = _readNames(b.inputName, _inputCount);
    _outputNames = _readNames(b.outputName, _outputCount);
  }

  final FfiBackend _backend;
  Pointer<Void> _handle;
  late final int _inputCount;
  late final int _outputCount;
  late final List<String> _inputNames;
  late final List<String> _outputNames;

  List<String> _readNames(_Name fn, int count) =>
      List<String>.generate(count, (i) {
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
    final b = _backend._b;
    final handles = <Pointer<Void>>[];
    final outArray = calloc<Pointer<Void>>(
      _outputCount == 0 ? 1 : _outputCount,
    );
    final inArray = calloc<Pointer<Void>>(inputs.isEmpty ? 1 : inputs.length);
    try {
      for (final t in inputs) {
        handles.add(_makeValue(b, t));
      }
      for (var i = 0; i < handles.length; i++) {
        inArray[i] = handles[i];
      }
      final n = b.run(_handle, inArray, inputs.length, outArray, _outputCount);
      if (n < 0) throw InferenceException(_backend.lastError());
      try {
        return [for (var i = 0; i < n; i++) _readValue(b, outArray[i])];
      } finally {
        // We own every output handle, including those after one that failed
        // to read.
        for (var i = 0; i < n; i++) {
          b.valueFree(outArray[i]);
        }
      }
    } finally {
      for (final h in handles) {
        b.valueFree(h);
      }
      calloc.free(inArray);
      calloc.free(outArray);
    }
  }

  Pointer<Void> _makeValue(_Bindings b, Tensor t) {
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
      final v = b.valueFromBytes(
        t.datumType.code,
        shapePtr,
        t.rank,
        dataPtr,
        t.bytes.length,
      );
      if (v == nullptr) {
        throw InferenceException(_backend.lastError());
      }
      return v;
    } finally {
      if (shapePtr != nullptr) calloc.free(shapePtr);
      if (dataPtr != nullptr) malloc.free(dataPtr);
    }
  }

  Tensor _readValue(_Bindings b, Pointer<Void> v) {
    final dt = DatumType.fromCode(b.valueDatumType(v));
    final rank = b.valueRank(v);
    final shapePtr = calloc<Size>(rank == 0 ? 1 : rank);
    final shape = <int>[];
    try {
      b.valueShape(v, shapePtr, rank);
      for (var i = 0; i < rank; i++) {
        shape.add(shapePtr[i]);
      }
    } finally {
      calloc.free(shapePtr);
    }
    final byteLen = b.valueByteLen(v);
    final dataPtr = b.valueData(v);
    final bytes = Uint8List.fromList(dataPtr.asTypedList(byteLen));
    return Tensor.fromBytes(datumType: dt, shape: shape, bytes: bytes);
  }

  @override
  void dispose() {
    if (_handle != nullptr) {
      _backend._b.modelFree(_handle);
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
