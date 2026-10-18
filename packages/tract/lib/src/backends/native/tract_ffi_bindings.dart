// Bindings to the tract_ffi C ABI in native/src/lib.rs.
//
// The build hook (hook/build.dart) compiles the crate and bundles it as a code
// asset whose ID is this library's URI, which is where @Native looks symbols
// up by default. Handles are opaque pointers owned by the native side.

import 'dart:ffi';

import 'package:ffi/ffi.dart';

@Native<Pointer<Utf8> Function()>(symbol: 'tract_last_error')
external Pointer<Utf8> tractLastError();

@Native<Pointer<Utf8> Function()>(symbol: 'tract_version')
external Pointer<Utf8> tractVersion();

// --- tensors -----------------------------------------------------------------

@Native<
  Pointer<Void> Function(Int32, Pointer<Size>, Size, Pointer<Void>, Size)
>(symbol: 'tract_value_from_bytes')
external Pointer<Void> tractValueFromBytes(
  int dtype,
  Pointer<Size> shape,
  int rank,
  Pointer<Void> data,
  int len,
);

@Native<Void Function(Pointer<Void>)>(symbol: 'tract_value_free')
external void tractValueFree(Pointer<Void> value);

@Native<Int32 Function(Pointer<Void>)>(symbol: 'tract_value_datum_type')
external int tractValueDatumType(Pointer<Void> value);

@Native<Size Function(Pointer<Void>)>(symbol: 'tract_value_rank')
external int tractValueRank(Pointer<Void> value);

@Native<Size Function(Pointer<Void>, Pointer<Size>, Size)>(
  symbol: 'tract_value_shape',
)
external int tractValueShape(Pointer<Void> value, Pointer<Size> out, int cap);

@Native<Size Function(Pointer<Void>)>(symbol: 'tract_value_byte_len')
external int tractValueByteLen(Pointer<Void> value);

@Native<Pointer<Uint8> Function(Pointer<Void>)>(symbol: 'tract_value_data')
external Pointer<Uint8> tractValueData(Pointer<Void> value);

// --- models ------------------------------------------------------------------

@Native<Pointer<Void> Function(Pointer<Utf8>)>(
  symbol: 'tract_onnx_model_load_path',
)
external Pointer<Void> tractOnnxModelLoadPath(Pointer<Utf8> path);

@Native<Pointer<Void> Function(Pointer<Utf8>, Int32, Pointer<Size>, Size)>(
  symbol: 'tract_onnx_model_load_path_with_shape',
)
external Pointer<Void> tractOnnxModelLoadPathWithShape(
  Pointer<Utf8> path,
  int dtype,
  Pointer<Size> shape,
  int rank,
);

@Native<Pointer<Void> Function(Pointer<Void>, Size)>(
  symbol: 'tract_onnx_model_load_bytes',
)
external Pointer<Void> tractOnnxModelLoadBytes(Pointer<Void> data, int len);

@Native<
  Pointer<Void> Function(Pointer<Void>, Size, Int32, Pointer<Size>, Size)
>(symbol: 'tract_onnx_model_load_bytes_with_shape')
external Pointer<Void> tractOnnxModelLoadBytesWithShape(
  Pointer<Void> data,
  int len,
  int dtype,
  Pointer<Size> shape,
  int rank,
);

@Native<Pointer<Void> Function(Pointer<Utf8>)>(
  symbol: 'tract_nnef_model_load_path',
)
external Pointer<Void> tractNnefModelLoadPath(Pointer<Utf8> path);

@Native<Pointer<Void> Function(Pointer<Void>, Size)>(
  symbol: 'tract_nnef_model_load_bytes',
)
external Pointer<Void> tractNnefModelLoadBytes(Pointer<Void> data, int len);

@Native<Void Function(Pointer<Void>)>(symbol: 'tract_model_free')
external void tractModelFree(Pointer<Void> model);

@Native<Size Function(Pointer<Void>)>(symbol: 'tract_model_input_count')
external int tractModelInputCount(Pointer<Void> model);

@Native<Size Function(Pointer<Void>)>(symbol: 'tract_model_output_count')
external int tractModelOutputCount(Pointer<Void> model);

@Native<Pointer<Utf8> Function(Pointer<Void>, Size)>(
  symbol: 'tract_model_input_name',
)
external Pointer<Utf8> tractModelInputName(Pointer<Void> model, int index);

@Native<Pointer<Utf8> Function(Pointer<Void>, Size)>(
  symbol: 'tract_model_output_name',
)
external Pointer<Utf8> tractModelOutputName(Pointer<Void> model, int index);

@Native<
  IntPtr Function(
    Pointer<Void>,
    Pointer<Pointer<Void>>,
    Size,
    Pointer<Pointer<Void>>,
    Size,
  )
>(symbol: 'tract_model_run')
external int tractModelRun(
  Pointer<Void> model,
  Pointer<Pointer<Void>> inputs,
  int inputCount,
  Pointer<Pointer<Void>> outputs,
  int outputCapacity,
);
