import 'dart:typed_data';

import 'datum_type.dart';

/// A dense, row-major tensor: an element [datumType], a [shape], and the raw
/// native-endian [bytes] backing it.
///
/// Tensors are the values passed to and returned from [Model.run]. The
/// `Tensor.<type>` constructors and `as<Type>List` views handle the common
/// numeric element types; [Tensor.fromBytes] covers any [DatumType].
class Tensor {
  /// Creates a tensor from raw native-endian [bytes].
  ///
  /// Throws [ArgumentError] if `bytes.length` does not match the product of
  /// [shape] times the element size of [datumType].
  Tensor.fromBytes({
    required this.datumType,
    required List<int> shape,
    required this.bytes,
  }) : shape = List.unmodifiable(shape) {
    final expected = _product(this.shape) * datumType.byteSize;
    if (bytes.lengthInBytes != expected) {
      throw ArgumentError(
        'data is ${bytes.lengthInBytes} bytes but shape ${this.shape} of '
        '${datumType.name} needs $expected',
      );
    }
  }

  /// The element type.
  final DatumType datumType;

  /// The dimensions, outermost first (e.g. `[1, 3, 224, 224]`).
  final List<int> shape;

  /// The contiguous, row-major, native-endian element data.
  final Uint8List bytes;

  /// Number of elements (the product of [shape]); 1 for a scalar.
  int get length => _product(shape);

  /// Rank (number of dimensions); 0 for a scalar.
  int get rank => shape.length;

  /// Builds an `f32` tensor from [values] laid out row-major as [shape].
  factory Tensor.float32(List<double> values, List<int> shape) =>
      Tensor.fromBytes(
        datumType: DatumType.f32,
        shape: shape,
        bytes: Float32List.fromList(values).buffer.asUint8List(),
      );

  /// Builds an `f64` tensor from [values] laid out row-major as [shape].
  factory Tensor.float64(List<double> values, List<int> shape) =>
      Tensor.fromBytes(
        datumType: DatumType.f64,
        shape: shape,
        bytes: Float64List.fromList(values).buffer.asUint8List(),
      );

  /// Builds an `i32` tensor from [values] laid out row-major as [shape].
  factory Tensor.int32(List<int> values, List<int> shape) => Tensor.fromBytes(
    datumType: DatumType.i32,
    shape: shape,
    bytes: Int32List.fromList(values).buffer.asUint8List(),
  );

  /// Builds an `i64` tensor from [values] laid out row-major as [shape].
  factory Tensor.int64(List<int> values, List<int> shape) => Tensor.fromBytes(
    datumType: DatumType.i64,
    shape: shape,
    bytes: Int64List.fromList(values).buffer.asUint8List(),
  );

  /// Builds a `u8` tensor from [values] laid out row-major as [shape].
  factory Tensor.uint8(List<int> values, List<int> shape) => Tensor.fromBytes(
    datumType: DatumType.u8,
    shape: shape,
    bytes: Uint8List.fromList(values),
  );

  /// A copy of the data viewed as `f32`. Throws [StateError] if [datumType] is
  /// not [DatumType.f32].
  Float32List asFloat32List() {
    _expect(DatumType.f32);
    return bytes.buffer.asFloat32List(bytes.offsetInBytes, length);
  }

  /// A copy of the data viewed as `f64`. Throws [StateError] if [datumType] is
  /// not [DatumType.f64].
  Float64List asFloat64List() {
    _expect(DatumType.f64);
    return bytes.buffer.asFloat64List(bytes.offsetInBytes, length);
  }

  /// A view of the data as `i32`. Throws [StateError] if [datumType] is not
  /// [DatumType.i32].
  Int32List asInt32List() {
    _expect(DatumType.i32);
    return bytes.buffer.asInt32List(bytes.offsetInBytes, length);
  }

  /// A view of the data as `i64`. Throws [StateError] if [datumType] is not
  /// [DatumType.i64].
  Int64List asInt64List() {
    _expect(DatumType.i64);
    return bytes.buffer.asInt64List(bytes.offsetInBytes, length);
  }

  void _expect(DatumType wanted) {
    if (datumType != wanted) {
      throw StateError('tensor is ${datumType.name}, not ${wanted.name}');
    }
  }

  @override
  String toString() => 'Tensor(${datumType.name}, shape: $shape)';

  static int _product(List<int> dims) => dims.fold(1, (a, b) => a * b);
}
