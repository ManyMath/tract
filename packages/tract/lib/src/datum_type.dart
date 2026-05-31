/// The element type of a [Tensor], mirroring tract's `DatumType`.
///
/// Each value carries a stable [code] shared with the native side and the
/// [byteSize] of a single element.
enum DatumType {
  f32(0, 4),
  f64(1, 8),
  f16(2, 2),
  i8(3, 1),
  i16(4, 2),
  i32(5, 4),
  i64(6, 8),
  u8(7, 1),
  u16(8, 2),
  u32(9, 4),
  u64(10, 8),
  boolean(11, 1);

  const DatumType(this.code, this.byteSize);

  /// The stable integer code shared with the native library.
  final int code;

  /// The size in bytes of a single element of this type.
  final int byteSize;

  /// The [DatumType] for a native [code]. Throws [ArgumentError] if unknown.
  static DatumType fromCode(int code) {
    for (final dt in values) {
      if (dt.code == code) return dt;
    }
    throw ArgumentError('unknown datum type code $code');
  }
}
