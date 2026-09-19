import 'dart:typed_data';

import 'package:tract/tract.dart';
import 'package:test/test.dart';

void main() {
  group('DatumType', () {
    test('round-trips through its code', () {
      for (final dt in DatumType.values) {
        expect(DatumType.fromCode(dt.code), dt);
      }
    });

    test('throws on an unknown code', () {
      expect(() => DatumType.fromCode(999), throwsArgumentError);
    });
  });

  group('Tensor', () {
    test('float32 constructor and view round-trip', () {
      final t = Tensor.float32([1, 2, 3, 4, 5, 6], [2, 3]);
      expect(t.datumType, DatumType.f32);
      expect(t.shape, [2, 3]);
      expect(t.rank, 2);
      expect(t.length, 6);
      expect(t.asFloat32List(), [1, 2, 3, 4, 5, 6]);
    });

    test('int64 constructor and view round-trip', () {
      final t = Tensor.int64([10, 20, 30], [3]);
      expect(t.datumType, DatumType.i64);
      expect(t.asInt64List(), [10, 20, 30]);
    });

    test('scalar has rank 0 and length 1', () {
      final t = Tensor.float32([42], []);
      expect(t.rank, 0);
      expect(t.length, 1);
      expect(t.asFloat32List().single, 42);
    });

    test('rejects data that does not match the shape', () {
      expect(
        () => Tensor.fromBytes(
          datumType: DatumType.f32,
          shape: [2, 2],
          bytes: Uint8List(4),
        ),
        throwsArgumentError,
      );
    });

    test('rejects negative dimensions', () {
      // [-1, -1] multiplies out to 1, so the byte-length check alone passes.
      expect(
        () => Tensor.fromBytes(
          datumType: DatumType.f32,
          shape: [-1, -1],
          bytes: Uint8List(4),
        ),
        throwsArgumentError,
      );
    });

    test('views bytes that start at an unaligned offset', () {
      final f32 = Float32List.fromList([1.5, -2.5]).buffer.asUint8List();
      final unaligned = Uint8List(f32.length + 1)..setAll(1, f32);
      final t = Tensor.fromBytes(
        datumType: DatumType.f32,
        shape: [2],
        bytes: Uint8List.sublistView(unaligned, 1),
      );
      expect(t.asFloat32List(), [1.5, -2.5]);
    });

    test('a view of the wrong type throws', () {
      final t = Tensor.float32([1, 2], [2]);
      expect(t.asInt64List, throwsStateError);
    });
  });

  group('ModelSource', () {
    test('exposes its file path', () {
      const source = ModelSource.file('model.onnx');
      expect(source, isA<FileSource>());
      expect((source as FileSource).path, 'model.onnx');
    });

    test('exposes its bytes', () {
      final bytes = Uint8List.fromList([1, 2, 3]);
      final source = ModelSource.bytes(bytes);
      expect(source, isA<BytesSource>());
      expect((source as BytesSource).bytes, bytes);
    });
  });
}
