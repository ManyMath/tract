@TestOn('vm')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:tract/tract.dart';
import 'package:test/test.dart';

/// Locates a committed asset regardless of whether tests run from the repo root
/// or the package directory.
String? _asset(String name) {
  for (final base in [
    'test/fixtures',
    'packages/tract/test/fixtures',
    'example/assets',
    'packages/tract/example/assets',
  ]) {
    final path = '$base/$name';
    if (File(path).existsSync()) return path;
  }
  return null;
}

void main() {
  setUp(() async => Tract.reset());
  tearDown(() async => Tract.reset());

  test('the build hook bundles the native library', () {
    expect(Tract.backend.name, 'ffi');
    expect(Tract.backend.isAvailable, isTrue);
  });

  test('runs the identity ONNX model end to end', () async {
    final fixture = _asset('identity.onnx');
    if (fixture == null) {
      markTestSkipped('identity.onnx fixture not found');
      return;
    }

    final model = await Tract.loadOnnxFile(fixture);
    addTearDown(model.dispose);

    expect(model.inputCount, 1);
    expect(model.outputCount, 1);

    final outputs = model.run([
      Tensor.float32([1.5, 2.5, 3.5], [3]),
    ]);
    expect(outputs, hasLength(1));
    expect(outputs.single.asFloat32List(), [1.5, 2.5, 3.5]);
  });

  test('classifies real MNIST digits correctly', () async {
    final modelPath = _asset('mnist-8.onnx');
    final samplesPath = _asset('mnist_samples.bin');
    if (modelPath == null || samplesPath == null) {
      markTestSkipped('MNIST assets not found');
      return;
    }

    final model = await Tract.loadOnnxFile(modelPath);
    addTearDown(model.dispose);

    // mnist_samples.bin: records of (1 byte label, 784 bytes pixels 0..255).
    final bytes = File(samplesPath).readAsBytesSync();
    const stride = 1 + 784;
    var count = 0;
    for (var o = 0; o + stride <= bytes.length; o += stride) {
      final label = bytes[o];
      final pixels = Uint8List.sublistView(bytes, o + 1, o + stride);
      final input = Tensor.float32(
        [for (final p in pixels) p.toDouble()],
        [1, 1, 28, 28],
      );
      final logits = model.run([input]).single.asFloat32List();
      expect(logits, hasLength(10));
      // argmax
      var pred = 0;
      for (var i = 1; i < 10; i++) {
        if (logits[i] > logits[pred]) pred = i;
      }
      expect(pred, label, reason: 'sample $count (true=$label) misclassified');
      count++;
    }
    expect(count, 10);
  });

  test('a negative inputShape dimension throws ArgumentError', () async {
    final fixture = _asset('identity.onnx');
    if (fixture == null) {
      markTestSkipped('identity.onnx fixture not found');
      return;
    }
    expect(
      () => Tract.loadOnnxFile(fixture, inputShape: [-1]),
      throwsArgumentError,
    );
  });

  test('loading garbage bytes throws ModelLoadException', () async {
    final garbage = Uint8List.fromList(List<int>.generate(32, (i) => i));
    expect(
      () => Tract.loadOnnxBytes(garbage),
      throwsA(isA<ModelLoadException>()),
    );
  });
}
