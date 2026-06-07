// Classifies handwritten digits from the command line using `tract` (pure Dart,
// no Flutter), running the real MNIST ONNX model from the ONNX Model Zoo.
//
// Build the native library first:
//   cargo build --manifest-path native/tract_ffi/Cargo.toml
// then, from the repo root:
//   dart run packages/tract/example/tract_example.dart       # classify all, show digit 0
//   dart run packages/tract/example/tract_example.dart 7     # also render sample #7
//
// Each bundled sample is a real MNIST test image with its true label, so the
// prediction is checkable: the printed digit must match what you see and the
// label, and it could not be produced without actually running the network.
//
// If the library is elsewhere, point to it with TRACT_LIB=/path/to/lib.

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:tract/tract.dart';

Future<void> main(List<String> args) async {
  final modelPath = _find('mnist-8.onnx');
  final samplesPath = _find('mnist_samples.bin');
  if (modelPath == null || samplesPath == null) {
    stderr.writeln(
      'Could not find the bundled assets (mnist-8.onnx, '
      'mnist_samples.bin). Run from the repo root.',
    );
    exitCode = 2;
    return;
  }

  print('Active backend: ${Tract.backend.name}');
  // The model already declares a concrete float[1,1,28,28] input.
  final model = await Tract.loadOnnxFile(modelPath);
  print('Model: ${model.inputNames} -> ${model.outputNames}\n');

  final samples = _readSamples(File(samplesPath).readAsBytesSync());
  final show = args.isNotEmpty ? (int.tryParse(args.first) ?? 0) : 0;

  var correct = 0;
  for (var i = 0; i < samples.length; i++) {
    final s = samples[i];
    final input = Tensor.float32(
      [
        for (final p in s.pixels) p.toDouble(),
      ], // raw 0..255, as the model wants
      [1, 1, 28, 28],
    );
    final logits = model.run([input]).single.asFloat32List();
    final probs = _softmax(logits);
    final pred = _argmax(probs);
    if (pred == s.label) correct++;

    if (i == show) {
      print(_render(s.pixels));
    }
    final ok = pred == s.label ? 'OK ' : 'XX ';
    print(
      '$ok sample $i: true=${s.label}  predicted=$pred  '
      '(p=${(probs[pred] * 100).toStringAsFixed(1)}%)',
    );
  }
  print(
    '\nAccuracy on ${samples.length} real MNIST digits: '
    '$correct/${samples.length}',
  );
  model.dispose();
}

class _Sample {
  _Sample(this.label, this.pixels);
  final int label;
  final Uint8List pixels; // 784 bytes, row-major, grayscale 0..255
}

List<_Sample> _readSamples(Uint8List b) {
  const stride = 1 + 784;
  return [
    for (var o = 0; o + stride <= b.length; o += stride)
      _Sample(b[o], Uint8List.sublistView(b, o + 1, o + stride)),
  ];
}

/// Renders a 28x28 image compactly with Unicode half-blocks (28 cols, 14 rows).
String _render(Uint8List px) {
  final sb = StringBuffer();
  for (var r = 0; r < 28; r += 2) {
    for (var c = 0; c < 28; c++) {
      final top = px[r * 28 + c] > 96;
      final bot = r + 1 < 28 && px[(r + 1) * 28 + c] > 96;
      sb.write(
        top && bot
            ? '█'
            : top
            ? '▀'
            : bot
            ? '▄'
            : ' ',
      );
    }
    sb.writeln();
  }
  return sb.toString();
}

List<double> _softmax(List<double> xs) {
  final m = xs.reduce(math.max);
  final exps = [for (final x in xs) math.exp(x - m)];
  final sum = exps.reduce((a, b) => a + b);
  return [for (final e in exps) e / sum];
}

int _argmax(List<double> xs) {
  var best = 0;
  for (var i = 1; i < xs.length; i++) {
    if (xs[i] > xs[best]) best = i;
  }
  return best;
}

String? _find(String name) {
  for (final base in [
    'packages/tract/example/assets',
    'example/assets',
    'assets',
  ]) {
    final path = '$base/$name';
    if (File(path).existsSync()) return path;
  }
  return null;
}
