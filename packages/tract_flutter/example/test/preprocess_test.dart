import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:tract_flutter/tract_flutter.dart';
import 'package:tract_flutter_example/digit_preprocess.dart';

bool get _nativeAvailable {
  try {
    return Tract.backend.isAvailable;
  } on NoBackendAvailableException {
    return false;
  }
}

String? _modelPath() {
  for (final base in ['assets', 'packages/tract_flutter/example/assets']) {
    final p = '$base/mnist-8.onnx';
    if (File(p).existsSync()) return p;
  }
  return null;
}

int _predict(Model model, Uint8List px) {
  final logits = model
      .run([
        Tensor.float32([for (final p in px) p.toDouble()], [1, 1, 28, 28]),
      ])
      .single
      .asFloat32List();
  var best = 0;
  for (var i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

({double cx, double cy, int w, int h, double mass}) _stats(Uint8List px) {
  double mass = 0, cx = 0, cy = 0;
  int minX = 28, minY = 28, maxX = -1, maxY = -1;
  for (var y = 0; y < 28; y++) {
    for (var x = 0; x < 28; x++) {
      final v = px[y * 28 + x];
      if (v > 0) {
        mass += v;
        cx += v * x;
        cy += v * y;
        minX = math.min(minX, x);
        maxX = math.max(maxX, x);
        minY = math.min(minY, y);
        maxY = math.max(maxY, y);
      }
    }
  }
  return (
    cx: mass == 0 ? 0 : cx / mass,
    cy: mass == 0 ? 0 : cy / mass,
    w: maxX - minX + 1,
    h: maxY - minY + 1,
    mass: mass,
  );
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('normalizeToMnist', () {
    test('an empty buffer yields all zeros', () {
      final out = normalizeToMnist(Uint8List(64 * 64), 64);
      expect(out, hasLength(28 * 28));
      expect(out.every((v) => v == 0), isTrue);
    });

    test('an off-center blob is centered and fit within a 20px box', () {
      const r = 64;
      final g = Uint8List(r * r);
      // A 16x16 filled square pinned to the top-left corner.
      for (var y = 4; y < 20; y++) {
        for (var x = 4; x < 20; x++) {
          g[y * r + x] = 255;
        }
      }
      final s = _stats(normalizeToMnist(g, r));
      expect(s.mass, greaterThan(0));
      // center of mass moved to the middle of the 28x28 frame
      expect((s.cx - 14).abs(), lessThan(2.5));
      expect((s.cy - 14).abs(), lessThan(2.5));
      // and the ink was scaled to fit the 20px inner box
      expect(s.w, lessThanOrEqualTo(20));
      expect(s.h, lessThanOrEqualTo(20));
    });
  });

  group('rasterizeStrokes + MNIST model', () {
    late Model model;

    setUp(() async {
      if (!_nativeAvailable) return;
      final path = _modelPath();
      if (path == null) return;
      model = await Tract.loadOnnxFile(path);
      addTearDown(model.dispose);
    });

    tearDown(() async => Tract.reset());

    Future<void> expectStrokesRecognizedAs(
      int digit,
      List<List<Offset>> strokes,
    ) async {
      if (!_nativeAvailable || _modelPath() == null) {
        markTestSkipped('tract_ffi library / model asset not available');
        return;
      }
      final px = await rasterizeStrokes(strokes, 280);
      expect(px, hasLength(28 * 28));
      expect(px.any((v) => v > 0), isTrue, reason: 'nothing was rasterized');
      // Normalization must center the drawing's mass in the 28x28 frame.
      final s = _stats(px);
      expect((s.cx - 14).abs(), lessThan(4));
      expect((s.cy - 14).abs(), lessThan(4));
      expect(_predict(model, px), digit);
    }

    test('a drawn "7" is recognized as 7', () async {
      await expectStrokesRecognizedAs(7, [
        [const Offset(80, 60), const Offset(200, 60), const Offset(110, 235)],
      ]);
    });

    test('a drawn "1" is recognized as 1', () async {
      await expectStrokesRecognizedAs(1, [
        [const Offset(150, 45), const Offset(130, 240)],
      ]);
    });
  });
}
