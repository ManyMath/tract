import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

/// High-resolution raster size the strokes are drawn at before being shrunk
/// to 28x28 (9x the target), so the inked bounding box is measured finely.
const int kRasterSize = 252;

/// Renders [strokes] (captured on a [boardSize] x [boardSize] board, white ink
/// on black) to a 28x28 grayscale image (0..255), normalized the way MNIST
/// models expect: the inked region is fit into a 20x20 box and centered by its
/// center of mass within the 28x28 frame.
///
/// Returns 784 bytes, row-major. An empty drawing yields all zeros.
Future<Uint8List> rasterizeStrokes(
  List<List<ui.Offset>> strokes,
  double boardSize,
) async {
  const r = kRasterSize;
  final recorder = ui.PictureRecorder();
  final canvas = ui.Canvas(recorder);
  canvas.drawRect(
    ui.Rect.fromLTWH(0, 0, r.toDouble(), r.toDouble()),
    ui.Paint()..color = const ui.Color(0xFF000000),
  );
  final sc = r / boardSize;
  final ink = ui.Paint()
    ..color = const ui.Color(0xFFFFFFFF)
    ..style = ui.PaintingStyle.stroke
    ..strokeWidth = boardSize * 0.085 * sc
    ..strokeCap = ui.StrokeCap.round
    ..strokeJoin = ui.StrokeJoin.round;
  final dot = ui.Paint()..color = const ui.Color(0xFFFFFFFF);
  for (final s in strokes) {
    if (s.isEmpty) continue;
    if (s.length == 1) {
      canvas.drawCircle(s.first * sc, ink.strokeWidth / 2, dot);
      continue;
    }
    final path = ui.Path()..moveTo(s.first.dx * sc, s.first.dy * sc);
    for (final p in s.skip(1)) {
      path.lineTo(p.dx * sc, p.dy * sc);
    }
    canvas.drawPath(path, ink);
  }
  final picture = recorder.endRecording();
  final image = await picture.toImage(r, r);
  picture.dispose();
  final bytes = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
  image.dispose();
  final rgba = bytes!.buffer.asUint8List();
  final gray = Uint8List(r * r);
  for (var i = 0; i < r * r; i++) {
    gray[i] = rgba[i * 4]; // white-on-black, so the R channel is the ink amount
  }
  return normalizeToMnist(gray, r);
}

/// Fits the inked region of an `r` x `r` grayscale buffer [g] into a 20x20 box
/// and centers it (by center of mass) in a fresh 28x28 image - the
/// normalization MNIST models were trained on. Returns 784 bytes, row-major.
Uint8List normalizeToMnist(Uint8List g, int r) {
  final out = Uint8List(28 * 28);
  int minX = r, minY = r, maxX = -1, maxY = -1;
  for (var y = 0; y < r; y++) {
    for (var x = 0; x < r; x++) {
      if (g[y * r + x] > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < minX) return out; // nothing drawn

  final bw = maxX - minX + 1;
  final bh = maxY - minY + 1;
  final scale = 20.0 / math.max(bw, bh);
  final dw = (bw * scale).round().clamp(1, 20);
  final dh = (bh * scale).round().clamp(1, 20);

  final tmp = Float64List(dw * dh);
  for (var ty = 0; ty < dh; ty++) {
    for (var tx = 0; tx < dw; tx++) {
      final sx = minX + (tx + 0.5) * bw / dw - 0.5;
      final sy = minY + (ty + 0.5) * bh / dh - 0.5;
      tmp[ty * dw + tx] = _bilinear(g, r, sx, sy);
    }
  }

  double mass = 0, cx = 0, cy = 0;
  for (var ty = 0; ty < dh; ty++) {
    for (var tx = 0; tx < dw; tx++) {
      final v = tmp[ty * dw + tx];
      mass += v;
      cx += v * tx;
      cy += v * ty;
    }
  }
  if (mass <= 0) return out;
  cx /= mass;
  cy /= mass;
  final offX = (14 - cx).round();
  final offY = (14 - cy).round();
  for (var ty = 0; ty < dh; ty++) {
    for (var tx = 0; tx < dw; tx++) {
      final ox = tx + offX, oy = ty + offY;
      if (ox >= 0 && ox < 28 && oy >= 0 && oy < 28) {
        out[oy * 28 + ox] = tmp[ty * dw + tx].round().clamp(0, 255);
      }
    }
  }
  return out;
}

double _bilinear(Uint8List g, int r, double x, double y) {
  final x0 = x.floor().clamp(0, r - 1);
  final y0 = y.floor().clamp(0, r - 1);
  final x1 = (x0 + 1).clamp(0, r - 1);
  final y1 = (y0 + 1).clamp(0, r - 1);
  final fx = (x - x0).clamp(0.0, 1.0);
  final fy = (y - y0).clamp(0.0, 1.0);
  final a = g[y0 * r + x0].toDouble();
  final b = g[y0 * r + x1].toDouble();
  final c = g[y1 * r + x0].toDouble();
  final d = g[y1 * r + x1].toDouble();
  return a * (1 - fx) * (1 - fy) +
      b * fx * (1 - fy) +
      c * (1 - fx) * fy +
      d * fx * fy;
}
