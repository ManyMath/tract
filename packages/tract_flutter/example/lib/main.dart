import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tract_flutter/tract_flutter.dart';

import 'digit_preprocess.dart';

void main() {
  runApp(const TractApp());
}

/// Size (logical px) of the square drawing board.
const double _boardSize = 280;

class TractApp extends StatelessWidget {
  const TractApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'tract MNIST',
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.indigo),
      home: const DigitPage(),
    );
  }
}

class DigitPage extends StatefulWidget {
  const DigitPage({super.key});

  @override
  State<DigitPage> createState() => _DigitPageState();
}

class _DigitPageState extends State<DigitPage> {
  String _backend = 'loading...';
  Model? _model;
  Uint8List? _samples; // the bundled real MNIST digits

  final List<List<Offset>> _strokes = [];
  Uint8List? _sample; // when set, the board shows a bundled 28x28 digit instead
  List<double>? _probs;
  String _note = 'Draw a digit (0-9), then Recognize.';

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      final name = await TractFlutter.ensureInitialized();
      final model = await TractFlutter.loadOnnxAsset('assets/mnist-8.onnx');
      final samples = await rootBundle.load('assets/mnist_samples.bin');
      if (!mounted) return;
      setState(() {
        _backend = name;
        _model = model;
        _samples = samples.buffer.asUint8List();
      });
    } on Object catch (e) {
      if (mounted) setState(() => _note = 'Failed to load model: $e');
    }
  }

  @override
  void dispose() {
    _model?.dispose();
    super.dispose();
  }

  int? get _prediction {
    final p = _probs;
    if (p == null) return null;
    var best = 0;
    for (var i = 1; i < p.length; i++) {
      if (p[i] > p[best]) best = i;
    }
    return best;
  }

  void _classify(Uint8List px28) {
    final model = _model;
    if (model == null) return;
    final input = Tensor.float32(
      [for (final p in px28) p.toDouble()], // raw 0..255 as the model expects
      [1, 1, 28, 28],
    );
    final logits = model.run([input]).single.asFloat32List();
    setState(() {
      _probs = _softmax(logits);
      _note =
          'Predicted ${_prediction!} '
          '(${(_probs![_prediction!] * 100).toStringAsFixed(1)}%).';
    });
  }

  Future<void> _recognize() async {
    if (_model == null) return;
    if (_sample != null) {
      _classify(_sample!);
      return;
    }
    if (_strokes.isEmpty) {
      setState(() => _note = 'Draw something first.');
      return;
    }
    final px = await rasterizeStrokes(_strokes, _boardSize);
    _classify(px);
  }

  void _clear() {
    setState(() {
      _strokes.clear();
      _sample = null;
      _probs = null;
      _note = 'Draw a digit (0-9), then Recognize.';
    });
  }

  void _randomSample() {
    final samples = _samples;
    if (samples == null) return;
    const stride = 1 + 784;
    final count = samples.length ~/ stride;
    // Vary by time-independent rotation through the digits.
    final pick = (_sampleCursor++) % count;
    final off = pick * stride;
    setState(() {
      _strokes.clear();
      _sample = Uint8List.sublistView(samples, off + 1, off + stride);
      _probs = null;
      _note = 'Bundled real MNIST digit (true label ${samples[off]}).';
    });
    _classify(_sample!);
  }

  int _sampleCursor = 0;

  @override
  Widget build(BuildContext context) {
    final pred = _prediction;
    return Scaffold(
      appBar: AppBar(title: Text('tract MNIST - backend: $_backend')),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _Board(
                strokes: _strokes,
                sample: _sample,
                onStart: (p) => setState(() {
                  _sample = null;
                  _strokes.add([p]);
                }),
                onUpdate: (p) => setState(() => _strokes.last.add(p)),
              ),
              const SizedBox(height: 12),
              Text(_note, style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              SizedBox(
                width: _boardSize,
                child: _Bars(probs: _probs, prediction: pred),
              ),
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                children: [
                  FilledButton(
                    onPressed: _model == null ? null : _recognize,
                    child: const Text('Recognize'),
                  ),
                  OutlinedButton(
                    onPressed: _model == null ? null : _randomSample,
                    child: const Text('Random sample'),
                  ),
                  TextButton(onPressed: _clear, child: const Text('Clear')),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

List<double> _softmax(List<double> xs) {
  final m = xs.reduce(math.max);
  final exps = [for (final x in xs) math.exp(x - m)];
  final sum = exps.reduce((a, b) => a + b);
  return [for (final e in exps) e / sum];
}

/// The 280x280 drawing board (black with white ink), or a bundled sample.
class _Board extends StatelessWidget {
  const _Board({
    required this.strokes,
    required this.sample,
    required this.onStart,
    required this.onUpdate,
  });

  final List<List<Offset>> strokes;
  final Uint8List? sample;
  final void Function(Offset) onStart;
  final void Function(Offset) onUpdate;

  Offset _clamp(Offset p) =>
      Offset(p.dx.clamp(0.0, _boardSize), p.dy.clamp(0.0, _boardSize));

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onPanStart: (d) => onStart(_clamp(d.localPosition)),
      onPanUpdate: (d) => onUpdate(_clamp(d.localPosition)),
      child: Container(
        width: _boardSize,
        height: _boardSize,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.indigo, width: 2),
        ),
        child: CustomPaint(
          painter: _BoardPainter(strokes, sample),
          size: const Size.square(_boardSize),
        ),
      ),
    );
  }
}

class _BoardPainter extends CustomPainter {
  _BoardPainter(this.strokes, this.sample);

  final List<List<Offset>> strokes;
  final Uint8List? sample;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = Colors.black);

    final sample = this.sample;
    if (sample != null) {
      final cw = size.width / 28, ch = size.height / 28;
      for (var y = 0; y < 28; y++) {
        for (var x = 0; x < 28; x++) {
          final v = sample[y * 28 + x];
          if (v > 0) {
            canvas.drawRect(
              Rect.fromLTWH(x * cw, y * ch, cw + 0.6, ch + 0.6),
              Paint()..color = Color.fromARGB(255, v, v, v),
            );
          }
        }
      }
      return;
    }

    final ink = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = _boardSize * 0.085
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    final dot = Paint()..color = Colors.white;
    for (final s in strokes) {
      if (s.isEmpty) continue;
      if (s.length == 1) {
        canvas.drawCircle(s.first, ink.strokeWidth / 2, dot);
        continue;
      }
      final path = Path()..moveTo(s.first.dx, s.first.dy);
      for (final p in s.skip(1)) {
        path.lineTo(p.dx, p.dy);
      }
      canvas.drawPath(path, ink);
    }
  }

  @override
  bool shouldRepaint(_BoardPainter old) => true;
}

/// A row of 10 probability bars, one per digit, predicted one highlighted.
class _Bars extends StatelessWidget {
  const _Bars({required this.probs, required this.prediction});

  final List<double>? probs;
  final int? prediction;

  @override
  Widget build(BuildContext context) {
    final probs = this.probs;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.end,
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        for (var d = 0; d < 10; d++)
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 16,
                height: 60 * (probs == null ? 0.0 : probs[d]) + 1,
                color: d == prediction ? Colors.indigo : Colors.indigo.shade100,
              ),
              Text(
                '$d',
                style: TextStyle(
                  fontWeight: d == prediction
                      ? FontWeight.bold
                      : FontWeight.normal,
                ),
              ),
            ],
          ),
      ],
    );
  }
}
