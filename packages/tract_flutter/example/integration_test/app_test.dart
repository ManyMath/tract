// End-to-end check on a real device or desktop: the app loads the native
// library bundled by tract's build hook and classifies digits with it.
//
//   flutter test integration_test -d <device>

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:tract_flutter_example/main.dart';

Future<void> _pumpUntil(WidgetTester tester, Finder finder) async {
  for (var i = 0; i < 300 && finder.evaluate().isEmpty; i++) {
    await tester.pump(const Duration(milliseconds: 100));
  }
  expect(finder, findsOneWidget);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('classifies the bundled MNIST digits', (tester) async {
    await tester.pumpWidget(const TractApp());
    await _pumpUntil(tester, find.text('tract MNIST - backend: ffi'));

    // "Random sample" steps through the bundled digits 0..9 in order.
    for (var digit = 0; digit < 10; digit++) {
      await tester.tap(find.text('Random sample'));
      await tester.pump();
      expect(
        find.textContaining(RegExp('^Predicted $digit .*true label $digit')),
        findsOneWidget,
      );
    }
  });

  testWidgets('recognizes a drawn digit', (tester) async {
    await tester.pumpWidget(const TractApp());
    await _pumpUntil(tester, find.text('tract MNIST - backend: ffi'));

    final board = find.byWidgetPredicate(
      (w) =>
          w is CustomPaint &&
          w.painter.runtimeType.toString() == '_BoardPainter',
    );
    final origin = tester.getTopLeft(board);
    await tester.dragFrom(
      origin + const Offset(150, 45),
      const Offset(-20, 195),
    );
    await tester.tap(find.text('Recognize'));
    await _pumpUntil(tester, find.textContaining(RegExp('^Predicted 1 ')));
  });
}
