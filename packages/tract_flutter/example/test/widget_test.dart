// Smoke tests for the example app: it builds, shows its controls, and the
// board takes strokes.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:tract_flutter_example/main.dart';

void main() {
  testWidgets('renders the digit recognizer scaffold', (tester) async {
    await tester.pumpWidget(const TractApp());
    await tester.pump();

    expect(find.text('Recognize'), findsOneWidget);
    expect(find.text('Random sample'), findsOneWidget);
    expect(find.text('Clear'), findsOneWidget);
  });

  testWidgets('vertical strokes draw even when the page scrolls', (
    tester,
  ) async {
    // Landscape-phone sized: the page overflows, so its scroll view competes
    // for vertical drags.
    tester.view.physicalSize = const Size(800, 380);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(const TractApp());
    await tester.pump();

    final board = find.byWidgetPredicate(
      (w) =>
          w is CustomPaint &&
          w.painter.runtimeType.toString() == '_BoardPainter',
    );
    await tester.dragFrom(tester.getCenter(board), const Offset(0, 100));
    await tester.pump();

    final painter = tester.widget<CustomPaint>(board).painter;
    // The painter is private to the app; reach its strokes dynamically.
    expect((painter as dynamic).strokes, hasLength(1));
  });
}
