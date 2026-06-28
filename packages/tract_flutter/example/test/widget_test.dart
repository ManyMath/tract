// A basic smoke test for the example app: it builds and shows its controls.

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
}
