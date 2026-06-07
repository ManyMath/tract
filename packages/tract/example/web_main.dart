// A tiny web entry point used in CI to verify that `package:tract` compiles to
// JavaScript - i.e. that the conditional imports keep `dart:ffi`/`dart:io` out
// of the web build. There is no native inference backend on the web yet.

import 'package:tract/tract.dart';

void main() {
  // Selecting a backend throws when none is available (the web case).
  try {
    print('Active backend: ${Tract.backend.name}');
  } on NoBackendAvailableException catch (e) {
    print('No native inference on web: ${e.message}');
  }
}
