/// Flutter integration for the `tract` package.
///
/// This package contributes the native `tract_ffi` library to your Flutter app
/// (built automatically via cargokit) and re-exports the full `tract` API.
/// In most cases you only need the re-exported [Tract] entry point:
///
/// ```dart
/// import 'package:tract_flutter/tract_flutter.dart';
///
/// await TractFlutter.ensureInitialized();
/// final model = await TractFlutter.loadOnnxAsset('assets/mobilenet.onnx',
///     inputShape: [1, 3, 224, 224]);
/// final outputs = model.run([Tensor.float32(pixels, [1, 3, 224, 224])]);
/// ```
library;

import 'dart:typed_data';

import 'package:flutter/services.dart' show AssetBundle, rootBundle;
import 'package:tract/tract.dart';

export 'package:tract/tract.dart';

/// Flutter-side conveniences over the `tract` registry.
abstract final class TractFlutter {
  /// Ensures a native backend is selected and ready.
  ///
  /// On native platforms `tract` already auto-registers the FFI backend that
  /// loads the bundled `tract_ffi` library; this initializes it eagerly so the
  /// first load has no setup latency, and surfaces load errors early. Returns
  /// the name of the active backend (e.g. `ffi`).
  static Future<String> ensureInitialized() async {
    final backend = Tract.backend;
    if (backend.isAvailable) {
      await backend.initialize();
    }
    return backend.name;
  }

  /// Loads an ONNX model from a bundled Flutter asset.
  ///
  /// [assetPath] is the key as declared in `pubspec.yaml` (e.g.
  /// `assets/model.onnx`). Pass [package] to load an asset that ships with
  /// another package, or [bundle] to read from a non-default [AssetBundle].
  /// See [Tract.loadOnnx] for [inputShape]/[inputType].
  static Future<Model> loadOnnxAsset(
    String assetPath, {
    String? package,
    AssetBundle? bundle,
    List<int>? inputShape,
    DatumType inputType = DatumType.f32,
  }) async {
    final bytes = await _loadAssetBytes(assetPath, package, bundle);
    return Tract.loadOnnxBytes(
      bytes,
      inputShape: inputShape,
      inputType: inputType,
    );
  }

  /// Loads a tract NNEF model (a `.nnef.tgz` archive) from a bundled asset.
  static Future<Model> loadNnefAsset(
    String assetPath, {
    String? package,
    AssetBundle? bundle,
  }) async {
    final bytes = await _loadAssetBytes(assetPath, package, bundle);
    return Tract.loadNnef(ModelSource.bytes(bytes));
  }

  static Future<Uint8List> _loadAssetBytes(
    String assetPath,
    String? package,
    AssetBundle? bundle,
  ) async {
    final key = package == null ? assetPath : 'packages/$package/$assetPath';
    final data = await (bundle ?? rootBundle).load(key);
    return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  }
}
