import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tract_flutter/tract_flutter.dart';

/// An [AssetBundle] that returns fixed bytes for any key, so asset loading can
/// be tested without a built asset bundle.
class _FakeBundle extends CachingAssetBundle {
  _FakeBundle(this._data);
  final Uint8List _data;

  @override
  Future<ByteData> load(String key) async => ByteData.sublistView(_data);
}

/// A backend that records what it was asked to load and returns a stub model.
class _FakeBackend extends TractBackend {
  final List<ModelSource> loaded = [];

  @override
  String get name => 'fake';

  @override
  bool get isAvailable => true;

  @override
  int get priority => 1000;

  @override
  Future<void> initialize() async {}

  @override
  Future<Model> loadOnnx(
    ModelSource source, {
    List<int>? inputShape,
    DatumType inputType = DatumType.f32,
  }) async {
    loaded.add(source);
    return _FakeModel();
  }

  @override
  Future<Model> loadNnef(ModelSource source) async {
    loaded.add(source);
    return _FakeModel();
  }

  @override
  Future<void> dispose() async {}
}

class _FakeModel implements Model {
  @override
  int get inputCount => 0;
  @override
  int get outputCount => 0;
  @override
  List<String> get inputNames => const [];
  @override
  List<String> get outputNames => const [];
  @override
  List<Tensor> run(List<Tensor> inputs) => const [];
  @override
  void dispose() {}
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() => Tract.reset());
  tearDown(() => Tract.reset());

  test(
    'loadOnnxAsset reads bundle bytes and dispatches to the backend',
    () async {
      final fake = _FakeBackend();
      Tract.registerBackend(fake, makeActive: true);

      final model = Uint8List.fromList(List<int>.generate(64, (i) => i % 256));
      await TractFlutter.loadOnnxAsset(
        'assets/model.onnx',
        bundle: _FakeBundle(model),
      );

      expect(fake.loaded, hasLength(1));
      final source = fake.loaded.single;
      expect(source, isA<BytesSource>());
      expect((source as BytesSource).bytes.length, 64);
    },
  );

  test('ensureInitialized returns the active backend name', () async {
    Tract.registerBackend(_FakeBackend(), makeActive: true);
    expect(await TractFlutter.ensureInitialized(), 'fake');
  });
}
