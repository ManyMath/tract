import 'dart:typed_data';

/// Describes a model to be loaded, independent of how a backend reads it.
///
/// This is a sealed hierarchy so backends can exhaustively switch over the
/// concrete kinds they support and surface a clear error for the rest.
sealed class ModelSource {
  const ModelSource();

  /// A model held in memory (e.g. a downloaded file or a bundled asset).
  const factory ModelSource.bytes(Uint8List bytes) = BytesSource;

  /// A model read from a file on disk by [path].
  ///
  /// For ONNX this is a `.onnx` file; for NNEF a directory or `.nnef.tgz`.
  const factory ModelSource.file(String path) = FileSource;
}

/// An in-memory model. See [ModelSource.bytes].
final class BytesSource extends ModelSource {
  /// Wraps in-memory model [bytes]. See [ModelSource.bytes].
  const BytesSource(this.bytes);

  /// The raw model bytes.
  final Uint8List bytes;
}

/// A model file on disk. See [ModelSource.file].
final class FileSource extends ModelSource {
  /// Points at a model file located at [path]. See [ModelSource.file].
  const FileSource(this.path);

  /// The path to the model file.
  final String path;
}
