import 'tensor.dart';

/// A loaded, runnable model.
///
/// Obtain one from [Tract.loadOnnx] / [Tract.loadNnef]. A model can be [run]
/// any number of times; call [dispose] to release the native resources when
/// done.
abstract class Model {
  /// The number of input tensors the model expects.
  int get inputCount;

  /// The number of output tensors the model produces.
  int get outputCount;

  /// The model's input names, in order.
  List<String> get inputNames;

  /// The model's output names, in order.
  List<String> get outputNames;

  /// Runs inference on [inputs] (one tensor per input, in order) and returns
  /// the output tensors.
  ///
  /// This is a synchronous, blocking call. Throws [InferenceException] if the
  /// inputs do not match what the model expects.
  List<Tensor> run(List<Tensor> inputs);

  /// Releases the native resources held by this model.
  void dispose();
}
