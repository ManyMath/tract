# tract_flutter_example

Draw a digit on the board and press **Recognize** to classify it with the MNIST
ONNX model from the ONNX Model Zoo, running on-device through `tract_flutter`.
**Random sample** steps through the bundled real MNIST test digits (see
[`assets/README.md`](assets/README.md)).

```sh
flutter run
```

Building needs a Rust toolchain (`rustup`); `tract`'s build hook compiles the
`tract_ffi` crate as part of the Flutter build.

To check a platform end to end (it loads the model, classifies the bundled
digits and a drawn one):

```sh
flutter test integration_test -d <device>
```
