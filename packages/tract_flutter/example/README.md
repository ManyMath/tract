# tract_flutter_example

Draw a digit on the board and press **Recognize** to classify it with the MNIST
ONNX model from the ONNX Model Zoo, running on-device through `tract_flutter`.
**Random sample** steps through the bundled real MNIST test digits (see
[`assets/README.md`](assets/README.md)).

```sh
flutter run
```

Building needs a Rust toolchain (`rustup`); cargokit compiles the `tract_ffi`
crate as part of the Flutter build.
