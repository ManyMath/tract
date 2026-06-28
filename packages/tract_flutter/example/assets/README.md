# Example assets

## `mnist-8.onnx`

The MNIST handwritten-digit classifier from the [ONNX Model Zoo][zoo]
(`validated/vision/classification/mnist`, opset 8), licensed Apache-2.0. Input
`Input3` is a `float[1,1,28,28]` image with pixel values in `0..255`; output
`Plus214_Output_0` is `float[1,10]` of class logits.

## `mnist_samples.bin`

Ten labeled digits (one per class `0..9`) taken from the [MNIST test set][mnist].
Layout is ten fixed-size records, one per digit in order `0..9`:

```
record = 1 byte label | 784 bytes pixels (28x28, row-major, grayscale 0..255)
```

The MNIST database is by Yann LeCun, Corinna Cortes and Christopher Burges and
is commonly redistributed under CC BY-SA 3.0.

[zoo]: https://github.com/onnx/models
[mnist]: http://yann.lecun.com/exdb/mnist/
