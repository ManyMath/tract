/** Anything `load` can instantiate the wasm from. */
export type WasmSource =
  | ArrayBuffer
  | ArrayBufferView
  | WebAssembly.Module
  | Response
  | Promise<Response>
  | URL
  | string;

/**
 * Input normalization applied to every value `v` of a pixel input (`Uint8Array`
 * or raw-unit `Float32Array`): `((invert ? max - v : v) * scale - mean) / std`.
 */
export interface Normalization {
  /** Multiplier, e.g. `1 / 255`. Default 1. */
  scale?: number;
  /** Subtracted after scaling. Default 0. */
  mean?: number;
  /** Divisor after the mean; must be non-zero. Default 1. */
  std?: number;
  /** Top of the raw range, used by `invert`. Default 255. */
  max?: number;
  /** Replace `v` with `max - v` first (white-on-black vs black-on-white). */
  invert?: boolean;
}

/**
 * Named normalizations:
 * - `none`: values are passed through unchanged (default);
 * - `unit`: `x / 255`, which the bundled `mnist-8.onnx` works with;
 * - `mnist`: `(x / 255 - 0.1307) / 0.3081`, the usual MNIST statistics. The
 *   bundled `mnist-8.onnx` classifies equally well with `unit` and `mnist`,
 *   and noticeably worse with raw 0..255.
 */
export type NormalizationPreset = 'none' | 'unit' | 'mnist';

export interface LoadModelOptions {
  normalization?: NormalizationPreset | Normalization;
  /** Overrides the preset's/normalization's `invert`. */
  invert?: boolean;
  /** Pins input 0 to f32 with this shape (for models with symbolic dimensions). */
  inputShape?: number[];
}

export interface Classification {
  /** Index of the most probable class. */
  label: number;
  /** Softmax over the model's output logits. */
  probs: Float32Array;
}

export interface TractModel {
  /**
   * Runs the model on one input tensor and returns output 0 as floats.
   * `Uint8Array` input is treated as raw pixels; both kinds go through the
   * model's normalization. Throws on a shape/length mismatch or inference error.
   */
  run(input: Float32Array | Uint8Array, shape: number[]): Float32Array;
  /** Classifies a 28x28 image (784 pixels, raw `0..255` or `Float32Array`). */
  classify28x28(pixels: Uint8Array | Float32Array): Classification;
  /** Releases the model; later calls throw. Safe to call more than once. */
  free(): void;
}

export interface Tract {
  /** The tract_ffi crate version. */
  readonly version: string;
  /** The raw wasm exports (`tract_*` C ABI), for advanced use. */
  readonly exports: WebAssembly.Exports;
  /** Parses and optimizes an ONNX model once; run it many times. */
  loadModel(bytes: Uint8Array | ArrayBuffer, opts?: LoadModelOptions): TractModel;
}

/** Instantiates tract_ffi.wasm (it needs no imports). */
export function load(source: WasmSource): Promise<Tract>;

/** Same as `tract.loadModel(bytes, opts)`. */
export function loadModel(
  tract: Tract,
  bytes: Uint8Array | ArrayBuffer,
  opts?: LoadModelOptions,
): TractModel;
