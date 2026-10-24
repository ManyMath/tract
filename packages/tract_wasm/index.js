// ESM wrapper around tract_ffi.wasm (the module has no imports, so it is
// instantiated with an empty import object).

const STATUS_OK = 0;
const PIXELS = 28 * 28;

const PRESETS = {
  none: { scale: 1, mean: 0, std: 1 },
  // x / 255, which is what the bundled mnist-8.onnx was trained on.
  unit: { scale: 1 / 255, mean: 0, std: 1 },
  // (x / 255 - 0.1307) / 0.3081, the usual MNIST statistics.
  mnist: { scale: 1 / 255, mean: 0.1307, std: 0.3081 },
};

/**
 * Instantiates tract_ffi.wasm.
 *
 * @param {import('./index.d.ts').WasmSource} source wasm bytes, a compiled
 *   `WebAssembly.Module`, a `Response` (streamed), or a URL/string to fetch.
 * @returns {Promise<import('./index.d.ts').Tract>}
 */
export async function load(source) {
  let instance;
  if (source instanceof WebAssembly.Module) {
    instance = await WebAssembly.instantiate(source, {});
  } else if (ArrayBuffer.isView(source) || source instanceof ArrayBuffer) {
    ({ instance } = await WebAssembly.instantiate(source, {}));
  } else {
    let response = await source;
    if (typeof response === 'string' || response instanceof URL) {
      response = await fetch(response);
    }
    if (!(response instanceof Response)) {
      throw new TypeError('load: expected wasm bytes, a Module, a Response or a URL');
    }
    if (!response.ok) {
      throw new Error(`load: fetching the wasm failed with HTTP ${response.status}`);
    }
    if (typeof WebAssembly.instantiateStreaming === 'function' &&
        response.headers.get('content-type') === 'application/wasm') {
      ({ instance } = await WebAssembly.instantiateStreaming(response, {}));
    } else {
      ({ instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {}));
    }
  }
  return new Tract(instance.exports);
}

/**
 * Loads an ONNX model into an already instantiated wasm module.
 *
 * @param {import('./index.d.ts').Tract} tract
 * @param {Uint8Array | ArrayBuffer} bytes
 * @param {import('./index.d.ts').LoadModelOptions} [opts]
 * @returns {import('./index.d.ts').TractModel}
 */
export function loadModel(tract, bytes, opts) {
  return tract.loadModel(bytes, opts);
}

/** A loaded wasm instance. */
class Tract {
  constructor(exports) {
    this.exports = exports;
    const ptr = exports.tract_version_ptr() >>> 0;
    this.version = new TextDecoder().decode(
      new Uint8Array(exports.memory.buffer, ptr, exports.tract_version_len() >>> 0));
  }

  /** Copies `bytes` into wasm memory; the caller must `release` it. */
  stage(bytes) {
    const e = this.exports;
    const view = toBytes(bytes);
    const ptr = view.length === 0 ? 0 : e.tract_alloc(view.length) >>> 0;
    if (view.length !== 0) {
      if (ptr === 0) throw new Error('tract: out of memory');
      // Read the buffer after alloc: allocating can grow (detach) the memory.
      new Uint8Array(e.memory.buffer, ptr, view.length).set(view);
    }
    return { ptr, len: view.length };
  }

  release(staged) {
    if (staged.ptr !== 0) this.exports.tract_free(staged.ptr, staged.len);
  }

  lastError() {
    const e = this.exports;
    const len = e.tract_last_error_len() >>> 0;
    const ptr = e.tract_last_error() >>> 0;
    return new TextDecoder().decode(new Uint8Array(e.memory.buffer, ptr, len));
  }

  /** Reads and frees a result buffer; throws with its message on error. */
  takeBuffer(buf) {
    const e = this.exports;
    const status = e.tract_buffer_status(buf);
    const len = e.tract_buffer_len(buf) >>> 0;
    const ptr = e.tract_buffer_data(buf) >>> 0;
    const data = new Uint8Array(e.memory.buffer, ptr, len).slice();
    e.tract_buffer_free(buf);
    if (status !== STATUS_OK) {
      throw new Error(`tract (status ${status}): ${new TextDecoder().decode(data)}`);
    }
    return new Float32Array(data.buffer, 0, len >>> 2);
  }

  /**
   * @param {Uint8Array | ArrayBuffer} bytes
   * @param {import('./index.d.ts').LoadModelOptions} [opts]
   */
  loadModel(bytes, opts = {}) {
    const e = this.exports;
    const norm = resolveNormalization(opts);
    const shape = opts.inputShape ?? [];
    const model = this.stage(bytes);
    const shapeBytes = this.stage(new Uint32Array(shape));
    let handle;
    try {
      handle = e.tract_model_load_with_options(
        model.ptr, model.len, norm.invert ? 1 : 0,
        norm.scale, norm.mean, norm.std, norm.max,
        shapeBytes.ptr, shape.length) >>> 0;
    } finally {
      this.release(model);
      this.release(shapeBytes);
    }
    if (handle === 0) throw new Error(`tract: loading the model failed: ${this.lastError()}`);
    return new TractModel(this, handle);
  }
}

class TractModel {
  #tract;
  #handle;

  constructor(tract, handle) {
    this.#tract = tract;
    this.#handle = handle;
  }

  #live() {
    if (this.#handle === 0) throw new Error('tract: the model was freed');
    return this.#handle;
  }

  /**
   * Runs the model on one input tensor and returns output 0 as floats.
   * @param {Float32Array | Uint8Array} input values in row-major order
   * @param {number[]} shape
   * @returns {Float32Array}
   */
  run(input, shape) {
    const handle = this.#live();
    const t = this.#tract;
    const e = t.exports;
    const isU8 = input instanceof Uint8Array;
    if (!isU8 && !(input instanceof Float32Array)) {
      throw new TypeError('run: input must be a Float32Array or Uint8Array');
    }
    const data = t.stage(input);
    const dims = t.stage(new Uint32Array(shape));
    try {
      const fn = isU8 ? e.tract_model_run_u8 : e.tract_model_run_f32;
      return t.takeBuffer(fn(handle, data.ptr, data.len, dims.ptr, shape.length));
    } finally {
      t.release(data);
      t.release(dims);
    }
  }

  /**
   * Classifies a 28x28 grayscale image with a 10-way output.
   * @param {Uint8Array | Float32Array} pixels 784 values
   * @returns {{label: number, probs: Float32Array}}
   */
  classify28x28(pixels) {
    const handle = this.#live();
    if (!(pixels instanceof Uint8Array) && !(pixels instanceof Float32Array)) {
      throw new TypeError('classify28x28: pixels must be a Uint8Array or Float32Array');
    }
    if (pixels.length !== PIXELS) {
      throw new RangeError(`classify28x28: expected ${PIXELS} pixels, got ${pixels.length}`);
    }
    const t = this.#tract;
    const staged = t.stage(pixels);
    let logits;
    try {
      logits = t.takeBuffer(t.exports.tract_infer_28x28(handle, staged.ptr, staged.len));
    } finally {
      t.release(staged);
    }
    if (logits.length < 2) throw new Error(`classify28x28: model output has ${logits.length} values`);
    const probs = softmax(logits);
    let label = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[label]) label = i;
    return { label, probs };
  }

  /** Releases the model; further calls throw. Safe to call twice. */
  free() {
    if (this.#handle !== 0) {
      this.#tract.exports.tract_model_free(this.#handle);
      this.#handle = 0;
    }
  }
}

function toBytes(data) {
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  throw new TypeError('expected an ArrayBuffer or typed array');
}

function resolveNormalization(opts) {
  const n = opts.normalization ?? 'none';
  const base = typeof n === 'string' ? PRESETS[n] : n;
  if (!base) throw new RangeError(`unknown normalization preset "${n}"`);
  return {
    scale: base.scale ?? 1,
    mean: base.mean ?? 0,
    std: base.std ?? 1,
    max: base.max ?? 255,
    invert: opts.invert ?? base.invert ?? false,
  };
}

function softmax(logits) {
  const out = new Float32Array(logits.length);
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += (out[i] = Math.exp(logits[i] - max));
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

