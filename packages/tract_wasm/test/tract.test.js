import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { test, before } from 'node:test';
import { fileURLToPath } from 'node:url';

import { load, loadModel } from '../index.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const wasmPath = process.env.TRACT_WASM ?? here('../tract_ffi.wasm');
const assets = here('../../tract/example/assets/');
const STRIDE = 1 + 28 * 28;

let tract;
let onnx;
let samples;

before(async () => {
  assert.ok(
    existsSync(wasmPath),
    `${wasmPath} not found: run packages/tract/native/tool/build_wasm.sh or set TRACT_WASM`);
  onnx = readFileSync(assets + 'mnist-8.onnx');
  samples = readFileSync(assets + 'mnist_samples.bin');
  tract = await load(readFileSync(wasmPath));
});

const sample = (i) => {
  const rec = samples.subarray(i * STRIDE, (i + 1) * STRIDE);
  return { label: rec[0], pixels: new Uint8Array(rec.subarray(1)) };
};

test('wasm has no imports and reports a version', () => {
  const mod = new WebAssembly.Module(readFileSync(wasmPath));
  assert.deepEqual(WebAssembly.Module.imports(mod), []);
  assert.match(tract.version, /^\d+\.\d+\.\d+/);
});

for (const normalization of ['unit', 'mnist']) {
  test(`classifies the 10 MNIST samples (u8 pixels, ${normalization})`, () => {
    const model = loadModel(tract, onnx, { normalization });
    let correct = 0;
    for (let i = 0; i < 10; i++) {
      const { label, pixels } = sample(i);
      const result = model.classify28x28(pixels);
      assert.equal(result.probs.length, 10);
      assert.ok(Math.abs(result.probs.reduce((a, b) => a + b, 0) - 1) < 1e-4);
      if (result.label === label) correct++;
    }
    model.free();
    assert.equal(correct, 10);
  });
}

test('Float32Array pixels in raw units match the Uint8Array result', () => {
  const model = tract.loadModel(onnx, { normalization: 'mnist' });
  for (let i = 0; i < 10; i++) {
    const { label, pixels } = sample(i);
    const viaF32 = model.classify28x28(Float32Array.from(pixels));
    const viaU8 = model.classify28x28(pixels);
    assert.equal(viaF32.label, label);
    assert.deepEqual(viaF32.probs, viaU8.probs);
  }
  model.free();
});

test('run() returns the logits for a shaped input', () => {
  const model = tract.loadModel(onnx, { normalization: 'unit' });
  const { label, pixels } = sample(3);
  const logits = model.run(pixels, [1, 1, 28, 28]);
  assert.equal(logits.length, 10);
  assert.equal(logits.indexOf(Math.max(...logits)), label);
  model.free();
});

test('the invert option undoes inverted pixels', () => {
  const normal = tract.loadModel(onnx, { normalization: 'unit' });
  const inverted = tract.loadModel(onnx, { normalization: 'unit', invert: true });
  const { pixels } = sample(1);
  const flipped = pixels.map((v) => 255 - v);
  assert.deepEqual(inverted.classify28x28(flipped).probs, normal.classify28x28(pixels).probs);
  normal.free();
  inverted.free();
});

test('bad input is rejected with an error', () => {
  const model = tract.loadModel(onnx);
  assert.throws(() => model.classify28x28(new Uint8Array(10)), RangeError);
  assert.throws(() => model.run(new Float32Array(5), [1, 1, 28, 28]), /data length/);
  assert.throws(() => tract.loadModel(new Uint8Array([1, 2, 3, 4])), /loading the model failed/);
  assert.throws(() => tract.loadModel(onnx, { normalization: { std: 0 } }), /std/);
  assert.throws(() => tract.loadModel(onnx, { normalization: 'nope' }), RangeError);
  model.free();
});

test('a freed model throws and free() is idempotent', () => {
  const model = tract.loadModel(onnx);
  model.free();
  model.free();
  assert.throws(() => model.classify28x28(new Uint8Array(784)), /freed/);
});

test('repeated inference does not leak wasm memory', () => {
  const model = tract.loadModel(onnx, { normalization: 'unit' });
  const { pixels } = sample(0);
  model.classify28x28(pixels);
  const before = tract.exports.memory.buffer.byteLength;
  for (let i = 0; i < 200; i++) model.classify28x28(pixels);
  assert.equal(tract.exports.memory.buffer.byteLength, before);
  model.free();
});

test('a failing second stage frees the first staged buffer', () => {
  const real = tract.exports;
  let live = 0;
  const counting = {
    ...real,
    memory: real.memory,
    tract_alloc: (n) => { const p = real.tract_alloc(n); if (p) live++; return p; },
    tract_free: (p, n) => { live--; return real.tract_free(p, n); },
  };
  const model = tract.loadModel(onnx);
  tract.exports = counting;
  try {
    // A shape this long makes `new Uint32Array(shape)` throw after the data
    // buffer has already been staged.
    const hugeShape = { length: 2 ** 40 };
    const before = real.memory.buffer.byteLength;
    for (let i = 0; i < 500; i++) {
      assert.throws(() => model.run(new Uint8Array(784), hugeShape), RangeError);
      assert.throws(() => tract.loadModel(onnx, { inputShape: hugeShape }), RangeError);
    }
    assert.equal(live, 0);
    assert.equal(real.memory.buffer.byteLength, before);
  } finally {
    tract.exports = real;
    model.free();
  }
});

test('a wrong input shape is INVALID_INPUT, not an internal error', () => {
  const model = tract.loadModel(onnx, { normalization: 'unit' });
  for (const shape of [[784], [1, 784], [1, 1, 28, 27]]) {
    assert.throws(
      () => model.run(new Uint8Array(shape.reduce((a, b) => a * b)), shape),
      (e) => /status 2\b/.test(e.message) && /expects rank|does not match/.test(e.message),
      JSON.stringify(shape));
  }
  model.free();
});
