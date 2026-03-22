#!/usr/bin/env node
// One-time offline training script.
// Run: node train.js
// Outputs: model/model.json + model/*.bin
//
// Requires:  npm install @tensorflow/tfjs-node fs path
// Data:      data/emnist-letters-train-images-idx3-ubyte
//            data/emnist-letters-train-labels-idx1-ubyte

const tf   = require('@tensorflow/tfjs-node');
const fs   = require('fs');
const path = require('path');

const EMNIST_IMAGES = path.join(__dirname, 'data', 'emnist-letters-train-images-idx3-ubyte');
const EMNIST_LABELS = path.join(__dirname, 'data', 'emnist-letters-train-labels-idx1-ubyte');
const OUT_DIR       = path.join(__dirname, 'model');
const TRAINING_N    = 30000;
const EPOCHS        = 20;

function parseIDX3(buf) {
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (v.getUint32(0) !== 0x00000803) throw new Error('Bad IDX3 magic');
  return { data: buf.slice(16), count: v.getUint32(4), rows: v.getUint32(8), cols: v.getUint32(12) };
}

function parseIDX1(buf) {
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (v.getUint32(0) !== 0x00000801) throw new Error('Bad IDX1 magic');
  return { data: buf.slice(8), count: v.getUint32(4) };
}

function fixOrientation(src, rows, cols) {
  const dst = new Uint8Array(rows * cols);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      dst[r * cols + c] = src[c * cols + r];
  return dst;
}

function buildModel() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: 0.3 }));
  m.add(tf.layers.dense({ units: 26, activation: 'softmax' }));
  m.compile({ optimizer: 'adam', loss: 'sparseCategoricalCrossentropy', metrics: ['acc'] });
  return m;
}

async function main() {
  console.log('Loading EMNIST data…');
  const imgBuf = fs.readFileSync(EMNIST_IMAGES);
  const lblBuf = fs.readFileSync(EMNIST_LABELS);

  const imgs = parseIDX3(imgBuf);
  const lbls = parseIDX1(lblBuf);
  const n    = Math.min(imgs.count, TRAINING_N);
  const { rows, cols } = imgs;

  console.log(`Building tensors from ${n} samples…`);
  const flat = new Float32Array(n * rows * cols);
  for (let i = 0; i < n; i++) {
    const raw   = imgs.data.slice(i * rows * cols, (i + 1) * rows * cols);
    const fixed = fixOrientation(raw, rows, cols);
    for (let p = 0; p < fixed.length; p++)
      flat[i * rows * cols + p] = fixed[p] / 255;
  }
  const xs = tf.tensor4d(flat, [n, rows, cols, 1]);
  const ys = tf.tensor1d(Array.from(lbls.data.slice(0, n)).map(v => v - 1), 'float32');

  const m = buildModel();

  let bestValAcc = -Infinity, patience = 0;
  console.log('Training…');
  await m.fit(xs, ys, {
    epochs: EPOCHS,
    batchSize: 512,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`  Epoch ${epoch + 1}: acc=${(logs.acc * 100).toFixed(1)}%  val=${(logs.val_acc * 100).toFixed(1)}%`);
        if (logs.val_acc > bestValAcc) { bestValAcc = logs.val_acc; patience = 0; }
        else if (++patience >= 2) { m.stopTraining = true; }
      }
    }
  });

  xs.dispose();
  ys.dispose();

  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR);
  await m.save(`file://${OUT_DIR}`);
  console.log(`\nDone! Model saved to ${OUT_DIR}/`);
  console.log('Serve model/ alongside index.html and reload the game.');
}

main().catch(e => { console.error(e); process.exit(1); });
