#!/usr/bin/env python3
"""
One-time offline training script.
Run:   python train.py
Output: model/model.json + model/*.bin  (load with tf.loadLayersModel('./model/model.json'))

Requires:  pip install tensorflow tensorflowjs
Data:      data/emnist-letters-train-images-idx3-ubyte
           data/emnist-letters-train-labels-idx1-ubyte
"""

import os
import json
import struct
import numpy as np
import tensorflow as tf

IMAGES_PATH = os.path.join('data', 'emnist-letters-train-images-idx3-ubyte')
LABELS_PATH = os.path.join('data', 'emnist-letters-train-labels-idx1-ubyte')
OUT_DIR     = 'model'
TRAINING_N  = 30000
EPOCHS      = 20


def load_idx3(path):
    with open(path, 'rb') as f:
        magic, count, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 0x00000803, 'Bad IDX3 magic'
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols), rows, cols


def load_idx1(path):
    with open(path, 'rb') as f:
        magic, count = struct.unpack('>II', f.read(8))
        assert magic == 0x00000801, 'Bad IDX1 magic'
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def fix_orientation(images):
    # EMNIST images are stored transposed — match the JS fix
    return images.transpose(0, 2, 1)


def build_model():
    m = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(26, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def save_tfjs_model(model, output_dir):
    """Serialize a Keras model to TF.js layers-model format (no tensorflowjs needed)."""
    weight_specs = []
    buffers = []
    for w in model.weights:
        arr = w.numpy().astype(np.float32)
        name = w.name.removesuffix(':0')
        weight_specs.append({'name': name, 'shape': list(arr.shape), 'dtype': 'float32'})
        buffers.append(arr.flatten().tobytes())

    weights_file = 'group1-shard1of1.bin'
    with open(os.path.join(output_dir, weights_file), 'wb') as f:
        for buf in buffers:
            f.write(buf)

    manifest = {
        'format': 'layers-model',
        'generatedBy': 'keras',
        'convertedBy': None,
        'modelTopology': json.loads(model.to_json()),
        'weightsManifest': [{'paths': [weights_file], 'weights': weight_specs}],
    }
    with open(os.path.join(output_dir, 'model.json'), 'w') as f:
        json.dump(manifest, f)


def main():
    print('Loading EMNIST data…')
    images, rows, cols = load_idx3(IMAGES_PATH)
    labels = load_idx1(LABELS_PATH)

    n = min(len(images), TRAINING_N)
    images = fix_orientation(images[:n]).astype(np.float32) / 255.0
    images = images[..., np.newaxis]   # add channel dim → (n, 28, 28, 1)
    labels = labels[:n].astype(np.int32) - 1  # shift 1–26 → 0–25

    print(f'Training on {n} samples…')
    model = build_model()
    model.fit(
        images, labels,
        epochs=EPOCHS,
        batch_size=512,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)],
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    save_tfjs_model(model, OUT_DIR)
    print(f'\nDone! Model saved to {OUT_DIR}/')
    print('Commit the model/ directory and deploy alongside index.html.')


if __name__ == '__main__':
    main()
