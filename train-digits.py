#!/usr/bin/env python3
"""
One-time offline training script for digit recognition (Emily the Spy).
Run:   python train-digits.py
Output: model-digits/model.json + model-digits/group1-shard1of1.bin

Requires:  pip install tensorflow numpy
Data:      data/emnist-digits-train-images-idx3-ubyte
           data/emnist-digits-train-labels-idx1-ubyte
           (gunzip the .gz files first if you haven't already)

Writes model.json in TF.js 4 / Keras 2.x format directly — no manual
fix needed after training.
"""

import os
import json
import struct
import numpy as np
import tensorflow as tf

IMAGES_PATH = os.path.join('data', 'emnist-digits-train-images-idx3-ubyte')
LABELS_PATH = os.path.join('data', 'emnist-digits-train-labels-idx1-ubyte')
OUT_DIR     = 'model-digits'
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
    # EMNIST images are stored transposed
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
        tf.keras.layers.Dense(10, activation='softmax'),  # 10 classes: digits 0–9
    ])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


# Hardcoded TF.js-compatible Keras 2.x topology for this exact architecture.
# Bypasses model.to_json() which emits Keras 3 format that TF.js 4 can't load.
TFJS_TOPOLOGY = {
    "class_name": "Sequential",
    "config": {
        "name": "sequential",
        "layers": [
            {
                "class_name": "Conv2D",
                "config": {
                    "name": "conv2d", "trainable": True,
                    "batch_input_shape": [None, 28, 28, 1], "dtype": "float32",
                    "filters": 32, "kernel_size": [3, 3], "strides": [1, 1],
                    "padding": "same", "data_format": "channels_last",
                    "dilation_rate": [1, 1], "activation": "relu", "use_bias": True,
                    "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                    "bias_initializer": {"class_name": "Zeros", "config": {}},
                    "kernel_regularizer": None, "bias_regularizer": None,
                    "activity_regularizer": None, "kernel_constraint": None, "bias_constraint": None
                }
            },
            {
                "class_name": "MaxPooling2D",
                "config": {
                    "name": "max_pooling2d", "trainable": True, "dtype": "float32",
                    "pool_size": [2, 2], "padding": "valid", "strides": [2, 2],
                    "data_format": "channels_last"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "name": "conv2d_1", "trainable": True, "dtype": "float32",
                    "filters": 64, "kernel_size": [3, 3], "strides": [1, 1],
                    "padding": "same", "data_format": "channels_last",
                    "dilation_rate": [1, 1], "activation": "relu", "use_bias": True,
                    "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                    "bias_initializer": {"class_name": "Zeros", "config": {}},
                    "kernel_regularizer": None, "bias_regularizer": None,
                    "activity_regularizer": None, "kernel_constraint": None, "bias_constraint": None
                }
            },
            {
                "class_name": "MaxPooling2D",
                "config": {
                    "name": "max_pooling2d_1", "trainable": True, "dtype": "float32",
                    "pool_size": [2, 2], "padding": "valid", "strides": [2, 2],
                    "data_format": "channels_last"
                }
            },
            {
                "class_name": "Flatten",
                "config": {
                    "name": "flatten", "trainable": True, "dtype": "float32",
                    "data_format": "channels_last"
                }
            },
            {
                "class_name": "Dense",
                "config": {
                    "name": "dense", "trainable": True, "dtype": "float32",
                    "units": 128, "activation": "relu", "use_bias": True,
                    "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                    "bias_initializer": {"class_name": "Zeros", "config": {}},
                    "kernel_regularizer": None, "bias_regularizer": None,
                    "activity_regularizer": None, "kernel_constraint": None, "bias_constraint": None
                }
            },
            {
                "class_name": "Dropout",
                "config": {
                    "name": "dropout", "trainable": True, "dtype": "float32",
                    "rate": 0.3, "noise_shape": None, "seed": None
                }
            },
            {
                "class_name": "Dense",
                "config": {
                    "name": "dense_1", "trainable": True, "dtype": "float32",
                    "units": 10, "activation": "softmax", "use_bias": True,
                    "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                    "bias_initializer": {"class_name": "Zeros", "config": {}},
                    "kernel_regularizer": None, "bias_regularizer": None,
                    "activity_regularizer": None, "kernel_constraint": None, "bias_constraint": None
                }
            },
        ]
    },
    "keras_version": "2.9.0",
    "backend": "tensorflow"
}

# Explicit weight name mapping in the order Keras stores them.
# Dropout has no weights, so 8 weight tensors total.
WEIGHT_NAMES = [
    "conv2d/kernel",    # [3, 3, 1, 32]
    "conv2d/bias",      # [32]
    "conv2d_1/kernel",  # [3, 3, 32, 64]
    "conv2d_1/bias",    # [64]
    "dense/kernel",     # [3136, 128]
    "dense/bias",       # [128]
    "dense_1/kernel",   # [128, 10]
    "dense_1/bias",     # [10]
]


def save_tfjs_model(model, output_dir):
    """Write model-digits/model.json and the weights .bin in TF.js 4 format."""
    trainable = [w for w in model.weights if 'dropout' not in w.name and not w.name.endswith('_m0')]
    # Filter to only trainable kernel/bias weights (exclude optimizer state etc.)
    weight_tensors = [w for w in model.weights if w.trainable]

    assert len(weight_tensors) == len(WEIGHT_NAMES), (
        f"Expected {len(WEIGHT_NAMES)} weight tensors, got {len(weight_tensors)}. "
        "Check that the model architecture matches WEIGHT_NAMES."
    )

    weight_specs = []
    buffers = []
    for name, w in zip(WEIGHT_NAMES, weight_tensors):
        arr = w.numpy().astype(np.float32)
        weight_specs.append({'name': name, 'shape': list(arr.shape), 'dtype': 'float32'})
        buffers.append(arr.flatten().tobytes())

    weights_file = 'group1-shard1of1.bin'
    with open(os.path.join(output_dir, weights_file), 'wb') as f:
        for buf in buffers:
            f.write(buf)

    manifest = {
        'format': 'layers-model',
        'generatedBy': 'keras v2.9.0',
        'convertedBy': 'TensorFlow.js Converter v3.18.0',
        'modelTopology': TFJS_TOPOLOGY,
        'weightsManifest': [{'paths': [weights_file], 'weights': weight_specs}],
    }
    with open(os.path.join(output_dir, 'model.json'), 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    print('Loading EMNIST digits data…')
    images, rows, cols = load_idx3(IMAGES_PATH)
    labels = load_idx1(LABELS_PATH)

    n = min(len(images), TRAINING_N)
    images = fix_orientation(images[:n]).astype(np.float32) / 255.0
    images = images[..., np.newaxis]       # → (n, 28, 28, 1)
    labels = labels[:n].astype(np.int32)   # already 0–9, no offset needed

    print(f'Training on {n} samples…')
    model = build_model()
    model.fit(
        images, labels,
        epochs=EPOCHS,
        batch_size=512,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=2, restore_best_weights=True
        )],
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    save_tfjs_model(model, OUT_DIR)
    print(f'\nDone! Model saved to {OUT_DIR}/')
    print('Commit model-digits/ and push — the spy game will activate automatically.')


if __name__ == '__main__':
    main()
