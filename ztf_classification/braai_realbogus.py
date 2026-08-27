import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_braai(models_dir, base_name="braai_d6_m9"):
    """
    Load braai VGG6 model.
    We rebuild the architecture manually (braai's JSON was saved with Keras 2.2.4,
    incompatible with Keras3). Weights are loaded via h5py by name — the most
    robust path across TF/Keras versions.
    """
    import h5py

    model = keras.Sequential([
        keras.Input(shape=(63, 63, 3)),
        keras.layers.Conv2D(16, (3,3), activation='relu', name='conv1'),
        keras.layers.Conv2D(16, (3,3), activation='relu', name='conv2'),
        keras.layers.MaxPooling2D((2,2), name='max_pooling2d'),
        keras.layers.Dropout(0.25, name='dropout'),
        keras.layers.Conv2D(32, (3,3), activation='relu', name='conv3'),
        keras.layers.Conv2D(32, (3,3), activation='relu', name='conv4'),
        keras.layers.MaxPooling2D((4,4), name='max_pooling2d_1'),
        keras.layers.Dropout(0.25, name='dropout_1'),
        keras.layers.Flatten(name='flatten'),
        keras.layers.Dense(256, activation='relu', name='fc_1'),
        keras.layers.Dropout(0.5, name='dropout_2'),
        keras.layers.Dense(1, activation='sigmoid', name='fc_out'),
    ], name="VGG6")

    # Build the model so layers have allocated weights
    model.build((None, 63, 63, 3))

    # Load weights manually via h5py — works with old Keras2 HDF5 format
    weights_path = os.path.join(models_dir, f"{base_name}.weights.h5")
    with h5py.File(weights_path, 'r') as f:
        for layer in model.layers:
            lname = layer.name
            if lname not in f:
                continue                                  # dropout/pooling have no weights
            grp = f[lname]
            # Old Keras2 format nests weights under a sub-group with the same name
            if lname in grp:
                grp = grp[lname]
            # Keys stored as 'kernel:0', 'bias:0' — sort so kernel always loads first
            weight_names = sorted(grp.keys(), key=lambda k: (1 if 'bias' in k else 0))
            weight_values = [grp[k][()] for k in weight_names]
            layer.set_weights(weight_values)

    return model


def make_triplet(science, template, difference, l2_normalize=True):
    channels = []
    for data in (science, template, difference):
        data = np.nan_to_num(data.astype(np.float32))
        if l2_normalize:
            norm = np.linalg.norm(data)
            if norm > 0:
                data = data / norm
        h, w = data.shape
        if (h, w) != (63, 63):
            data = np.pad(
                data,
                [(0, 63 - h), (0, 63 - w)],
                mode="constant",
                constant_values=1e-9
            )
        channels.append(data)
    return np.stack(channels, axis=-1)


# give a score on how bogus or real the model is. Score if the cutout is real or not
def braai_score(model, triplet):
    return float(model.predict(np.expand_dims(triplet, 0), verbose=0)[0, 0])